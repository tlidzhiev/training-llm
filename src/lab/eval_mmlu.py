"""
MMLU accuracy of the base model and the fine-tuned runs (the forgetting check).

The standard k-shot prompt of Hendrycks et al. (the dev split gives the shots); the prediction
is the letter among ' A', ' B', ' C', ' D' with the highest next-token logit after 'Answer:'.

Usage:
    python -m lab.eval_mmlu --model base --output results/mmlu/base.json
    python -m lab.eval_mmlu --model logs/training-llm/lora_base/checkpoints \
        --output results/mmlu/lora_base.json
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

from lab.model.load_finetuned import load_for_eval

logger = logging.getLogger(__name__)

LETTERS = ('A', 'B', 'C', 'D')
# MMLU subjects close to the MedQA domain: the in-domain part of the forgetting check.
MEDICAL_SUBJECTS = {
    'anatomy',
    'clinical_knowledge',
    'college_biology',
    'college_medicine',
    'medical_genetics',
    'nutrition',
    'professional_medicine',
    'virology',
}


def format_question(example: dict, with_answer: bool) -> str:
    text = example['question'].strip()
    for letter, choice in zip(LETTERS, example['choices'], strict=True):
        text += f'\n{letter}. {choice}'
    text += '\nAnswer:'
    if with_answer:
        text += f' {LETTERS[example["answer"]]}\n\n'
    return text


def build_prompts(
    num_fewshot: int, subjects: list[str] | None, limit_per_subject: int | None
) -> list[dict]:
    test = load_dataset('cais/mmlu', 'all', split='test')
    dev = load_dataset('cais/mmlu', 'all', split='dev')
    shots = defaultdict(list)
    for example in dev:
        shots[example['subject']].append(example)

    taken = defaultdict(int)
    prompts = []
    for example in test:
        subject = example['subject']
        if subjects is not None and subject not in subjects:
            continue
        if limit_per_subject is not None and taken[subject] >= limit_per_subject:
            continue
        taken[subject] += 1
        header = (
            'The following are multiple choice questions (with answers) about '
            f'{subject.replace("_", " ")}.\n\n'
        )
        fewshot = ''.join(format_question(s, True) for s in shots[subject][:num_fewshot])
        prompts.append(
            {
                'subject': subject,
                'text': header + fewshot + format_question(example, False),
                'answer': example['answer'],
            }
        )
    return prompts


def get_letter_ids(tokenizer) -> list[int]:
    ids = [tokenizer.encode(f' {letter}', add_special_tokens=False) for letter in LETTERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f'Answer letters must be single tokens, got {ids}')
    return [i[0] for i in ids]


@torch.no_grad()
def predict(model, tokenizer, prompts: list[dict], max_tokens: int, max_batch: int) -> list[int]:
    """
    Predicted option (0-3) for every prompt.

    Prompts are left-padded so the next-token logits are at the last position for every row;
    with rotary embeddings the shift of the positions does not change the attention.
    """
    letter_ids = torch.tensor(get_letter_ids(tokenizer))
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    encoded = [tokenizer.encode(p['text'], add_special_tokens=False) for p in prompts]
    order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i]))

    batches, batch = [], []
    for i in order:
        width = len(encoded[batch[0]]) if batch else len(encoded[i])
        if batch and ((len(batch) + 1) * width > max_tokens or len(batch) >= max_batch):
            batches.append(batch)
            batch = []
        batch.append(i)
    if batch:
        batches.append(batch)

    predictions = [-1] * len(prompts)
    device = next(model.parameters()).device
    for n, batch in enumerate(batches):
        width = max(len(encoded[i]) for i in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, i in enumerate(batch):
            ids = encoded[i]
            input_ids[row, width - len(ids) :] = torch.tensor(ids)
            attention_mask[row, width - len(ids) :] = 1
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            logits_to_keep=1,
        ).logits[:, -1, :]
        choice = logits[:, letter_ids.to(device)].argmax(dim=-1).tolist()
        for i, c in zip(batch, choice, strict=True):
            predictions[i] = c
        if n % 50 == 0:
            logger.info(f'batch {n + 1}/{len(batches)} (width {width}, size {len(batch)})')
    return predictions


def summarize(prompts: list[dict], predictions: list[int]) -> dict:
    per_subject = defaultdict(lambda: [0, 0])
    for p, pred in zip(prompts, predictions, strict=True):
        per_subject[p['subject']][0] += int(pred == p['answer'])
        per_subject[p['subject']][1] += 1

    def accuracy(subjects) -> float | None:
        correct = sum(per_subject[s][0] for s in subjects)
        total = sum(per_subject[s][1] for s in subjects)
        return correct / total if total else None

    medical = [s for s in per_subject if s in MEDICAL_SUBJECTS]
    other = [s for s in per_subject if s not in MEDICAL_SUBJECTS]
    return {
        'accuracy': accuracy(per_subject),
        'accuracy_macro': sum(c / t for c, t in per_subject.values()) / len(per_subject),
        'accuracy_medical': accuracy(medical),
        'accuracy_non_medical': accuracy(other),
        'num_examples': len(prompts),
        'num_medical': sum(per_subject[s][1] for s in medical),
        'per_subject': {s: c / t for s, (c, t) in sorted(per_subject.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', required=True, help="'base', a HF id or a checkpoint folder")
    parser.add_argument('--output', required=True, help='JSON file with the results')
    parser.add_argument('--num_fewshot', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=32768, help='Padded tokens per batch')
    parser.add_argument('--max_batch', type=int, default=64)
    parser.add_argument('--subjects', nargs='*', default=None, help='Only these subjects')
    parser.add_argument('--limit_per_subject', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start = time.time()
    model, tokenizer = load_for_eval(args.model)
    prompts = build_prompts(args.num_fewshot, args.subjects, args.limit_per_subject)
    logger.info(f'{len(prompts)} questions, {args.num_fewshot}-shot, model {args.model}')

    predictions = predict(model, tokenizer, prompts, args.max_tokens, args.max_batch)
    results = {
        'model': args.model,
        'num_fewshot': args.num_fewshot,
        **summarize(prompts, predictions),
        'runtime_s': time.time() - start,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding='utf8')
    logger.info(
        f'MMLU accuracy {results["accuracy"]:.4f} (medical {results["accuracy_medical"]}, '
        f'non-medical {results["accuracy_non_medical"]}) -> {output}'
    )


if __name__ == '__main__':
    main()
