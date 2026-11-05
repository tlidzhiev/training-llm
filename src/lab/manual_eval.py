"""
Answers of several models on the same random MedQA test questions, for the manual comparison.

Greedy generation with the chat prompt used in training; the report shows the full answers,
the gold option and whether each answer starts with the correct letter.

Usage:
    python -m lab.manual_eval --models base=base \
        full_ft=logs/training-llm/full_ft/checkpoints \
        lora_base=logs/training-llm/lora_base/checkpoints \
        --num_examples 8 --output results/manual_eval.md
"""

import argparse
import gc
import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset

from lab.dataset.medqa import build_messages
from lab.model.load_finetuned import load_for_eval

# The chosen option: a leading letter, 'answer is X', or the first 'X.' / 'X)'.
LETTER_PATTERNS = (
    re.compile(r'^\W*([ABCD])(?:[.):]|\s*$)'),
    re.compile(r'answer is:?\s*\(?([ABCD])\b', re.IGNORECASE),
    re.compile(r'\b([ABCD])[.)]'),
)


def extract_letter(answer: str) -> str | None:
    for pattern in LETTER_PATTERNS:
        match = pattern.search(answer)
        if match:
            return match.group(1).upper()
    return None


@torch.no_grad()
def generate(model, tokenizer, examples: list[dict], max_new_tokens: int) -> list[str]:
    answers = []
    device = next(model.parameters()).device
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            build_messages(example), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
        output = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs['input_ids'].shape[1] :]
        answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return answers


def cell(text: str) -> str:
    return text.replace('|', '\\|').replace('\n', ' / ')


def to_markdown(examples: list[dict], indices: list[int], answers: dict[str, list[str]]) -> str:
    names = list(answers)
    correct = {
        name: [
            extract_letter(a) == ex['answer_idx']
            for a, ex in zip(answers[name], examples, strict=True)
        ]
        for name in names
    }
    lines = [
        '# Ручная оценка на тесте MedQA',
        '',
        (
            f'{len(examples)} случайных вопросов из test (индексы {", ".join(map(str, indices))}), '
            'жадная генерация, промпт как при обучении. «Верно» = выбранная в ответе буква '
            '(первая буква, «answer is X» или первое «X.») совпадает с правильной.'
        ),
        '',
        '| Модель | Верных |',
        '| --- | --- |',
        *[f'| {name} | {sum(correct[name])} / {len(examples)} |' for name in names],
        '',
    ]
    for k, (index, example) in enumerate(zip(indices, examples, strict=True)):
        options = '  \n'.join(f'{o["key"]}. {o["value"]}' for o in example['options'])
        lines += [
            f'## Вопрос {k + 1} (test #{index}, {example["meta_info"]})',
            '',
            example['question'],
            '',
            options,
            '',
            f'**Правильный ответ:** {example["answer_idx"]}. {example["answer"]}',
            '',
            '| Модель | Ответ | Верно |',
            '| --- | --- | --- |',
            *[
                f'| {name} | {cell(answers[name][k])} | {"да" if correct[name][k] else "нет"} |'
                for name in names
            ],
            '',
        ]
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--models', nargs='+', required=True, help='name=path pairs; path=base ok')
    parser.add_argument('--num_examples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=96)
    parser.add_argument('--output', default='results/manual_eval.md')
    args = parser.parse_args()

    test = load_dataset('bigbio/med_qa', 'med_qa_en_4options_source', split='test')
    indices = sorted(random.Random(args.seed).sample(range(len(test)), args.num_examples))
    examples = [test[i] for i in indices]

    answers = {}
    for spec in args.models:
        name, path = spec.split('=', 1)
        model, tokenizer = load_for_eval(path)
        answers[name] = generate(model, tokenizer, examples, args.max_new_tokens)
        print(f'{name}: done')
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_markdown(examples, indices, answers), encoding='utf8')
    output.with_suffix('.json').write_text(
        json.dumps({'indices': indices, 'answers': answers}, ensure_ascii=False, indent=2),
        encoding='utf8',
    )
    print(f'Saved to {output}')


if __name__ == '__main__':
    main()
