import logging
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

LETTERS = ('A', 'B', 'C', 'D')
SYSTEM_PROMPT = (
    'You are a medical expert. Answer the multiple-choice question '
    'with the letter of the correct option.'
)
# our split name -> split name in bigbio/med_qa
SPLITS = {'train': 'train', 'val': 'validation', 'test': 'test'}


def build_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """
    Chat messages (system + user) with the question and the options of a MedQA example.
    """
    options = '\n'.join(f'{o["key"]}. {o["value"]}' for o in example['options'])
    question = (
        f'{example["question"]}\n\n{options}\n\nAnswer with the letter of the correct option.'
    )
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]


def get_letter_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """
    Token ids of the answer letters A, B, C, D (each must be a single token).
    """
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f'Answer letters must be single tokens, got {ids}')
    return [i[0] for i in ids]


def tokenize(example: dict[str, Any], tokenizer: Any, letter_ids: list[int]) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    # The eos token of Qwen2.5-Instruct is <|im_end|>, the end of the assistant turn.
    answer = f'{example["answer_idx"]}. {example["answer"]}{tokenizer.eos_token}'
    answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
    # The accuracy metric reads the logits of the letters at the first answer token.
    if answer_ids[0] != letter_ids[LETTERS.index(example['answer_idx'])]:
        raise ValueError(f'The answer does not start with its letter token: {answer!r}')

    input_ids = prompt_ids + answer_ids
    return {
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'labels': [-100] * len(prompt_ids) + answer_ids,  # loss only on the answer
    }


def build_medqa(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = 'bigbio/med_qa',
    config_name: str = 'med_qa_en_4options_source',
    max_seq_len: int = 1024,
    num_proc: int | None = 4,
    limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> dict[str, Dataset]:
    """
    Tokenized train / val / test splits of MedQA (the official splits are used).

    Only the train split is filtered by length, so every model is evaluated on exactly
    the same val and test examples.
    """
    raw = load_dataset(dataset_name, config_name)
    letter_ids = get_letter_ids(tokenizer)
    limits = {'train': limit, 'val': val_limit, 'test': test_limit}

    datasets = {}
    for name, hf_split in SPLITS.items():
        ds = raw[hf_split]
        split_limit = limits[name]
        if split_limit is not None:
            ds = ds.select(range(min(split_limit, len(ds))))
        ds = ds.map(
            tokenize,
            fn_kwargs={'tokenizer': tokenizer, 'letter_ids': letter_ids},
            remove_columns=ds.column_names,
            num_proc=num_proc,
            desc=f'Tokenizing {name}',
        )
        if name == 'train':
            before = len(ds)
            ds = ds.filter(lambda ids: len(ids) <= max_seq_len, input_columns='input_ids')
            logger.info(f'train: dropped {before - len(ds)} examples longer than {max_seq_len}')
        datasets[name] = ds
        logger.info(f'{name}: {len(ds)} examples')
    return datasets
