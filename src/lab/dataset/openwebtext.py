import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, Features, Value, load_dataset
from datasets import List as ListFeature
from transformers import AutoTokenizer

from lab.utils.io import get_root, read_json, write_json

logger = logging.getLogger(__name__)

TOKENIZERS = {
    'gpt2': 'openai-community/gpt2',
    'mistralai': 'mistralai/Mistral-7B-v0.1',
    'llama': 'meta-llama/Llama-2-7b-hf',
    'llama3': 'meta-llama/Meta-Llama-3-8B',
}

DATASET_NAME = 'ashaba1in/small_openwebtext'
SPLITS = ('train', 'val')


def resolve_tokenizer(tokenizer: str) -> str:
    return TOKENIZERS.get(tokenizer, tokenizer)


def tokens_dir(dataset_name: str, tokenizer: str, val_size: float, split_seed: int) -> Path:
    slug = f'{resolve_tokenizer(tokenizer)}-val{val_size}-seed{split_seed}'.replace('/', '__')
    return get_root() / 'data' / dataset_name.replace('/', '__') / slug


def tokenize(
    examples: dict[str, list],
    tokenizer: Any,
    bos_token_id: int | None,
    eos_token_id: int,
) -> dict[str, list]:
    outputs = tokenizer(examples['text'], add_special_tokens=False)
    prefix = [] if bos_token_id is None else [bos_token_id]
    ids = [prefix + doc + [eos_token_id] for doc in outputs['input_ids']]
    return {'ids': ids, 'len': [len(doc) for doc in ids]}


class TokenWindows(torch.utils.data.Dataset):
    def __init__(self, path: Path | str, dtype: str, max_seq_len: int, limit: int | None = None):
        if max_seq_len < 2:
            raise ValueError(f'max_seq_len must be at least 2, got {max_seq_len}')
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.max_seq_len = max_seq_len
        self.stride = max_seq_len - 1
        self.num_tokens = self.path.stat().st_size // self.dtype.itemsize

        num_windows = max(0, (self.num_tokens - max_seq_len) // self.stride + 1)
        self.num_windows = num_windows if limit is None else min(limit, num_windows)
        self._tokens: np.memmap | None = None

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.num_windows:
            raise IndexError(f'Window {index} is out of range [0, {self.num_windows})')
        if self._tokens is None:
            # Opened lazily, so every dataloader worker maps the file itself.
            self._tokens = np.memmap(self.path, dtype=self.dtype, mode='r')
        start = index * self.stride
        window = self._tokens[start : start + self.max_seq_len]
        input_ids = torch.from_numpy(window.astype(np.int64))

        # Labels equal input_ids: the model shifts them by one inside the loss, as
        # Hugging Face causal LMs do. The first label is never a target after the shift;
        # masking it makes Trainer's num_items_in_batch ((labels != -100).sum()) equal
        # the real number of targets, L - 1 per window, instead of L. Otherwise models
        # that use it (e.g. HF GPT-2) get their loss scaled by (L - 1) / L.
        labels = input_ids.clone()
        labels[0] = -100
        return {'input_ids': input_ids, 'labels': labels}

    def __getstate__(self) -> dict:
        # A pickled memmap would copy the whole file into every worker.
        state = self.__dict__.copy()
        state['_tokens'] = None
        return state


def prepare_tokens(
    tokenizer: str = 'mistralai',
    dataset_name: str = DATASET_NAME,
    val_size: float = 0.005,
    split_seed: int = 42,
    save_dir: Path | str | None = None,
    force_rebuild: bool = False,
    num_proc: int | None = None,
) -> Path:
    path = (
        Path(save_dir)
        if save_dir is not None
        else tokens_dir(dataset_name, tokenizer, val_size, split_seed)
    )
    if (path / 'meta.json').exists() and not force_rebuild:
        return path
    if not val_size:
        raise ValueError('val_size must be positive: models are compared on the val split')

    if num_proc is None:
        num_proc = max(1, (os.cpu_count() or 1) - 2)

    tokenizer_name = resolve_tokenizer(tokenizer)
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_token_id = hf_tokenizer.eos_token_id  # ty:ignore[unresolved-attribute]
    bos_token_id = hf_tokenizer.bos_token_id  # ty:ignore[unresolved-attribute]
    if eos_token_id is None:
        raise ValueError(f'Tokenizer {tokenizer_name} has no EOS token to separate documents')
    if bos_token_id == eos_token_id:
        bos_token_id = None

    vocab_size = len(hf_tokenizer)  # ty:ignore[invalid-argument-type]
    dtype = 'uint16' if vocab_size <= 2**16 else 'uint32'
    logger.info(f'Tokenizer {tokenizer_name}: vocab size {vocab_size} -> {dtype}')

    logger.info(f'Loading {dataset_name}...')
    dataset = load_dataset(dataset_name, split='train')
    split = dataset.train_test_split(
        test_size=int(val_size) if val_size >= 1 else val_size, seed=split_seed, shuffle=True
    )
    splits = DatasetDict({'train': split['train'], 'val': split['test']})

    logger.info(f'Tokenizing using {num_proc} CPU processes...')
    tokenized = splits.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={
            'tokenizer': hf_tokenizer,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id,
        },
        remove_columns=dataset.column_names,
        features=Features({'ids': ListFeature(Value(dtype)), 'len': Value('int64')}),
        desc='Tokenizing documents',
    )

    tmp_path = path.with_name(path.name + '.tmp')
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True)

    stats = {}
    for name in SPLITS:
        lengths = tokenized[name].with_format('numpy', columns=['len'])[:]['len']
        num_tokens = int(lengths.sum())
        stream = np.memmap(tmp_path / f'{name}.bin', dtype=dtype, mode='w+', shape=(num_tokens,))
        offset = 0
        for batch in tokenized[name].with_format('numpy', columns=['ids']).iter(10_000):
            chunk = np.concatenate(list(batch['ids']))
            stream[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        stream.flush()
        del stream
        stats[name] = {'num_documents': len(lengths), 'num_tokens': num_tokens}
        logger.info(f'{name}: {len(lengths)} documents, {num_tokens} tokens')

    write_json(
        {
            'dataset_name': dataset_name,
            'tokenizer': tokenizer_name,
            'vocab_size': vocab_size,
            'dtype': dtype,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id,
            'val_size': val_size,
            'split_seed': split_seed,
            'splits': stats,
        },
        tmp_path / 'meta.json',
    )

    shutil.rmtree(path, ignore_errors=True)
    tmp_path.rename(path)
    logger.info(f'Saved token streams to {path}')

    for name in SPLITS:
        for cache_file in tokenized[name].cache_files:
            Path(cache_file['filename']).unlink(missing_ok=True)
    return path


def build_openwebtext(
    max_seq_len: int = 1024,
    tokenizer: str = 'mistralai',
    dataset_name: str = DATASET_NAME,
    val_size: float = 0.005,
    split_seed: int = 42,
    save_dir: Path | str | None = None,
    force_rebuild: bool = False,
    num_proc: int | None = None,
    limit: int | None = None,
    val_limit: int | None = None,
) -> dict[str, TokenWindows]:
    path = prepare_tokens(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        val_size=val_size,
        split_seed=split_seed,
        save_dir=save_dir,
        force_rebuild=force_rebuild,
        num_proc=num_proc,
    )
    meta = read_json(path / 'meta.json')
    assert isinstance(meta, dict)

    limits = {'train': limit, 'val': val_limit}
    partitions = {
        name: TokenWindows(path / f'{name}.bin', meta['dtype'], max_seq_len, limits[name])
        for name in SPLITS
    }
    for name, windows in partitions.items():
        logger.info(
            f'{name}: {len(windows)} windows of {max_seq_len} tokens '
            f'({windows.num_tokens} tokens in the stream) from {path}'
        )
    return partitions
