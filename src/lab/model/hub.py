"""
Export a trained model of this project into a folder ready for the Hugging Face Hub.

The folder holds config.json (with auto_map), the weights, the model code
(configuration_*.py, modeling_*.py) and the tokenizer, so it loads with

    AutoModelForCausalLM.from_pretrained(path_or_repo, trust_remote_code=True)

and runs with lm-eval: lm_eval --model hf --model_args pretrained=<path_or_repo> ...

Usage:
    python -m lab.model.hub --checkpoint <trainer checkpoint dir> --tokenizer mistralai \
        --save_dir <output dir> [--repo_id <user>/<name> [--private]]

Pushing requires `hf auth login` (or the HF_TOKEN environment variable).
"""

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from lab.dataset.openwebtext import resolve_tokenizer
from lab.model import MODELS
from lab.utils.io import read_json

logger = logging.getLogger(__name__)

# model_type saved in config.json -> model class
MODEL_TYPES: dict[str, type[PreTrainedModel]] = {
    config_cls.model_type: model_cls for config_cls, model_cls in MODELS.values()
}


def export_for_hub(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path | str,
    repo_id: str | None = None,
    private: bool = True,
) -> Path:
    """
    Save the model with its code and tokenizer into save_dir and optionally push it.

    Parameters
    ----------
    model : PreTrainedModel
        Trained model of the lab.model registry.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer the model was trained with.
    save_dir : Path | str
        Output folder.
    repo_id : str | None
        Hub repository, e.g. 'user/llama3-owt'; nothing is pushed if None.
    private : bool
        Create the Hub repository as private.

    Returns
    -------
    Path
        The output folder.
    """
    if len(tokenizer) > model.config.vocab_size:
        raise ValueError(
            f'Tokenizer has {len(tokenizer)} tokens, more than the model vocab_size '
            f'{model.config.vocab_size}: it is not the tokenizer the model was trained with'
        )

    # Generation and lm-eval read the special tokens from the model config.
    special_tokens = {
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
    }
    for name, value in special_tokens.items():
        setattr(model.config, name, value)
        setattr(model.generation_config, name, value)

    save_dir = Path(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f'Saved model, code and tokenizer to {save_dir}')

    if repo_id is not None:
        model.push_to_hub(repo_id, private=private)  # ty:ignore[invalid-argument-type]
        tokenizer.push_to_hub(repo_id, private=private)
        logger.info(f'Pushed to https://huggingface.co/{repo_id}')
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--checkpoint', required=True, help='Folder saved by save_pretrained')
    parser.add_argument('--tokenizer', required=True, help='Short name or Hugging Face id')
    parser.add_argument('--save_dir', required=True, help='Output folder')
    parser.add_argument('--repo_id', default=None, help='Hub repository to push to')
    parser.add_argument('--private', action='store_true', help='Create a private repository')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = read_json(Path(args.checkpoint) / 'config.json')
    assert isinstance(config, dict)
    if config.get('model_type') not in MODEL_TYPES:
        raise ValueError(
            f'Unknown model_type {config.get("model_type")!r}, expected one of {list(MODEL_TYPES)}'
        )
    model = MODEL_TYPES[config['model_type']].from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer(args.tokenizer))
    export_for_hub(model, tokenizer, args.save_dir, args.repo_id, args.private)  # ty:ignore[invalid-argument-type]


if __name__ == '__main__':
    main()
