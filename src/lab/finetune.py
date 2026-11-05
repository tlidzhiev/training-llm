from functools import partial

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from lab.dataset.medqa import build_medqa, get_letter_ids
from lab.logger.comet import log_final_model, setup_comet
from lab.metrics.accuracy import answer_letter_for_metrics, compute_accuracy
from lab.model.pretrained import get_pretrained_model
from lab.utils.init import setup_saving_and_logging
from lab.utils.io import get_root


@hydra.main(
    version_base='1.3',
    config_path=str(get_root() / 'src/lab/configs'),
    config_name='finetune',
)
def main(config: DictConfig) -> None:
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    """
    set_seed(config.training_args.seed)

    logger = setup_saving_and_logging(config)
    setup_comet(config, logger)

    logger.info(f'Config:\n{OmegaConf.to_yaml(config, resolve=True)}')

    training_args = TrainingArguments(**OmegaConf.to_container(config.training_args, resolve=True))  # ty:ignore[invalid-argument-type]
    logger.info(f'Device: {training_args.device}')

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    datasets = build_medqa(tokenizer, **config.dataset)  # ty:ignore[invalid-argument-type]
    logger.info(f'Datasets: {datasets}')

    lora_config = config.get('lora')  # None for full fine-tuning
    lora = None if lora_config is None else OmegaConf.to_container(lora_config, resolve=True)
    model = get_pretrained_model(**config.model, lora=lora)  # ty:ignore[invalid-argument-type]
    trainable = model.num_parameters(only_trainable=True)
    total = model.num_parameters()
    logger.info(
        f'Model:\n{model}\n'
        f'Mode: {"LoRA" if lora else "full fine-tuning"}\n'
        f'Parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable '
        f'({100 * trainable / total:.3f}%), '
        f'{model.num_parameters(exclude_embeddings=True) / 1e6:.2f}M non-embedding'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),  # ty:ignore[invalid-argument-type]
        compute_metrics=compute_accuracy,
        preprocess_logits_for_metrics=partial(
            answer_letter_for_metrics,
            letter_ids=get_letter_ids(tokenizer),  # ty:ignore[invalid-argument-type]
        ),
    )

    base_metrics = {}
    if config.resume_from_checkpoint is None:
        # Before training this is the original model: the LoRA B matrices are zeros.
        base_metrics = trainer.evaluate(datasets['test'], metric_key_prefix='test_base')  # ty:ignore[invalid-argument-type]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_output = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    cost = {
        'trainable_params': trainable,
        'total_params': total,
        'train_runtime_s': train_output.metrics['train_runtime'],
    }
    if torch.cuda.is_available():
        cost['peak_mem_allocated_gb'] = torch.cuda.max_memory_allocated() / 2**30
        cost['peak_mem_reserved_gb'] = torch.cuda.max_memory_reserved() / 2**30

    trainer.log(cost)
    logger.info(f'Cost: {cost}')

    trainer.save_model()  # LoRA: only the adapter
    trainer.save_state()

    test_metrics = trainer.evaluate(datasets['test'], metric_key_prefix='test')  # ty:ignore[invalid-argument-type]
    trainer.log_metrics('test', test_metrics)
    trainer.save_metrics('all', {**base_metrics, **test_metrics, **cost})

    log_final_model(training_args.output_dir, config.writer.run_name, logger)  # ty:ignore[invalid-argument-type]


if __name__ == '__main__':
    main()
