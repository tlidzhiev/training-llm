import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from lab.dataset import build_openwebtext, resolve_tokenizer
from lab.logger.comet import log_final_model, setup_comet
from lab.metrics import compute_metrics, token_loss_for_metrics
from lab.model import get_model
from lab.utils.init import setup_saving_and_logging
from lab.utils.io import get_root


@hydra.main(
    version_base='1.3',
    config_path=str(get_root() / 'src/lab/configs'),
    config_name='train',
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

    # Plain dicts and lists: nested DictConfig values (lr_scheduler_kwargs) are not JSON serializable.
    training_args = TrainingArguments(**OmegaConf.to_container(config.training_args, resolve=True))  # ty:ignore[invalid-argument-type]
    logger.info(f'Device: {training_args.device}')

    datasets = build_openwebtext(**config.dataset)
    logger.info(f'Datasets: {", ".join(f"{k}: {len(v)} windows" for k, v in datasets.items())}')

    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer(config.dataset.tokenizer))

    model = get_model(
        **config.model,
        tokenizer=tokenizer,  # ty:ignore[invalid-argument-type]
        max_seq_len=config.dataset.max_seq_len,
    )
    logger.info(
        f'Model:\n{model}\n'
        f'Parameters: {model.num_parameters() / 1e6:.2f}M total, '
        f'{model.num_parameters(exclude_embeddings=True) / 1e6:.2f}M non-embedding'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=token_loss_for_metrics,
    )

    if config.resume_from_checkpoint is None:
        trainer.evaluate()

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    trainer.save_model()
    trainer.save_state()

    val_metrics = trainer.evaluate()
    trainer.log_metrics('eval', val_metrics)
    trainer.save_metrics('eval', val_metrics)

    log_final_model(training_args.output_dir, config.writer.run_name, logger)  # ty:ignore[invalid-argument-type]


if __name__ == '__main__':
    main()
