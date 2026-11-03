import os
from logging import Logger
from pathlib import Path

import comet_ml
from omegaconf import DictConfig

from lab.utils.io import get_root


def setup_comet(cfg: DictConfig, logger: Logger) -> None:
    if 'comet_ml' not in cfg.training_args.report_to:
        return

    comet_config = get_root() / '.comet.config'
    if comet_config.exists():
        os.environ.setdefault('COMET_CONFIG', str(comet_config))

    os.environ.setdefault('COMET_LOG_ASSETS', 'TRUE' if cfg.writer.log_assets else 'FALSE')
    os.environ.setdefault('COMET_START_ONLINE', 'TRUE' if cfg.writer.mode == 'online' else 'FALSE')

    resolved = comet_ml.get_config()
    if resolved['comet.api_key']:
        logger.info(
            f'Comet: project {resolved["comet.project_name"] or "(default)"}, '
            f'config {os.environ.get("COMET_CONFIG", "not from a file")}'
        )
    else:
        logger.warning(
            'Comet: no api_key found in .comet.config, .env or the environment. '
            'The run will fall back to an offline archive, or fail if Comet cannot '
            'start at all. Set report_to=None to disable logging entirely.'
        )


def log_final_model(model_dir: str | Path, name: str, logger: Logger) -> None:
    """
    Upload the final model saved by `Trainer.save_model` to the running Comet experiment.

    Only top-level files are uploaded, so intermediate `checkpoint-*` folders stored
    in the same directory are skipped. Does nothing if no Comet experiment is running.

    Parameters
    ----------
    model_dir : str | Path
        Directory the final model was saved to.
    name : str
        Model name in Comet.
    logger : Logger
        Logger for progress messages.
    """
    experiment = comet_ml.get_running_experiment()
    if experiment is None:
        return

    files = sorted(path for path in Path(model_dir).iterdir() if path.is_file())
    logger.info(f'Comet: uploading final model "{name}" ({len(files)} files from {model_dir})')
    for path in files:
        experiment.log_model(name, str(path))
