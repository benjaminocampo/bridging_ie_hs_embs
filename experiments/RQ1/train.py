from omegaconf import DictConfig, OmegaConf
from tempfile import TemporaryDirectory
from pathlib import Path
from usd_transformer import TransformerModel
from collections.abc import MutableMapping
from typing import Any, Dict

from sklearn.utils import shuffle
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split

import pandas as pd
import logging
import hydra
import mlflow
import shlex
import sys
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _flatten_dict_gen(d: MutableMapping, parent_key: str, sep: str):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep).items()
        elif isinstance(v, list) or isinstance(v, list):
            #  For lists we transform them into strings with a join
            yield new_key, "#".join(map(str, v))
        else:
            yield new_key, v

def flatten_dict(d: MutableMapping,
                 parent_key: str = '',
                 sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a dictionary using recursion (via an auxiliary funciton).
    The list/tuples values are flattened as a string.
    Parameters
    ----------
    d : MutableMapping
        Dictionary (or, more generally something that is a MutableMapping) to flatten.
        It might be nested, thus the function will traverse it to flatten it.
    parent_key : str
        Key of the parent dictionary in order to append to the path of keys.
    sep : str
        Separator to use in order to represent nested structures.
    Returns
    -------
    Dict[str, Any]
        The flattened dict where each nested dictionary is expressed as a path with
        the `sep`.
    >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': {'e': {'f': 3}}})
    {'a.b': 1, 'a.c': 2, 'd.e.f': 3}
    >>> flatten_dict({'a': {'b': [1, 2]}})
    {'a.b': '1#2'}
    """
    return dict(_flatten_dict_gen(d, parent_key, sep))

def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    """
    Script that finetunes/train a model.
    """
    with TemporaryDirectory() as tmpfile:
        output_dir = Path(tmpfile)

        logger.info("Command-line Arguments:")
        logger.info(
            f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}"
        )

        # Load train and dev datasets from `cfg.input.train_file` and `cfg.input.dev_file`.
        train = pd.read_parquet(cfg.input.train_file)
        dev = pd.read_parquet(cfg.input.dev_file)
        test = pd.read_parquet(cfg.input.test_file)

        # Separate in messages and labels.
        x_train, y_train = train["text"], train["label"].astype(int)
        x_dev, y_dev = dev["text"], dev["label"].astype(int)
        x_test, y_test = test["text"], test["label"].astype(int)

        # Use only a proportion of the train set.
        if cfg.input.train_size is not None:
            x_train, _, y_train, _ = train_test_split(
                x_train, y_train, train_size=cfg.input.train_size)

        # Shuffle train and dev sets.
        x_train, y_train = shuffle(
            x_train, y_train, random_state=cfg.train.model.params.random_state)
        x_dev, y_dev = shuffle(
            x_dev, y_dev, random_state=cfg.train.model.params.random_state)

        model = cfg.train.model.module(output_dir=output_dir,
                                       **cfg.train.model.params)

        if cfg.input.do_train:
            logger.info("training model...")

            # Fit using train and dev sets.
            model.fit(x_train=x_train,
                      y_train=y_train,
                      x_dev=x_dev,
                      y_dev=y_dev)

            logger.info("saving model...")

            model.save_model()

            logger.info("training finished succesfully.")

        if cfg.input.do_eval:
            y_pred_dev = model.predict(x_dev)
            y_pred_test = model.predict(x_test)

            # Calculate and log metrics.
            report = (
                f"**Classification results dev set**\n```\n{classification_report(y_pred=y_pred_dev, y_true=y_dev, digits=4)}```\n"
                +
                f"**Classification results test set**\n```\n{classification_report(y_pred=y_pred_test, y_true=y_test, digits=4)}```\n"
            )
            mlflow.set_tag("mlflow.note.content", report)

            mlflow.log_metric("accuracy_dev", accuracy_score(y_pred_dev, y_dev))
            mlflow.log_metric("precision_dev", precision_score(y_pred_dev, y_dev, average="macro"))
            mlflow.log_metric("recall_dev", recall_score(y_pred_dev, y_dev, average="macro"))
            mlflow.log_metric("f1_score_dev", f1_score(y_pred_dev, y_dev, average="macro"))

            mlflow.log_metric("accuracy_test", accuracy_score(y_pred_test, y_test))
            mlflow.log_metric("precision_test", precision_score(y_pred_test, y_test, average="macro"))
            mlflow.log_metric("recall_test", recall_score(y_pred_test, y_test, average="macro"))
            mlflow.log_metric("f1_score_test", f1_score(y_pred_test, y_test, average="macro"))

        mlflow.log_artifact(output_dir)


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver('eval', lambda x: eval(x))

    if cfg.input.uri_path is not None:
        mlflow.set_tracking_uri(cfg.input.uri_path)
        assert cfg.input.uri_path == mlflow.get_tracking_uri()

    logger.info(f"Current tracking uri: {cfg.input.uri_path}")

    mlflow.set_experiment(cfg.input.experiment_name)
    mlflow.set_experiment_tag('mlflow.note.content',
                              cfg.input.experiment_description)

    with mlflow.start_run(run_name=cfg.input.run_name) as run:
        logger.info("Logging configuration as artifact")
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            with open(config_path, "wt") as fh:
                print(OmegaConf.to_yaml(cfg, resolve=False), file=fh)
            mlflow.log_artifact(config_path)

        logger.info("Logging configuration parameters")
        # Log params expects a flatten dictionary, since the configuration has nested
        # configurations (e.g. train.model), we need to use flatten_dict in order to
        # transform it into something that can be easilty logged by MLFlow.
        mlflow.log_params(
            flatten_dict(OmegaConf.to_container(cfg, resolve=False)))
        run_experiment(cfg, run)


if __name__ == '__main__':
    main()
