import os
from pathlib import Path
import logging
import re
import time

import nltk
import numpy as np
from rich.logging import RichHandler
from rich.progress import Progress, MofNCompleteColumn, SpinnerColumn
from scipy.stats import bootstrap


try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


def get_progress_bar(**kwargs):
    return Progress(
        SpinnerColumn(), *Progress.get_default_columns(), MofNCompleteColumn(), **kwargs
    )


def add_progress_task(
    progress, description, total=100.0, existing_ok=True, reset_existing=True
):
    tasks = [
        t_id for t_id, t in progress._tasks.items() if t.description == description
    ]
    task_exists = len(tasks) > 0

    if existing_ok or not task_exists:
        return progress.add_task(description, total=total)
    else:
        if reset_existing:
            progress.reset(tasks[0], visible=True)
        return tasks[0]


def word_tokenize(text):
    if isinstance(text, list):
        text = " ".join(text)
    words = nltk.word_tokenize(text)
    return words


def sent_tokenize(text):
    if type(text) == str:
        sents = nltk.sent_tokenize(text)
    else:
        sents = text
    return sents


def log_rouge_scores(scores):
    info = ["ROUGE scores:"]

    for k, v in sorted(scores.items()):
        if hasattr(v, "low"):
            score_info = [
                "%s-R: %f,%f,%f" % (k, v.low.recall, v.mid.recall, v.high.recall),
                "%s-P: %f,%f,%f"
                % (k, v.low.precision, v.mid.precision, v.high.precision),
                "%s-F: %f,%f,%f" % (k, v.low.fmeasure, v.mid.fmeasure, v.high.fmeasure),
            ]
        else:
            score_info = [
                "%s-R: %f,%f,%f" % (k, v.recall, v.recall, v.recall),
                "%s-P: %f,%f,%f" % (k, v.precision, v.precision, v.precision),
                "%s-F: %f,%f,%f" % (k, v.fmeasure, v.fmeasure, v.fmeasure),
            ]
        info.append("\n".join(score_info))
        info.append(" ")

    logger.info("\n".join(info))


def log_scores(name, scores):
    if len(scores) == 0:
        return

    if name == "rouge":
        log_rouge_scores(scores)
    else:
        info = [f"{name}:"]
        for key in scores.keys():
            if type(scores[key]) == dict:
                _scores = []
                for confidence_key in ["low", "mean", "high"]:
                    if isinstance(
                        scores[key][confidence_key], np.ndarray
                    ) or isinstance(scores[key][confidence_key], list):
                        score = [f"{x:.3f}" for x in scores[key][confidence_key]]
                        score = f"\n  {confidence_key}: {str(score)}"
                    else:
                        score = f"{scores[key][confidence_key]:.3f}"
                    _scores.append(score)

                _scores = ", ".join(_scores)
            else:
                _scores = f"{scores[key]:.3f}"
            info.append(f"{key}: {_scores}")
        info.append(" ")
        logger.info("\n".join(info))


def aggregate_scores(scores):
    if len(scores) == 1:
        return scores[0]
    elif len(scores) == 0:
        return {}

    agg_scores = {}

    for score in scores:
        for key in score.keys():
            key_scores = agg_scores.get(key, [])
            _score = score[key]
            if isinstance(_score, list) and len(_score) == 1:
                _score = _score[0]
            key_scores.append(_score)
            agg_scores[key] = key_scores

    confidence_intervals = {}
    for key in agg_scores.keys():
        ci = bootstrap(
            (agg_scores[key],),
            np.mean,
            vectorized=True,
            axis=0,
            confidence_level=0.95,
            random_state=17,
            method="BCa",
        )
        confidence_intervals[key] = {
            "low": ci.confidence_interval.low.tolist(),
            "high": ci.confidence_interval.high.tolist(),
            "mean": np.mean(agg_scores[key], axis=0).tolist(),
        }

    return confidence_intervals


def compute_metric(references, candidates, metric_fn, progress=None, **metric_kwargs):
    if progress is None:
        progress = get_progress_bar()

    results = []
    task = add_progress_task(
        progress,
        f"Computing {metric_fn.__name__}...",
        total=len(references),
        existing_ok=False,
    )
    with progress:
        for ref, cand in zip(references, candidates):
            results.append(
                metric_fn(references=[ref], candidates=[cand], **metric_kwargs)
            )
            progress.update(task, advance=1)
    return results


def get_output_path(
    output_dir,
    dataset_name,
    dataset_config,
    split,
    model_name=None,
    timestr=None,
    run_id=None,
):
    save_to = None
    if output_dir:
        dataset_name = full_dataset_name(dataset_name, dataset_config)
        save_subdir = f"{dataset_name}-{split}"
        if timestr:
            save_subdir = f"{save_subdir}_{timestr}"

        if run_id:
            save_subdir = f"{save_subdir}_{run_id}"

        if model_name:
            save_to = re.sub("[^\w\d]", "_", model_name)
            save_to = re.sub("^_+", "", save_to)
            save_to = os.path.join(output_dir, save_subdir, save_to)
        else:
            save_to = os.path.join(output_dir, save_subdir)
    return save_to


def get_log_path(
    log_dir,
    dataset_name,
    dataset_config,
    split,
    timestr=None,
    prefix=None,
    suffix=None,
):
    if log_dir:
        dataset_name = full_dataset_name(dataset_name, dataset_config)
        if prefix:
            log_path = f"{prefix}-{dataset_name}-{split}"
        else:
            log_path = f"{dataset_name}-{split}"

        if timestr:
            log_path = f"{log_path}_{timestr}"

        if suffix:
            log_path = f"{log_path}-{suffix}_log.txt"
        else:
            log_path = f"{log_path}_log.txt"

        log_path = os.path.join(log_dir, log_path)
        return log_path


def full_dataset_name(dataset_name, dataset_config):
    if dataset_config:
        dataset_name = f"{dataset_name}_{dataset_config}"
    else:
        dataset_name = dataset_name
    return dataset_name


def config_logging(
    dataset_name, dataset_config, split, output_dir, prefix=None, run_id=None
):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    log_dir = get_output_path(
        output_dir, dataset_name, dataset_config, split, timestr=timestr, run_id=run_id
    )
    if log_dir and Path(log_dir).is_dir():
        log_dir = Path(log_dir).parent

    log_path = get_log_path(
        log_dir,
        dataset_name,
        dataset_config,
        split,
        timestr=timestr,
        prefix=prefix,
    )
    handlers = [RichHandler()]
    if log_path:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w"))
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        handlers=handlers,
    )
    logging.getLogger("absl").setLevel(logging.WARNING)
    return timestr


def log(logger, message, verbose=False, max_length=300):
    level = logging.INFO if verbose else logging.DEBUG
    if max_length and len(message) > max_length:
        message = f"{message[:max_length//2]} ... {message[-max_length//2:]}"
    logger.log(level, message)
