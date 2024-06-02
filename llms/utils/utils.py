import os
from pathlib import Path
import logging
import re
import time

import nltk
from rich.logging import RichHandler
from rich.progress import Progress, MofNCompleteColumn, SpinnerColumn
import textdistance

from .fulltext import convert


try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)
LOG_LEVEL_FINE = 15
logging.addLevelName(LOG_LEVEL_FINE, 'DETAIL')


def get_cache_dir(key=None):
    home_dir = os.getenv("HOME", ".")
    cache_dir = os.path.join(home_dir, ".cache", "llms")
    if key:
        cache_dir = os.path.join(cache_dir, key)
    return cache_dir


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


def get_output_path(
    output_dir,
    dataset_name,
    dataset_config=None,
    split=None,
    model_name=None,
    timestr=None,
    run_id=None,
):
    save_to = None
    if output_dir:
        dataset_name = full_dataset_name(dataset_name, dataset_config)
        save_subdir = dataset_name
        if split:
            save_subdir = f"{save_subdir}-{split}"
        if timestr:
            save_subdir = f"{save_subdir}_{timestr}"

        if run_id:
            save_subdir = f"{save_subdir}_{run_id}"

        if model_name:
            save_to = re.sub(r"[^\w\d]", "_", model_name)
            save_to = re.sub("^_+", "", save_to)
            save_to = os.path.join(output_dir, save_subdir, save_to)
        else:
            save_to = os.path.join(output_dir, save_subdir)
    return save_to


def get_log_path(
    log_dir,
    dataset_name,
    dataset_config,
    split=None,
    timestr=None,
    prefix=None,
    suffix=None,
):
    if log_dir:
        dataset_name = full_dataset_name(dataset_name, dataset_config)
        if prefix:
            log_path = f"{prefix}-{dataset_name}"
        else:
            log_path = dataset_name

        if split:
            log_path = f"{log_path}_{split}"

        if timestr:
            log_path = f"{log_path}_{timestr}"

        if suffix:
            log_path = f"{log_path}-{suffix}_log.txt"
        else:
            log_path = f"{log_path}_log.txt"

        log_path = os.path.join(log_dir, log_path)
        return log_path


def full_dataset_name(dataset_name, dataset_config):
    dataset_name = Path(dataset_name).stem
    if dataset_config:
        dataset_name = f"{dataset_name}_{dataset_config}"
    else:
        dataset_name = dataset_name
    return dataset_name


def config_logging(
    dataset_name="run",
    dataset_config=None,
    split=None,
    output_dir=None,
    prefix=None,
    run_id=None,
    **kwargs,
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
        level=int(os.environ.get("LOG_LEVEL", logging.INFO)),
        handlers=handlers,
        force=True,
    )
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    return timestr


def log(logger, message, level=LOG_LEVEL_FINE, max_length=300):
    if max_length and len(message) > max_length:
        message = f"{message[:max_length//2]} ... {message[-max_length//2:]}"
    logger.log(level, message)


def remove_article_abstract(text, abstract):
    if isinstance(text, list):
        text = "".join(text)
    paragraphs = text.split("\n\n")
    abstract_idx = None
    abstract = abstract.split(" ")
    for idx, par in enumerate(paragraphs):
        dist = textdistance.lcsseq.distance(par.split(" "), abstract)
        if dist < 0:
            abstract_idx = idx
        elif abstract_idx:
            break
    if abstract_idx:
        text = paragraphs[abstract_idx + 1 :]
        text = "\n\n".join(text)
    else:
        text = None
    return text


def clean_before_section(text, section="introduction"):
    """
    Removes content before section section using a simple heuristic.
    """
    if isinstance(text, list):
        lines = text
    else:
        lines = text.split("\n")

    idx = 0
    max_line_len = len(section.split()) + 1
    for line in lines:
        line = line.strip().lower()
        if section in line and len(line.split(" ")) <= max_line_len:
            break
        idx += 1
    text = None

    if len(lines) > idx:
        text = "\n".join(lines[idx:])
    return text


def pdf_to_text(pdf_path):
    try:
        outpath = convert(pdf_path)
        with open(outpath) as fh:
            text = fh.readlines()
            return text
    except RuntimeError as err:
        logger.error(f"Error converting PDF: {pdf_path}")
        logger.error(str(err))
