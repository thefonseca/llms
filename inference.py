import logging
import re
import time

from .models.huggingface import (
    Text2TextSummarizer,
    CausalLMSummarizer,
    T5Summarizer,
    AlpacaSummarizer,
)
from .models.openai import OpenAISummarizer
from .models.cohere import CohereSummarizer
from .utils import get_progress_bar, add_progress_task

logger = logging.getLogger(__name__)


def summarizer_for_model(model_name, **kwargs):
    summarizer_map = {
        "gpt-[-\d\w]*": OpenAISummarizer,
        "facebook/opt-[\d\w]+": CausalLMSummarizer,
        ".*llama.*": CausalLMSummarizer,
        "bigscience/T0[_\d\w]*": T5Summarizer,
        "google/flan-t5[-\d\w]+": T5Summarizer,
        ".*alpaca.*": AlpacaSummarizer,
        "summarize-[\d\w]+": CohereSummarizer,
    }

    for key, val in summarizer_map.items():
        if re.match(key, model_name):
            summarizer_class = val
            break
    else:
        summarizer_class = Text2TextSummarizer

    summarizer = summarizer_class(model_name, **kwargs)
    logger.info(f"Using summarizer {summarizer}")
    return summarizer


def parse_kwargs(kwargs, model_prefix='model_'):
    model_kwargs = {}
    generation_kwargs = {}

    for key, value in kwargs.items():
        if key[:len(model_prefix)] == model_prefix:
            key = key[len(model_prefix):]
            model_kwargs[key] = value
        else:
            generation_kwargs[key] = value

    return model_kwargs, generation_kwargs


def predict_summaries(
    model_name_or_path,
    sources,
    max_length=256,
    cache_start=0,
    cache_end=None,
    **kwargs,
):
    summaries = []
    progress = get_progress_bar()
    task = add_progress_task(
        progress,
        f"Generating summaries for {model_name_or_path}...",
        total=len(sources),
        existing_ok=False,
    )
    cache_end = cache_end if cache_end is not None else len(sources)
    model_kwargs, generation_kwargs = parse_kwargs(kwargs)
    summarizer = summarizer_for_model(model_name_or_path, **model_kwargs)

    if hasattr(summarizer, "num_tokens_for_texts"):
        n_tokens = summarizer.num_tokens_for_texts(sources, max_length=max_length)
        logger.info(f"Total input tokens to be processed: {n_tokens}")

    with progress:
        for idx, text in enumerate(sources):
            ignore_cache = idx < cache_start or idx >= cache_end
            summary = summarizer.generate(
                text,
                max_length=max_length,
                memoizer_ignore_cache=ignore_cache,
                verbose=idx==0,
                **generation_kwargs,
            )
            summaries.append(summary)
            progress.update(task, advance=1)

            is_cache_hit = summarizer.is_last_result_from_cache()
            if (
                hasattr(summarizer, "request_interval")
                and summarizer.request_interval > 0
                and not is_cache_hit
            ):
                time.sleep(summarizer.request_interval)

    return summaries
