import logging
import re
import time

from .models.huggingface import Text2TextSummarizer, CausalLMSummarizer
from .models.openai import OpenAISummarizer
from .models.cohere import CohereSummarizer
from .utils import get_progress_bar, add_progress_task

logger = logging.getLogger(__name__)


def summarizer_for_model(model_name, dataset_name=None):
    summarizer_map = {
        "gpt-[-\d\w]*": OpenAISummarizer,
        "facebook/opt-[\d\w]+": CausalLMSummarizer,
        "summarize-[\d\w]+": CohereSummarizer,
        ".*llama.*": CausalLMSummarizer,
    }

    for key, val in summarizer_map.items():
        if re.match(key, model_name):
            summarizer_class = val
            break
    else:
        summarizer_class = Text2TextSummarizer

    summarizer = summarizer_class(model_name, dataset_name=dataset_name)
    return summarizer


def predict_summaries(
    model_name_or_path,
    sources,
    dataset_name=None,
    max_length=256,
    cache_start=0,
    cache_end=None,
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
    summarizer = summarizer_for_model(model_name_or_path, dataset_name=dataset_name)

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
