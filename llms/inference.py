import logging
from pprint import pformat
import re
import time

from .models.huggingface import (
    Text2TextLM,
    CausalLM,
    LlamaChat,
    InstructText2TextLM,
    InstructCausalLM,
    Alpaca,
    Vicuna,
)
from .models.openai import OpenAIChat
from .utils.utils import get_progress_bar, add_progress_task

logger = logging.getLogger(__name__)

MODEL_CACHE = {}

DEFAULT_MODEL_MAP = {
    "gpt-[-\d\w]*": OpenAIChat,
    "facebook/opt-[\d\w]+": CausalLM,
    ".*llama-?2.*chat.*": LlamaChat,
    ".*llama.*": CausalLM,
    "bigscience/T0[_\d\w]*": InstructText2TextLM,
    "google/flan-t5[-\d\w]+": InstructText2TextLM,
    "google/long-t5[-\d\w]+": InstructText2TextLM,
    ".*alpaca.*": Alpaca,
    ".*vicuna.*": Vicuna,
    "mosaicml/mpt[-\d\w]$": CausalLM,
    "tiiuae/falcon[-\d\w]$": CausalLM,
    "mosaicml/mpt[-\d\w]+instruct": Alpaca,
    "tiiuae/falcon[-\d\w]+instruct": InstructCausalLM,
}


def parse_kwargs(kwargs, model_prefix="model_"):
    model_kwargs = {}
    generation_kwargs = {}

    for key, value in kwargs.items():
        if key[: len(model_prefix)] == model_prefix:
            key = key[len(model_prefix) :]
            model_kwargs[key] = value
        else:
            generation_kwargs[key] = value

    return model_kwargs, generation_kwargs


def get_model_class(model_name, model_map=None, default_class=None):
    if model_map is None:
        model_map = DEFAULT_MODEL_MAP

    for key, val in model_map.items():
        if re.match(key, model_name):
            summarizer_class = val
            break
    else:
        logger.warning(f"Could not match model '{model_name}' to generator class")
        if default_class:
            summarizer_class = default_class
        else:
            summarizer_class = Text2TextLM

    return summarizer_class


def generate(
    model_name,
    sources,
    model_class=None,
    max_length=256,
    cache_start=0,
    cache_end=None,
    use_model_cache=False,
    ignore_errors=False,
    show_progress=True,
    **kwargs,
):
    outputs = []
    progress = get_progress_bar()
    task = add_progress_task(
        progress,
        f"Generating outputs for {model_name}...",
        total=len(sources),
        existing_ok=False,
    )
    progress.update(task, visible=show_progress)
    cache_end = cache_end if cache_end is not None else len(sources)
    model_kwargs, generation_kwargs = parse_kwargs(kwargs)

    if model_class is None:
        model_class = get_model_class(model_name)

    logger.info(f"Using model: {model_class}")
    model = model_class(model_name, **model_kwargs)

    if use_model_cache:
        cached_model = MODEL_CACHE.get(model_name)
        if cached_model and hasattr(cached_model, "model"):
            model.model = cached_model.model
        MODEL_CACHE[model_name] = model

    if hasattr(model, "token_statistics"):
        stats = model.token_statistics(
            sources,
            max_length=max_length,
            show_progress=show_progress,
            **generation_kwargs,
        )
        logger.info(f"Token statistics for input:\n{pformat(stats)}")

    with progress:
        for idx, text in enumerate(sources):
            ignore_cache = idx < cache_start or idx >= cache_end
            try:
                output = model.generate(
                    text,
                    max_length=max_length,
                    memoizer_ignore_cache=ignore_cache,
                    verbose=idx == 0,
                    **generation_kwargs,
                )
            except Exception as err:
                logger.error(f"Generation failed for sample {idx}")
                if ignore_errors:
                    logger.error(err)
                    output = None
                else:
                    raise err

            outputs.append(output)
            progress.update(task, advance=1)

            is_cache_hit = model.is_last_result_from_cache()
            if (
                hasattr(model, "request_interval")
                and model.request_interval > 0
                and not is_cache_hit
            ):
                time.sleep(model.request_interval)

    return outputs
