import logging
import os
from pprint import pformat
import re
import time

import fire
import numpy as np

from .models.huggingface import (
    Text2TextLM,
    LlamaChat,
    InstructText2TextLM,
    InstructCausalLM,
    Alpaca,
    Vicuna,
)
from .models.openai import OpenAIChat
from .utils.utils import get_progress_bar, add_progress_task, LOG_LEVEL_FINE

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gpt-3.5-turbo"
MODEL_CACHE = {}

DEFAULT_MODEL_MAP = {
    r"gpt-[-\d\w]*": OpenAIChat,
    r"facebook/opt-[\d\w]+": InstructCausalLM,
    r".*[cC]ode[Ll]lama-[\d\w]+-[Ii]nstruct-hf": LlamaChat,
    r".*llama-?2.*chat.*": LlamaChat,
    r".*[Ll]lama.*": InstructCausalLM,
    r".*gpt2.*": InstructCausalLM,
    r"bigscience/T0[_\d\w]*": InstructText2TextLM,
    r"google/flan-t5[-\d\w]+": InstructText2TextLM,
    r"google/long-t5[-\d\w]+": InstructText2TextLM,
    r".*alpaca.*": Alpaca,
    r".*vicuna.*": Vicuna,
    r"mosaicml/mpt[-\d\w]$": InstructCausalLM,
    r"tiiuae/falcon[-\d\w]$": InstructCausalLM,
    r"mosaicml/mpt[-\d\w]+instruct": Alpaca,
    r"tiiuae/falcon[-\d\w]+instruct": InstructCausalLM,
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
        if default_class:
            summarizer_class = default_class
        else:
            summarizer_class = Text2TextLM
        logger.warning(
            f"Could not match model '{model_name}' to generator class. Using {summarizer_class}."
        )
    return summarizer_class


def preprocess_kwargs(kwargs):
    kwargs = dict(kwargs)

    if "model" in kwargs:
        kwargs["model_name"] = kwargs.pop("model")
    elif "model_name" not in kwargs:
        kwargs["model_name"] = os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL_NAME)

    if "model_path" in kwargs:
        kwargs["model_checkpoint_path"] = kwargs.pop("model_path")

    if kwargs.pop("ignore_cache", False) is True:
        kwargs["cache_end"] = 0

    if "prompt" in kwargs:
        kwargs["user_prompt"] = kwargs.pop("prompt")

    if "verbose" not in kwargs and "output_dir" in kwargs:
        os.environ["LOG_LEVEL"] = str(LOG_LEVEL_FINE)
    elif kwargs.pop("verbose", False) is True:
        os.environ["LOG_LEVEL"] = str(LOG_LEVEL_FINE)

    if "input_path" in kwargs:
        kwargs["dataset_name"] = kwargs.pop("input_path", None)

    return kwargs


def get_sample_gen_kwargs(kwargs, sample_idx):
    sample_kwargs = {}
    for arg_name, arg_val in kwargs.items():
        if arg_val is not None and callable(arg_val):
            sample_kwargs[arg_name] = arg_val(sample_idx)
        elif arg_val is not None and isinstance(arg_val, list):
            sample_kwargs[arg_name] = arg_val[sample_idx]
        else:
            sample_kwargs[arg_name] = arg_val
    return sample_kwargs


def token_statistics(
    model,
    inputs,
    truncation=True,
    ignore_errors=False,
    show_progress=True,
    **generation_kwargs,
):
    progress = get_progress_bar()
    task = add_progress_task(
        progress,
        f"Calculating token statistics for {model.model_name}...",
        total=len(inputs),
        existing_ok=False,
    )
    progress.update(task, visible=show_progress)
    truncated_tokens = []
    num_tokens = []

    with progress:
        for idx, input_data in enumerate(inputs):
            sample_kwargs = get_sample_gen_kwargs(generation_kwargs, idx)
            try:
                result = model.token_statistics(input_data, truncation, **sample_kwargs)
                num_tokens.append(result[0])
                truncated_tokens.append(result[1])
            except Exception as err:
                logger.error(f"Failed to compute token statistics for sample {idx}")
                if ignore_errors:
                    logger.error(err)
                else:
                    raise err

            progress.update(task, advance=1)

    stats = dict(
        total_tokens=sum(num_tokens),
        mean_tokens=np.mean(num_tokens),
        total_truncation=sum(truncated_tokens),
        mean_truncation=np.mean(truncated_tokens),
    )
    return stats


def generate(
    model_name,
    sources=None,
    model_class=None,
    max_length=256,
    cache_start=0,
    cache_end=None,
    use_model_cache=False,
    ignore_errors=False,
    show_progress=True,
    verbose=None,
    **kwargs,
):
    outputs = []
    progress = get_progress_bar()
    if sources is None:
        sources = [None]
        show_progress = False

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

    logger.log(LOG_LEVEL_FINE, f"Using model: {model_class}")
    model = model_class(model_name, **model_kwargs)

    if use_model_cache:
        cached_model = MODEL_CACHE.get(model_name)
        if cached_model and hasattr(cached_model, "model"):
            model.model = cached_model.model
        MODEL_CACHE[model_name] = model

    if hasattr(model, "token_statistics"):
        stats = token_statistics(
            model,
            sources,
            max_length=max_length,
            ignore_errors=ignore_errors,
            show_progress=show_progress,
            **generation_kwargs,
        )
        logger.log(LOG_LEVEL_FINE, f"Token statistics for input:\n{pformat(stats)}")

    with progress:
        for idx, text in enumerate(sources):
            sample_kwargs = get_sample_gen_kwargs(generation_kwargs, idx)
            ignore_cache = idx < cache_start or idx >= cache_end

            if verbose is None:
                verbose = idx == 0

            try:
                output = model.generate(
                    text,
                    max_length=max_length,
                    memoizer_ignore_cache=ignore_cache,
                    verbose=verbose,
                    **sample_kwargs,
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
            if is_cache_hit and len(sources) == 1 and not ignore_cache:
                logger.warning(
                    "This model output is taken from cache. To force a new generation use --ignore_cache."
                )

            if (
                hasattr(model, "request_interval")
                and model.request_interval > 0
                and not is_cache_hit
            ):
                time.sleep(model.request_interval)

    return outputs


def run(**kwargs):
    kwargs = preprocess_kwargs(kwargs)
    outputs = generate(**kwargs)
    if len(outputs) == 1:
        print(f"> {kwargs['model_name']}:\n{outputs[0]}")


def main():
    fire.Fire(run)


if __name__ == "__main__":
    main()
