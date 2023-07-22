import logging
from pprint import pformat
import time

from .utils.utils import get_progress_bar, add_progress_task

logger = logging.getLogger(__name__)


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


def generate(
    model_name,
    sources,
    model_class,
    max_length=256,
    cache_start=0,
    cache_end=None,
    ignore_errors=False,
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
    cache_end = cache_end if cache_end is not None else len(sources)
    model_kwargs, generation_kwargs = parse_kwargs(kwargs)

    logger.info(f"Using model: {model_class}")
    model = model_class(model_name, **model_kwargs)

    if hasattr(model, "token_statistics"):
        stats = model.token_statistics(
            sources, max_length=max_length, **generation_kwargs
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
