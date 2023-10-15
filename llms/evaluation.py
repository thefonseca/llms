import logging
import os
from pathlib import Path

import datasets
import fire
import numpy as np
import pandas as pd
from p_tqdm import p_map

from .utils.arxiv import load_arxiv_data, pdf_to_text
from .inference import generate
from .metrics import (
    aggregate_metrics,
    compute_metrics,
    log_metrics,
    save_scores,
    generation_metrics,
)
from .utils.utils import config_logging, get_output_path

logger = logging.getLogger(__name__)


def _get_samples_for_key(data, key):
    samples = None

    if isinstance(key, str):
        samples = data[key]
    elif isinstance(key, dict):
        samples = []
        try:
            for template_key, data_key in key.items():
                data_values = data[data_key]
                for idx, val in enumerate(data_values):
                    if len(samples) <= idx:
                        sample = {template_key: val}
                        samples.append(sample)
                    else:
                        samples[idx][template_key] = val
        except:
            logger.warning(f"'Source/target {key}' not found in dataset")
            samples = None
    else:
        logger.warning(
            f"Source/target key must be a string or dict but is of type {type(key)}"
        )

    return samples


def _aggregate_parallel_scores(p_results):
    results = {}
    for result in p_results:
        for key in result.keys():
            values = results.get(key, [])
            if isinstance(result[key], list):
                values.extend(result[key])
            else:
                values.append(result[key])
            results[key] = values
    return results


def _aggregate_scores(p_scores, scores):
    if isinstance(p_scores, list):
        all_scores = _aggregate_parallel_scores(p_scores)

    if scores:
        for key in scores:
            if key in all_scores:
                logger.warning(f"Values for metric {key} already exist.")
            all_scores[key] = scores[key]

    agg_scores = aggregate_metrics(all_scores)
    return all_scores, agg_scores


def eval_job(
    prediction,
    target,
    source,
    doc_id,
    metrics,
):
    try:
        if target is not None and str(target) == "nan" and len(target) == 0:
            return
    except:
        logger.error(f"Invalid target: {target}")

    scores = compute_metrics(
        metrics,
        [prediction],
        sources=[source],
        references=[target],
        parallelized=True,
    )
    return scores


def evaluate(
    preds,
    metrics,
    sources=None,
    targets=None,
    scores=None,
    save_to=None,
    n_samples=None,
    parallelize=False,
    seed=17,
):
    # this seed is for confidence interval estimation via bootstrapping
    np.random.seed(seed)
    preds = preds[:n_samples]
    doc_ids = list(range(len(preds)))

    if metrics is None:
        metrics = [generation_metrics]
    if not isinstance(metrics, list):
        metrics = [metrics]

    def is_parallelizable_metric(m):
        # non-dicts (e.g., callables) are parallelizable by default
        return not isinstance(m, dict) or m.get("parallelizable", True)

    if parallelize:
        parallel_metrics = [m for m in metrics if is_parallelizable_metric(m)]
        non_parallel_metrics = [m for m in metrics if not is_parallelizable_metric(m)]
    else:
        non_parallel_metrics = metrics
        parallel_metrics = None

    # compute non-parallelizable metrics
    scores = compute_metrics(
        non_parallel_metrics,
        preds,
        sources=sources,
        references=targets,
        parallelized=False,
        verbose=True,
    )

    parallel_scores = []
    if parallel_metrics:
        _sources = [None] * len(preds) if sources is None else sources[:n_samples]
        _targets = [None] * len(preds) if targets is None else targets[:n_samples]
        parallel_scores = p_map(
            lambda pred, target, source, doc_id: eval_job(
                pred, target, source, doc_id, parallel_metrics
            ),
            preds,
            _targets,
            _sources,
            doc_ids,
        )

    scores, agg_scores = _aggregate_scores(parallel_scores, scores)
    log_metrics(agg_scores)

    if save_to:
        filepath = Path(save_to)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        preds_dict = {"prediction": preds}
        if targets is not None:
            preds_dict["reference"] = targets
        preds_df = pd.DataFrame(preds_dict)
        preds_filename = f"{filepath.stem}_predictions.csv"
        preds_filename = filepath.parent / preds_filename
        preds_df.to_csv(preds_filename, index=False)
        save_scores(scores, agg_scores, save_to)

    return scores, agg_scores


def evaluate_model(
    dataset_name=None,
    dataset_config=None,
    split=None,
    source_key=None,
    target_key=None,
    source_file=None,
    arxiv_id=None,
    arxiv_query=None,
    model_name=None,
    model_class=None,
    prediction_path=None,
    prediction_key="prediction",
    max_samples=None,
    shuffle=False,
    preprocess_fn=None,
    output_dir=None,
    cache_start=0,
    cache_end=None,
    data_cache_dir=None,
    metrics=None,
    run_id=None,
    timestr=None,
    parallelize=False,
    **kwargs,
):
    if arxiv_id or arxiv_query:
        dataset_name = "arxiv-api"

    if timestr is None:
        timestr = config_logging(
            dataset_name, dataset_config, split, output_dir, run_id=run_id
        )

    if all(x is None for x in [dataset_name, arxiv_id, arxiv_query, source_file]):
        raise ValueError(
            "Plese specify one of 'dataset_name', 'arxiv_id', 'arxiv_query', or 'source_file'"
        )

    if model_name is None and prediction_path is None:
        raise ValueError("Please specify one of 'model_name' or 'prediction_path'")

    if arxiv_id or arxiv_query:
        eval_data = load_arxiv_data(
            arxiv_id, arxiv_query, max_samples, source_key, target_key
        )
        save_to = get_output_path(
            output_dir,
            dataset_name,
            dataset_config,
            timestr=timestr,
            run_id=run_id,
        )
        if save_to:
            arxiv_data_path = os.path.join(save_to, "arxiv-data.json")
            pd.DataFrame(eval_data).to_json(arxiv_data_path)
    elif source_file and Path(source_file).suffix == ".pdf":
        text = pdf_to_text(source_file)
        eval_data = {source_key: [text], target_key: []}
    elif source_file and Path(source_file).suffix == ".txt":
        with open(source_file) as fh:
            text = fh.readlines()
        eval_data = {source_key: [text], target_key: []}
    elif Path(dataset_name).suffix == ".json":
        eval_data = pd.read_json(dataset_name)
        logger.info(f"Loaded {len(eval_data)} samples from {dataset_name}")
    elif Path(dataset_name).suffix == ".csv":
        eval_data = pd.read_csv(dataset_name)
        logger.info(f"Loaded {len(eval_data)} samples from {dataset_name}")
    else:
        eval_data = datasets.load_dataset(
            dataset_name, dataset_config, cache_dir=data_cache_dir
        )
        eval_data = eval_data[split]

    sources = _get_samples_for_key(eval_data, source_key)
    targets = _get_samples_for_key(eval_data, target_key) if target_key else None

    if preprocess_fn:
        sources, targets = preprocess_fn(
            sources, targets, max_samples=max_samples, logger=logger
        )

    if shuffle:
        seed = kwargs.get("seed")
        logger.info(f"Shuffling data using seed: {seed}")
        np.random.seed(seed)
        idxs = list(range(len(sources)))
        np.random.shuffle(idxs)
        sources = [sources[idx] for idx in idxs]
        if targets is not None:
            targets = [targets[idx] for idx in idxs]

    if sources is not None:
        sources = sources[:max_samples]
    if targets is not None:
        targets = targets[:max_samples]

    model_names = []
    if model_name and isinstance(model_name, (list, tuple)):
        model_names = model_name
    elif model_name:
        model_names.append(model_name)

    if prediction_path and isinstance(prediction_path, (list, tuple)):
        model_names.extend(prediction_path)
    elif prediction_path:
        model_names.append(prediction_path)

    for model_name in model_names:
        if Path(model_name).suffix == ".csv":
            logger.info(f"Loading predictions from {model_name}...")
            prediction_data = pd.read_csv(model_name)
            predictions = prediction_data[prediction_key].values[:max_samples]
            if sources is not None and len(predictions) != len(sources):
                raise ValueError(
                    f"Number of predictions from {model_name} ({len(predictions)}) "
                    f"is incompatible with number of source documents ({len(sources)})."
                )
            elif targets is not None and len(predictions) != len(targets):
                raise ValueError(
                    f"Number of predictions from {model_name} ({len(predictions)}) "
                    f"is incompatible with number of targets ({len(targets)})."
                )
            logger.info(f"Evaluating {model_name} on {len(predictions)} samples.")
        else:
            logger.info(f"Evaluating {model_name} on {len(sources)} samples.")
            predictions = generate(
                model_name,
                sources,
                model_class=model_class,
                cache_start=cache_start,
                cache_end=cache_end,
                **kwargs,
            )

        def is_valid_pred(pred):
            return pred is not None and str(pred) != "nan"

        valid_pred_idxs = [
            idx for idx, pred in enumerate(predictions) if is_valid_pred(pred)
        ]
        if len(valid_pred_idxs) < len(predictions):
            logger.warning(
                f"Found {len(predictions) - len(valid_pred_idxs)} predictions with no content"
            )
            predictions = [predictions[idx] for idx in valid_pred_idxs]
            if targets is not None:
                targets = [targets[idx] for idx in valid_pred_idxs]

        save_to = get_output_path(
            output_dir,
            dataset_name,
            dataset_config,
            split,
            model_name=model_name,
            timestr=timestr,
            run_id=run_id,
        )
        _kwargs = {}
        if "seed" in kwargs:
            _kwargs["seed"] = kwargs.get("seed")
        scores, agg_scores = evaluate(
            predictions,
            metrics,
            sources=sources,
            targets=targets,
            save_to=save_to,
            parallelize=parallelize,
            **_kwargs,
        )
        result = dict(
            predictions=predictions,
            targets=targets,
            scores=scores,
            agg_scores=agg_scores,
            output_path=save_to,
        )
        return result


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(evaluate_model)
