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
    is_valid_prediction,
    log_metrics,
    save_scores,
    generation_metrics,
)
from .utils.utils import config_logging, get_output_path

logger = logging.getLogger(__name__)


def _get_samples_for_key(data, key):
    samples = None

    if isinstance(key, str):
        try:
            samples = data[key]
        except:
            pass
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
            samples = None
    else:
        logger.warning(
            f"Key '{key}' must be a string or dict but is of type {type(key)}"
        )

    if samples is None:
        logger.warning(f"Key '{key}' not found in dataset")
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


def _preprocess_kwargs(kwargs, sample_idxs=None, max_samples=None):
    kwargs_ = dict(kwargs)

    for key, arg_val in kwargs.items():
        if (
            key.endswith("_path")
            and isinstance(arg_val, str)
            and Path(arg_val).exists()
        ):
            arg_key = key.replace("_path", "")
            arg_path = Path(arg_val)
            values = None
            if arg_path.suffix == ".txt":
                with open(arg_val) as fh:
                    values = fh.readlines()
            elif arg_path.suffix == ".json":
                values = pd.read_json(arg_val, header=None)
                values = values.iloc[:, 0].values.tolist()
            elif arg_path.suffix == ".csv":
                values = pd.read_csv(arg_val, header=None)
                values = values.iloc[:, 0].values.tolist()

            if values is not None:
                logger.info(f"Loaded {len(values)} '{arg_key}' values from {arg_val}.")

                if sample_idxs and len(sample_idxs) == len(values):
                    values = [values[idx] for idx in sample_idxs]
                    logger.info(
                        f"Shuffled {len(sample_idxs)} values for argment {arg_key}."
                    )
                elif sample_idxs:
                    logger.warning(
                        f"Sample indexes are inconsistent with list of arguments '{arg_key}'!"
                        f" Make sure the list matches the order of evaluation samples *after* shuffling."
                    )

                kwargs_[arg_key] = values[:max_samples]
                kwargs_.pop(key)

        elif key.endswith("_path"):
            logger.warning(
                f"Could not load argment values. File does not exist: {arg_val}"
            )

    return kwargs_


def _load_eval_data(
    source_key,
    target_key,
    dataset_name=None,
    dataset_config=None,
    split=None,
    arxiv_id=None,
    arxiv_query=None,
    max_samples=None,
    output_dir=None,
    timestr=None,
    run_id=None,
    data_cache_dir=None,
):
    if arxiv_id or arxiv_query:
        papers = load_arxiv_data(arxiv_id, arxiv_query, max_samples)
        eval_data = {
            "entry_id": [p["entry_id"] for p in papers],
            source_key: [p["text"] for p in papers],
            target_key: [p["summary"] for p in papers],
        }
        dataset_config = arxiv_id if arxiv_id else arxiv_query
        save_to = get_output_path(
            output_dir,
            "arxiv_api",
            dataset_config,
            timestr=timestr,
            run_id=run_id,
        )
        if save_to:
            arxiv_data_path = os.path.join(save_to, "arxiv-data.json")
            pd.DataFrame(eval_data).to_json(arxiv_data_path)
        logger.info(f"Loaded {len(eval_data)} samples from arXiv API: {dataset_config}")
    elif Path(dataset_name).suffix == ".pdf":
        text = pdf_to_text(dataset_name)
        eval_data = {source_key: [text]}
        logger.info(f"Loaded PDF from {dataset_name}")
    elif Path(dataset_name).suffix == ".txt":
        with open(dataset_name) as fh:
            text = fh.readlines()
        eval_data = {source_key: [text]}
        logger.info(f"Loaded text from {dataset_name}")
    elif Path(dataset_name).suffix == ".json":
        eval_data = pd.read_json(dataset_name)
        key = eval_data.columns[0]
        logger.info(f"Loaded {len(eval_data[key])} samples from {dataset_name}")
    elif Path(dataset_name).suffix == ".csv":
        eval_data = pd.read_csv(dataset_name)
        key = eval_data.columns[0]
        logger.info(f"Loaded {len(eval_data[key])} samples from {dataset_name}")
    else:
        eval_data = datasets.load_dataset(
            dataset_name, dataset_config, cache_dir=data_cache_dir
        )
        logger.info(
            f"Loaded {len(eval_data[split])} {split} samples from {dataset_name}/{dataset_config}"
        )
        eval_data = eval_data[split]
    return eval_data


def _load_predictions(path, key, sources, targets, sample_idxs=None, max_samples=None):
    prediction_data = pd.read_csv(path)
    predictions = prediction_data[key].values

    if sample_idxs and len(sample_idxs) == len(predictions):
        predictions = [predictions[idx] for idx in sample_idxs]
        logger.info(f"Shuffled {len(sample_idxs)} '{key}' values")
    elif sample_idxs:
        logger.warning(
            f"Sample indexes are inconsistent with list of predictiond '{key}'!"
            f" Make sure the list matches the order of evaluation samples *after* shuffling."
        )

    predictions = predictions[:max_samples]

    if sources is not None and len(predictions) != len(sources):
        raise ValueError(
            f"Number of predictions from {path} ({len(predictions)}) "
            f"is incompatible with number of source documents ({len(sources)})."
        )
    elif targets is not None and len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions from {path} ({len(predictions)}) "
            f"is incompatible with number of targets ({len(targets)})."
        )
    return predictions


def eval_job(
    prediction,
    target,
    source,
    doc_id,
    metrics,
    **kwargs,
):
    if not is_valid_prediction(prediction):
        logger.error(f"Invalid prediction: {prediction}")
        return

    scores = compute_metrics(
        metrics,
        [prediction],
        sources=[source],
        references=[target],
        parallelized=True,
        **kwargs,
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
    save_sources=False,
    seed=17,
    **kwargs,
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
        **kwargs,
    )

    parallel_scores = []
    if parallel_metrics:
        _sources = [None] * len(preds) if sources is None else sources[:n_samples]
        _targets = [None] * len(preds) if targets is None else targets[:n_samples]
        parallel_scores = p_map(
            lambda pred, target, source, doc_id: eval_job(
                pred, target, source, doc_id, parallel_metrics, **kwargs
            ),
            preds,
            _targets,
            _sources,
            doc_ids,
        )

    # We need to filter out invalid predictions *after* metric calculation because some
    # of the arguments used in the metrics are based on the original index. This should
    # be refactored in the future.
    valid_pred_idxs = [
        idx for idx, pred in enumerate(preds) if is_valid_prediction(pred)
    ]
    if len(valid_pred_idxs) < len(preds):
        logger.warning(
            f"Found {len(preds) - len(valid_pred_idxs)} predictions with no content"
        )
        preds = [preds[idx] for idx in valid_pred_idxs]
        if targets is not None:
            targets = [targets[idx] for idx in valid_pred_idxs]
        if sources is not None:
            sources = [sources[idx] for idx in valid_pred_idxs]

    scores, agg_scores = _aggregate_scores(parallel_scores, scores)
    log_metrics(agg_scores)

    if save_to:
        filepath = Path(save_to)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        preds_dict = {"prediction": preds}
        if targets is not None:
            preds_dict["reference"] = targets
        if save_sources and sources is not None:
            preds_dict["source"] = sources
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
    source_key="source",
    target_key="target",
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
    save_sources=False,
    **kwargs,
):
    if arxiv_id or arxiv_query:
        dataset_name = "arxiv-api"

    if timestr is None:
        timestr = config_logging(
            dataset_name, dataset_config, split, output_dir, run_id=run_id
        )

    if all(x is None for x in [dataset_name, arxiv_id, arxiv_query]):
        raise ValueError(
            "Plese specify one of 'dataset_name', 'arxiv_id', 'arxiv_query'"
        )

    if model_name is None and prediction_path is None:
        raise ValueError("Please specify one of 'model_name' or 'prediction_path'")

    eval_data = _load_eval_data(
        source_key,
        target_key,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        arxiv_id=arxiv_id,
        arxiv_query=arxiv_query,
        max_samples=max_samples,
        output_dir=output_dir,
        timestr=timestr,
        run_id=run_id,
        data_cache_dir=data_cache_dir,
    )
    sources = _get_samples_for_key(eval_data, source_key)
    targets = _get_samples_for_key(eval_data, target_key) if target_key else None

    if preprocess_fn:
        sources, targets = preprocess_fn(
            sources, targets, max_samples=max_samples, logger=logger
        )

    sample_idxs = None
    if shuffle:
        seed = kwargs.get("seed")
        logger.info(f"Shuffling data using seed: {seed}")
        np.random.seed(seed)
        sample_idxs = list(range(len(sources)))
        np.random.shuffle(sample_idxs)
        sample_idxs = sample_idxs
        sources = [sources[idx] for idx in sample_idxs]
        if targets is not None:
            targets = [targets[idx] for idx in sample_idxs]

    kwargs = _preprocess_kwargs(
        kwargs, sample_idxs=sample_idxs, max_samples=max_samples
    )

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
            predictions = _load_predictions(
                model_name,
                prediction_key,
                sources,
                targets,
                sample_idxs=sample_idxs,
                max_samples=max_samples,
            )
            logger.info(f"Evaluating {model_name} on {len(predictions)} samples...")
        else:
            logger.info(f"Evaluating {model_name} on {len(sources)} samples...")
            predictions = generate(
                model_name,
                sources,
                model_class=model_class,
                cache_start=cache_start,
                cache_end=cache_end,
                **kwargs,
            )

        def remove_prefix(x, p):
            return x.replace(p, "") if x.startswith(p) else x

        kwargs = {remove_prefix(k, "prompt_"): v for k, v in kwargs.items()}

        save_to = get_output_path(
            output_dir,
            dataset_name,
            dataset_config,
            split,
            model_name=model_name,
            timestr=timestr,
            run_id=run_id,
        )
        scores, agg_scores = evaluate(
            predictions,
            metrics,
            sources=sources,
            targets=targets,
            save_to=save_to,
            parallelize=parallelize,
            save_sources=save_sources,
            **kwargs,
        )
        result = dict(
            predictions=predictions,
            targets=targets,
            scores=scores,
            agg_scores=agg_scores,
            output_path=save_to,
        )
        return result


def run(**kwargs):
    evaluate_model(**kwargs)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(run)
