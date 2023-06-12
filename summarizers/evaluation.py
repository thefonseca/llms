import logging
import os
from pathlib import Path

from arxiv import SortCriterion
import datasets
import fire
import numpy as np
import pandas as pd
from p_tqdm import p_map

from .arxiv import search_arxiv, pdf_to_text
from .inference import predict_summaries
from .metrics import (
    aggregate_metrics,
    compute_metric,
    log_metrics,
    save_scores,
    summarization_metrics,
)
from .utils import config_logging, get_output_path

logger = logging.getLogger(__name__)


def _aggregate_parallel_results(p_results):
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


def _aggregate_results(results, scores):
    if type(results) == list:
        results = _aggregate_parallel_results(results)

    if scores:
        for key in scores:
            if key in results:
                logger.warning(f"Values for metric {key} already exist.")
            results[key] = scores[key]
    agg_scores = aggregate_metrics(results)
    return results, agg_scores


def eval_job(
    pred,
    target,
    source,
    doc_id,
):
    try:
        if target is None or str(target) == "nan" or len(target) == 0:
            return
    except:
        logger.error(f"Invalid target summary: {target}")

    metrics = summarization_metrics(pred, target_summary=target, source=source)
    return metrics


def evaluate(
    preds,
    targets,
    sources,
    scores=None,
    save_to=None,
    n_samples=None,
    seed=17,
):
    # this seed is for confidence interval estimation via bootstrapping
    np.random.seed(seed)
    _preds = preds[:n_samples]
    _targets = targets[:n_samples]
    _sources = sources[:n_samples]
    doc_ids = list(range(len(_preds)))
    results = p_map(
        lambda pred, target, source, doc_id: eval_job(
            pred,
            target,
            source,
            doc_id,
        ),
        _preds,
        _targets,
        _sources,
        doc_ids,
    )

    results, agg_scores = _aggregate_results(results, scores)
    log_metrics(agg_scores)

    if save_to:
        filepath = Path(save_to)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        preds_df = pd.DataFrame({"prediction": _preds, "target": _targets})
        preds_filename = f"{filepath.stem}_predictions.csv"
        preds_filename = filepath.parent / preds_filename
        preds_df.to_csv(preds_filename, index=False)
        save_scores(results, agg_scores, save_to)

    return scores


def load_arxiv_data(arxiv_id, arxiv_query, max_samples, source_key, target_key):
    if isinstance(arxiv_id, list):
        arxiv_id = [str(x) for x in arxiv_id]
    else:
        arxiv_id = str(arxiv_id)

    logger.info(f"Arxiv IDs: {arxiv_id}")
    logger.info(f"Arxiv query: {arxiv_query}")
    if arxiv_id and Path(arxiv_id).suffix == ".txt":
        arxiv_ids_file = arxiv_id
        arxiv_id = np.loadtxt(arxiv_ids_file)
        logger.info(f"Loaded {len(arxiv_id)} arXiv IDs from {arxiv_ids_file}")

    if max_samples is None:
        max_samples = float("inf")
    else:
        max_samples = int(max_samples * 1.1)

    papers = search_arxiv(
        arxiv_id,
        arxiv_query,
        # add 10% more samples as some of them will not be valid
        max_results=max_samples,
        sort_by=SortCriterion.SubmittedDate,
        remove_abstract=True,
    )
    eval_data = {
        "entry_id": [p["entry_id"] for p in papers],
        source_key: [p["text"] for p in papers],
        target_key: [p["summary"] for p in papers],
    }
    return eval_data


def evaluate_model(
    dataset_name=None,
    dataset_config=None,
    split=None,
    source_key="article",
    target_key="abstract",
    source_file=None,
    arxiv_id=None,
    arxiv_query=None,
    model_name=None,
    summarizer_class=None,
    prediction_path=None,
    prediction_key="prediction",
    max_samples=None,
    shuffle=False,
    output_dir=None,
    cache_start=0,
    cache_end=None,
    data_cache_dir=None,
    metrics=None,
    run_id=None,
    timestr=None,
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

    sources = eval_data[source_key]
    targets = eval_data[target_key]

    if shuffle:
        seed = kwargs.get("seed")
        logger.info(f"Shuffling data using seed: {seed}")
        np.random.seed(seed)
        idxs = list(range(len(sources)))
        np.random.shuffle(idxs)
        sources = [sources[idx] for idx in idxs]
        targets = [targets[idx] for idx in idxs]

    sources = sources[:max_samples]
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
        logger.info(f"Evaluating {model_name} on {len(sources)} samples.")

        if Path(model_name).suffix == ".csv":
            logger.info(f"Loading predictions from {model_name}...")
            summary_data = pd.read_csv(model_name)
            preds = summary_data[prediction_key].values[:max_samples]
            if len(preds) != len(sources):
                raise ValueError(
                    f"Number of predictions from {model_name} ({len(summary_data)}) "
                    f"is incompatible with number of source documents ({len(sources)})."
                )
        else:
            preds = predict_summaries(
                model_name,
                sources,
                summarizer_class=summarizer_class,
                cache_start=cache_start,
                cache_end=cache_end,
                **kwargs,
            )

        def is_valid_pred(pred):
            return pred and str(pred) != "nan"

        valid_pred_idxs = [idx for idx, pred in enumerate(preds) if is_valid_pred(pred)]
        if len(valid_pred_idxs) < len(preds):
            logger.warning(
                f"Found {len(preds) - len(valid_pred_idxs)} predictions with no content"
            )
            preds = [preds[idx] for idx in valid_pred_idxs]
            targets = [targets[idx] for idx in valid_pred_idxs]

        scores = None
        if metrics:
            scores = {}
            for metric_name, metric in metrics.items():
                metric_kwargs = metric.get("metric_kwargs", {})
                metric_scores = compute_metric(
                    targets,
                    preds,
                    metric["metric_fn"],
                    **metric_kwargs,
                )
                scores[metric_name] = metric_scores

        if targets is not None and len(targets):
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
            evaluate(
                preds,
                targets,
                sources,
                scores=scores,
                save_to=save_to,
                **_kwargs,
            )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(evaluate_model)
