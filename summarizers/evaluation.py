import json
import logging
import os
from pathlib import Path

from arxiv import SortCriterion
import datasets
import fire
import numpy as np
import pandas as pd
from p_tqdm import p_map
from rouge_score import scoring

from .arxiv import search_arxiv, pdf_to_text
from .inference import predict_summaries
from .metrics import summarization_metrics
from .utils import (
    aggregate_scores,
    compute_metric,
    config_logging,
    get_output_path,
    log_scores,
    sent_tokenize,
    word_tokenize,
)

logger = logging.getLogger(__name__)


def _print_eval_metrics(results):
    info_str = [
        f"Number of documents: {len(results['sents_per_summary'])}",
        f"Average sentences per summary: {np.mean(results['sents_per_summary'])}",
        f"Average tokens per summary: {np.mean(results['tokens_per_summary'])}",
        f"Average tokens per summary sent: {np.mean(results['tokens_per_summary_sent'])}",
        f"Average sentences per target: {np.mean(results['sents_per_abstract'])}",
        f"Average tokens per target: {np.mean(results['tokens_per_abstract'])}",
        f"Average tokens per target sent: {np.mean(results['tokens_per_abstract_sent'])}",
        f"Average token difference: {np.mean(results['length_diffs'])}",
    ]
    logger.info("\n".join(info_str))
    scores = results["scores"]
    for score_key in scores:
        log_scores(score_key, scores[score_key])


def _aggregate_parallel_results(p_results):
    results = {}

    for result in p_results:
        for key in result.keys():
            if type(result[key]) == dict:
                dict_values = results[key]
                for dict_key in result[key]:
                    values = dict_values.get(dict_key, [])
                    values.extend(result[key][dict_key])
                    dict_values[dict_key] = values
            else:
                values = results.get(key, [])
                values.extend(result[key])
                results[key] = values
    return results


def _aggregate_results(results):
    if type(results) == list:
        results = _aggregate_parallel_results(results)

    scores = {}
    aggregator = scoring.BootstrapAggregator()
    for rouge_score in results["rouge_scores"]:
        aggregator.add_scores(rouge_score)
    scores["rouge"] = aggregator.aggregate()
    results["scores"] = scores
    return results


def _get_text_statistics(target, summary):
    sents_per_summary = []
    tokens_per_summary = []
    tokens_per_summary_sent = []
    sents_per_abstract = []
    tokens_per_abstract = []
    tokens_per_abstract_sent = []
    length_diffs = []

    if isinstance(target, list):
        target_sents = target
        target = "\n".join(target)
    else:
        target_sents = sent_tokenize(target)

    if isinstance(summary, list):
        summary_sents = summary
        summary = "\n".join(summary)
    else:
        summary_sents = sent_tokenize(summary)

    pred_words = word_tokenize(summary)
    target_words = word_tokenize(target)

    length_diff = len(pred_words) - len(target_words)
    length_diffs.append(length_diff)
    sents_per_summary.append(len(summary_sents))
    tokens_per_summary_sent.append(
        np.mean([len(word_tokenize(s)) for s in summary_sents])
    )
    tokens_per_summary.append(len(pred_words))
    sents_per_abstract.append(len(target_sents))
    tokens_per_abstract_sent.append(
        np.mean([len(word_tokenize(s)) for s in target_sents])
    )
    tokens_per_abstract.append(len(target_words))

    statistics = dict(
        sents_per_summary=sents_per_summary,
        tokens_per_summary=tokens_per_summary,
        tokens_per_summary_sent=tokens_per_summary_sent,
        sents_per_abstract=sents_per_abstract,
        tokens_per_abstract=tokens_per_abstract,
        tokens_per_abstract_sent=tokens_per_abstract_sent,
        length_diffs=length_diffs,
    )
    return statistics


def eval_job(
    pred,
    target,
    doc_id,
):
    rouge_scores = []

    try:
        if target is None or str(target) == "nan" or len(target) == 0:
            return
    except:
        logger.error(f"Invalid target summary: {target}")

    metrics = summarization_metrics(pred, target_summary=target)
    rouge_score = metrics["rouge"]

    if rouge_score:
        rouge_scores.append(rouge_score)

    stats = _get_text_statistics(target, pred)
    results = dict(rouge_scores=rouge_scores, **stats)
    return results


def evaluate(
    preds,
    targets,
    scores=None,
    save_preds_to=None,
    n_samples=None,
    seed=17,
):
    np.random.seed(seed)
    _preds = preds[:n_samples]
    _targets = targets[:n_samples]

    doc_ids = list(range(len(_preds)))
    results = p_map(
        lambda pred, target, doc_id: eval_job(
            pred,
            target,
            doc_id,
        ),
        _preds,
        _targets,
        doc_ids,
    )

    results = _aggregate_results(results)
    scores_df = {}

    if scores:
        for score_key in scores:
            agg_scores = aggregate_scores(scores[score_key])
            results["scores"][score_key] = agg_scores

            for sample_scores in scores[score_key]:
                for sub_key, value in sample_scores.items():
                    values = scores_df.get(f"{score_key}_{sub_key}", [])
                    if isinstance(sample_scores[sub_key], list):
                        value = sample_scores[sub_key][0]
                    else:
                        value = sample_scores[sub_key]
                    values.append(value)
                    scores_df[f"{score_key}_{sub_key}"] = values

    for scores in results["rouge_scores"]:
        for score_key, score in scores.items():
            for sub_key in ["precision", "recall", "fmeasure"]:
                values = scores_df.get(f"{score_key}_{sub_key}", [])
                value = getattr(score, sub_key)
                values.append(value)
                scores_df[f"{score_key}_{sub_key}"] = values

    _print_eval_metrics(results)

    if save_preds_to:
        filepath = Path(save_preds_to)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        preds_df = pd.DataFrame({"prediction": _preds, "target": _targets})
        preds_filename = f"{filepath.stem}_predictions.csv"
        preds_filename = filepath.parent / preds_filename
        preds_df.to_csv(preds_filename, index=False)

        scores_df = pd.DataFrame(scores_df)
        scores_filename = f"{filepath.stem}_metrics_per_sample.csv"
        scores_filename = filepath.parent / scores_filename
        scores_df.to_csv(scores_filename, index=False)

        results_filename = f"{filepath.stem}_metrics.json"
        results_filename = filepath.parent / results_filename
        with open(results_filename, "w") as file:
            json.dump(results["scores"], file, indent=2)

    return results["scores"]


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

    sources = eval_data[source_key][:max_samples]
    targets = eval_data[target_key][:max_samples]

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
        logger.info(f"Evaluating {model_name}")

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
            evaluate(preds, targets, scores=scores, save_preds_to=save_to, **_kwargs)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(evaluate_model)
