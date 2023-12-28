from itertools import zip_longest
import json
import logging
from pathlib import Path

from nltk import ngrams
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer, scoring
from rouge_score.tokenizers import DefaultTokenizer
from scipy.stats import bootstrap
import textstat

from .utils.utils import (
    add_progress_task,
    get_progress_bar,
    sent_tokenize,
    word_tokenize,
)

logger = logging.getLogger(__name__)


def scores_to_df(scores, key=None, scores_df=None):
    if scores_df is None:
        scores_df = {}

    if isinstance(scores, dict):
        if key and "rouge" in key:
            for rouge_key, score in scores.items():
                for sub_key in ["precision", "recall", "fmeasure"]:
                    _key = f"{key}_{rouge_key}_{sub_key}"
                    values = scores_df.get(_key, [])
                    value = getattr(score, sub_key)
                    values.append(value)
                    scores_df[_key] = values

        else:
            for sub_key, score in scores.items():
                if key:
                    sub_key = f"{key}_{sub_key}"
                scores_to_df(score, key=sub_key, scores_df=scores_df)

    elif isinstance(scores, list) and isinstance(scores[0], dict):
        for score in scores:
            scores_to_df(score, key=key, scores_df=scores_df)

    else:
        values = scores_df.get(key, [])
        if isinstance(scores, list):
            value = scores[0]
        else:
            value = scores
        values.append(value)
        scores_df[key] = values

    return scores_df


def save_scores(scores_per_sample, agg_scores, save_to):
    filepath = Path(save_to)
    scores_df = scores_to_df(scores_per_sample)
    scores_df = pd.DataFrame(scores_df)
    scores_filename = f"{filepath.stem}_metrics_per_sample.csv"
    scores_filename = filepath.parent / scores_filename
    scores_df.to_csv(scores_filename, index=False)

    results_filename = f"{filepath.stem}_metrics.json"
    results_filename = filepath.parent / results_filename
    with open(results_filename, "w") as file:
        json.dump(agg_scores, file, indent=2)


def get_rouge_info(scores):
    info = ["ROUGE scores:"]

    for k, v in sorted(scores.items()):
        if hasattr(v, "low"):
            score_info = [
                f"{k}-R: {v.low.recall:.5f}, {v.mid.recall:.5f}, {v.high.recall:.5f}",
                f"{k}-P: {v.low.precision:.5f}, {v.mid.precision:.5f}, {v.high.precision:.5f}",
                f"{k}-F: {v.low.fmeasure:.5f}, {v.mid.fmeasure:.5f}, {v.high.fmeasure:.5f}",
            ]
        else:
            score_info = [
                f"{k}-R: {v.recall:.5f}, {v.recall:.5f}, {v.recall:.5f}",
                f"{k}-P: {v.precision:.5f}, {v.precision:.5f}, {v.precision:.5f}",
                f"{k}-F: {v.fmeasure:.5f}, {v.fmeasure:.5f}, {v.fmeasure:.5f}",
            ]
        info.append("\n".join(score_info))
        info.append(" ")

    return info


def log_rouge_scores(scores):
    info = get_rouge_info(scores)
    logger.info("\n".join(info))


def get_metric_info(scores, key=None, info=None):
    if info is None:
        info = []
    _score = None
    if isinstance(scores, dict):
        if key == "rouge":
            info.append("\n")
            info.extend(get_rouge_info(scores))

        elif all([k in scores.keys() for k in ["low", "mean", "high"]]):
            _score = []
            for confidence_key in ["low", "mean", "high"]:
                if isinstance(scores[confidence_key], (np.ndarray, list)):
                    score = [f"{x:.3f}" for x in scores[confidence_key]]
                    score = f"\n  {confidence_key}: {str(score)}"
                else:
                    score = f"{scores[confidence_key]:.3f}"
                _score.append(score)
            _score = ", ".join(_score)

        else:
            if key:
                if info:
                    info.append("\n")
                    info.append(f"> {key}:")
                else:
                    info.append(f"{key}:")
            for key_, score in scores.items():
                get_metric_info(score, key=key_, info=info)

    elif isinstance(scores, (np.ndarray, list)):
        _score = [f"{x:.3f}" for x in scores]
    else:
        _score = f"{scores:.3f}"

    if _score:
        info.append(f"{key}: {_score}")
    return info


def log_metrics(scores):
    info = get_metric_info(scores)
    info = "\n".join(info)
    info = info.replace("\n\n", "\n")
    logger.info(info)


def get_confidence_interval(scores):
    _scores = [x for x in scores if x is not None]

    if len(_scores) < len(scores):
        null_count = len(scores) - len(_scores)
        logger.warning(
            f"Ignoring {null_count} null score values in confidence interval computation"
        )
    scores = _scores
    confidence_interval = {}

    def is_constant(a): # avoid DegenerateDataWarning
        a = np.array(a)
        return (np.isclose(a, a[0]) | np.isnan(a)).all()

    if is_constant(scores):
        confidence_interval["mean"] = scores[0]

    elif len(scores) > 1:
        ci = bootstrap(
            (scores,),
            np.mean,
            vectorized=True,
            axis=0,
            confidence_level=0.95,
            random_state=17,
            method="BCa",
        )
        confidence_interval = {
            "low": ci.confidence_interval.low.tolist(),
            "high": ci.confidence_interval.high.tolist(),
            "mean": np.mean(scores, axis=0).tolist(),
        }
    elif len(scores) == 1:
        confidence_interval["mean"] = scores[0]

    return confidence_interval


def aggregate_metrics(metrics):
    def _add_scores(results_dict, key, scores):
        key_results = results_dict.get(key, [])
        key_results.append(scores)
        results_dict[key] = key_results

    def _scores_dict_to_list(scores, results_dict=None, key="root"):
        """Recursively aggregate a list of dicts into a dict of list of scores"""

        if results_dict is None:
            results_dict = {}

        if isinstance(scores, dict) and key != "rouge":
            for key_, score in scores.items():
                key_results = results_dict.get(key, {})
                results_dict[key] = key_results
                _scores_dict_to_list(score, key_results, key=key_)
        elif isinstance(scores, list) and isinstance(scores[0], list):
            # This case cover metrics that are vectors
            for score in scores:
                _add_scores(results_dict, key, score)
        elif isinstance(scores, list):
            for score in scores:
                _scores_dict_to_list(score, results_dict, key=key)
        else:
            _add_scores(results_dict, key, scores)
        return results_dict.get("root")

    def _aggregate_scores(scores, results_dict=None, key="root"):
        """Recursively calculate confidence intervals for lists of metric scores"""

        if results_dict is None:
            results_dict = {}

        if isinstance(scores, dict):
            for key_, score in scores.items():
                key_results = results_dict.get(key, {})
                results_dict[key] = key_results
                _aggregate_scores(score, key_results, key=key_)
        elif isinstance(scores, list):
            if key == "rouge":
                aggregator = scoring.BootstrapAggregator()
                for rouge_score in scores:
                    aggregator.add_scores(rouge_score)
                agg_scores = aggregator.aggregate()
            else:
                agg_scores = get_confidence_interval(scores)
            results_dict[key] = agg_scores
        return results_dict.get("root")

    scores = _scores_dict_to_list(metrics)
    agg_scores = _aggregate_scores(scores)
    return agg_scores


def abstractiveness(source, prediction):
    result = {}
    if isinstance(source, list):
        source = "\n".join(source)
    if isinstance(prediction, list):
        prediction = "\n".join(prediction)

    tokenizer = DefaultTokenizer(use_stemmer=True)
    source_tokens = tokenizer.tokenize(source)
    prediction_tokens = tokenizer.tokenize(prediction)

    for n in range(1, 5):
        source_ngrams = list(ngrams(source_tokens, n))
        prediction_ngrams = list(ngrams(prediction_tokens, n))
        novel_ngrams = [x for x in prediction_ngrams if x not in source_ngrams]
        if len(prediction_ngrams):
            result[f"{n}_gram"] = len(novel_ngrams) / len(prediction_ngrams)
        else:
            result[f"{n}_gram"] = 0
    return result


def text_statistics(text, prefix=None):
    if isinstance(text, dict):
        text = "\n".join(text.values())

    if isinstance(text, list):
        sentences = text
        text = "\n".join(text)
    else:
        sentences = sent_tokenize(text)

    sentences_per_sample = len(sentences)
    tokens_per_sentence = np.mean([len(word_tokenize(s)) for s in sentences])
    tokens_per_sample = len(word_tokenize(text))
    flesch_kincaid = textstat.flesch_kincaid_grade(text)
    statistics = dict(
        sentences_per_sample=sentences_per_sample,
        tokens_per_sentence=tokens_per_sentence,
        tokens_per_sample=tokens_per_sample,
        fkgl_readability=flesch_kincaid,
    )
    if prefix:
        statistics = {f"{prefix}_{k}": v for k, v in statistics.items()}
    return statistics


def rouge_score(prediction, reference, rouge_ngrams=None):
    if rouge_ngrams is None or len(rouge_ngrams) == 0:
        rouge_ngrams = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge = rouge_scorer.RougeScorer(rouge_ngrams, use_stemmer=True)

    if isinstance(reference, (list, tuple)):
        reference_str = "\n".join(reference)
    else:
        reference_str = reference

    if isinstance(prediction, (list, tuple)):
        prediction_str = "\n".join(prediction)
    else:
        prediction_str = prediction

    return rouge.score(reference_str, prediction_str)


def generation_metrics(
    prediction, reference=None, source=None, budget=None, index=None, parallelized=False
):
    metrics = {}
    if source is not None:
        metrics["source_stats"] = text_statistics(source)

    metrics["prediction_stats"] = text_statistics(str(prediction))

    if budget and "sentences_per_sample" in metrics["prediction_stats"]:
        n_sents = metrics["prediction_stats"]["sentences_per_sample"]
        metrics["prediction_budget_guidance_diff"] = abs(n_sents - budget)

    if reference is not None:
        if str(reference) == "nan":
            reference = "None"

        if isinstance(reference, str):
            metrics["reference_stats"] = text_statistics(reference)
            if budget and "sentences_per_sample" in metrics["reference_stats"]:
                n_sents = metrics["reference_stats"]["sentences_per_sample"]
                metrics["reference_budget_guidance_diff"] = abs(n_sents - budget)
            
            metrics["length_diff_prediction_vs_reference"] = {}
            for x in ["sentences", "tokens"]:
                prediction_val = metrics["prediction_stats"][f"{x}_per_sample"]
                reference_val = metrics["reference_stats"][f"{x}_per_sample"]
                metrics["length_diff_prediction_vs_reference"][f"{x}_diff"] = abs(
                    reference_val - prediction_val
                )

    return metrics


def compute_metric(
    metric_fn,
    predictions,
    references=None,
    sources=None,
    progress=None,
    verbose=False,
    sample_level=True,
    parallelized=None,
    metric_name=None,
    **metric_kwargs,
):
    def _compute_metric(references, predictions, sources, progress=None):
        if references is None:
            references = [None]
        if sources is None:
            sources = [None]

        if sample_level:
            results = []
            for idx, (ref, pred, source) in enumerate(
                zip_longest(references, predictions, sources)
            ):
                result = metric_fn(
                    pred,
                    reference=ref,
                    source=source,
                    index=idx,
                    parallelized=parallelized,
                    **metric_kwargs,
                )
                results.append(result)
                if progress:
                    progress.update(task, advance=1)
        else:
            results = metric_fn(
                predictions=predictions, references=references, **metric_kwargs
            )
        return results

    if verbose:
        if progress is None:
            progress = get_progress_bar()

        if metric_name is None:
            metric_name = metric_fn.__name__

        task = add_progress_task(
            progress,
            f"Computing {metric_name}...",
            total=len(predictions),
            existing_ok=False,
        )
        with progress:
            results = _compute_metric(
                references, predictions, sources, progress=progress
            )
    else:
        results = _compute_metric(references, predictions, sources)

    return results


def compute_metrics(
    metrics,
    predictions,
    sources=None,
    references=None,
    parallelized=None,
    verbose=False,
    **kwargs,
):
    scores = {}
    if metrics and not isinstance(metrics, list):
        metrics = [metrics]

    for metric in metrics:
        if callable(metric):
            metric = {"metric_fn": metric, "metric_name": metric.__name__}

        metric_name = metric.get("metric_name")
        metric_kwargs = dict(metric.get("metric_kwargs", {}))
        metric_kwargs.update(kwargs)
        if sources is not None and metric.get("include_sources", True):
            metric_kwargs["sources"] = sources

        metric_scores = compute_metric(
            metric["metric_fn"],
            predictions,
            references=references,
            verbose=verbose,
            parallelized=parallelized,
            metric_name=metric_name,
            **metric_kwargs,
        )
        scores[metric_name] = metric_scores

    return scores
