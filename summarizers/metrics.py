import json
import logging
from pathlib import Path

from nltk import ngrams
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer, scoring
from scipy.stats import bootstrap
import textstat

from .utils import add_progress_task, get_progress_bar, sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


def scores_to_df(scores):
    scores_df = {}

    for score_key, _scores in scores.items():
        if score_key == "rouge":
            for rouge_scores in _scores:
                for rouge_key, score in rouge_scores.items():
                    for sub_key in ["precision", "recall", "fmeasure"]:
                        values = scores_df.get(f"{rouge_key}_{sub_key}", [])
                        value = getattr(score, sub_key)
                        values.append(value)
                        scores_df[f"{rouge_key}_{sub_key}"] = values

        else:
            for sample_scores in _scores:
                for sub_key in sample_scores:
                    values = scores_df.get(f"{score_key}_{sub_key}", [])
                    if isinstance(sample_scores[sub_key], list):
                        value = sample_scores[sub_key][0]
                    else:
                        value = sample_scores[sub_key]
                    values.append(value)
                    scores_df[f"{score_key}_{sub_key}"] = values

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


def log_rouge_scores(scores):
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

    logger.info("\n".join(info))


def log_metric(name, scores):
    if len(scores) == 0:
        return

    if len(scores) == 1:
        logger.info(f"{name}: {scores[0]}")
        return

    if name == "rouge":
        log_rouge_scores(scores)
    else:
        info = [f"{name}:"]
        for key in scores:
            if isinstance(scores[key], dict):
                _score = []
                for confidence_key in ["low", "mean", "high"]:
                    if isinstance(
                        scores[key][confidence_key], np.ndarray
                    ) or isinstance(scores[key][confidence_key], list):
                        score = [f"{x:.3f}" for x in scores[key][confidence_key]]
                        score = f"\n  {confidence_key}: {str(score)}"
                    else:
                        score = f"{scores[key][confidence_key]:.3f}"
                    _score.append(score)

                _score = ", ".join(_score)
            else:
                _score = f"{scores[key]:.3f}"
            info.append(f"{key}: {_score}")
        info.append(" ")
        logger.info("\n".join(info))


def log_metrics(metrics):
    for key, metric in metrics.items():
        log_metric(key, metric)


def aggregate_scores(scores):
    if len(scores) == 1:
        return scores[0]
    elif len(scores) == 0:
        return {}

    agg_scores = {}

    for score in scores:
        for key in score.keys():
            key_scores = agg_scores.get(key, [])
            agg_scores[key] = key_scores

            _score = score[key]
            if isinstance(_score, list):
                if np.any(np.isinf(_score)):
                    continue
                if len(_score) == 1:
                    _score = _score[0]
            elif np.isinf(_score):
                continue

            key_scores.append(_score)

    confidence_intervals = {}
    for key in agg_scores.keys():
        if len(agg_scores[key]) > 1:
            ci = bootstrap(
                (agg_scores[key],),
                np.mean,
                vectorized=True,
                axis=0,
                confidence_level=0.95,
                random_state=17,
                method="BCa",
            )
            confidence_intervals[key] = {
                "low": ci.confidence_interval.low.tolist(),
                "high": ci.confidence_interval.high.tolist(),
                "mean": np.mean(agg_scores[key], axis=0).tolist(),
            }

    return confidence_intervals


def aggregate_metrics(metrics):
    scores = {}
    for metric_name, metric_scores in metrics.items():
        if metric_name == "rouge":
            aggregator = scoring.BootstrapAggregator()
            for rouge_score in metric_scores:
                aggregator.add_scores(rouge_score)
            agg_scores = aggregator.aggregate()
        else:
            agg_scores = aggregate_scores(metric_scores)
        scores[metric_name] = agg_scores
    return scores


def abstractiveness(source, summary):
    result = {}
    if isinstance(source, list):
        source = "\n".join(source)
    if isinstance(summary, list):
        summary = "\n".join(summary)

    for n in range(1, 5):
        source_ngrams = list(ngrams(source.split(" "), n))
        summary_ngrams = list(ngrams(summary.split(" "), n))
        novel_ngrams = [x for x in summary_ngrams if x not in source_ngrams]
        result[f"{n}_gram"] = len(novel_ngrams) / len(summary_ngrams)
    return result


def text_statistics(text, prefix=None):
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


def rouge_score(summary, target_summary, rouge_ngrams=None):
    if rouge_ngrams is None or len(rouge_ngrams) == 0:
        rouge_ngrams = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge = rouge_scorer.RougeScorer(rouge_ngrams, use_stemmer=True)

    if isinstance(target_summary, (list, tuple)):
        target_summary_str = "\n".join(target_summary)
    else:
        target_summary_str = target_summary

    if isinstance(summary, (list, tuple)):
        summary_str = "\n".join(summary)
    else:
        summary_str = summary

    return rouge.score(target_summary_str, summary_str)


def summarization_metrics(
    summary, target_summary=None, source=None, rouge_ngrams=None, verbose=False
):
    metrics = {}
    if source is not None:
        metrics["source_stats"] = text_statistics(source)
        metrics["summary_abstractiveness"] = abstractiveness(source, summary)
        metrics["target_abstractiveness"] = abstractiveness(source, target_summary)

    metrics["summary_stats"] = text_statistics(summary)

    if target_summary is not None:
        metrics["target_stats"] = text_statistics(target_summary)

        metrics["conciseness"] = {}
        for x in ["sentences", "tokens"]:
            summary_val = metrics["summary_stats"][f"{x}_per_sample"]
            target_val = metrics["target_stats"][f"{x}_per_sample"]
            metrics["conciseness"][f"{x}_diff"] = abs(target_val - summary_val)

        rouge = rouge_score(summary, target_summary, rouge_ngrams=rouge_ngrams)
        metrics["rouge"] = [rouge]

    if verbose:
        log_metrics(metrics)

    return metrics


def compute_metric(references, candidates, metric_fn, progress=None, **metric_kwargs):
    if progress is None:
        progress = get_progress_bar()

    results = []
    task = add_progress_task(
        progress,
        f"Computing {metric_fn.__name__}...",
        total=len(references),
        existing_ok=False,
    )
    with progress:
        for ref, cand in zip(references, candidates):
            results.append(
                metric_fn(references=[ref], candidates=[cand], **metric_kwargs)
            )
            progress.update(task, advance=1)
    return results
