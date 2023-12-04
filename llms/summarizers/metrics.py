from ..metrics import generation_metrics, rouge_score, abstractiveness


def summarization_metrics(
    prediction,
    reference=None,
    source=None,
    rouge_ngrams=None,
    parallelized=False,
    index=None,
):
    metrics = generation_metrics(
        prediction, reference=reference, source=source, parallelized=parallelized
    )

    if source is not None:
        if prediction is not None:
            metrics["prediction_abstractiveness"] = abstractiveness(source, prediction)
        if reference is not None:
            metrics["reference_abstractiveness"] = abstractiveness(source, reference)

    if reference is not None:
        rouge = rouge_score(prediction, reference, rouge_ngrams=rouge_ngrams)
        metrics["rouge"] = [rouge]

    return metrics
