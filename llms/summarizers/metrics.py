from ..metrics import generation_metrics, rouge_score


def summarization_metrics(
    prediction, reference=None, source=None, rouge_ngrams=None, parallelized=False
):
    metrics = generation_metrics(
        prediction, reference=reference, source=source, parallelized=parallelized
    )

    if reference is not None:
        rouge = rouge_score(prediction, reference, rouge_ngrams=rouge_ngrams)
        metrics["rouge"] = [rouge]

    return metrics
