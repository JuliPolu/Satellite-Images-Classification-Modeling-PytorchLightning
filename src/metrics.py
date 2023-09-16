from torchmetrics import F1Score, MetricCollection, Precision, Recall, HammingDistance


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'precision': Precision(**kwargs),
            'recall': Recall(**kwargs),
            'HD': HammingDistance(**kwargs),
        },
    )

