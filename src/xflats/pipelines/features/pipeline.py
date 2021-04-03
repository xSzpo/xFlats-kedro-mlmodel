from kedro.pipeline import Pipeline, node

from .nodes import aggregates_prices_in_neighbourhood


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=aggregates_prices_in_neighbourhood,
                inputs="preprocessed_flats",
                outputs="aggregate_avg_flats_prices",
                name="create_aggregates_prices_neighbourhood",
            )
        ]
    )
