from kedro.pipeline import Pipeline, node

from .nodes import split_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_flats", "aggregate_avg_flats_prices",
                        "parameters"],
                outputs="model_input",
                name="create_data_split",
            )
        ]
    )
