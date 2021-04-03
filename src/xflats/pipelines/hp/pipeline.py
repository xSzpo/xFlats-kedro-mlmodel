from kedro.pipeline import Pipeline, node

from .nodes import split_data, transform_data_hp


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_flats", "aggregate_avg_flats_prices",
                        "parameters"],
                outputs="hp_model_input",
                name="create_data_split_hp",
            ),
            node(
                func=transform_data_hp,
                inputs="hp_model_input",
                outputs=["hp_train", "hp_test", "hp_valid"],
                name="prepare_transformed_data_hp"
            )
        ]
    )
