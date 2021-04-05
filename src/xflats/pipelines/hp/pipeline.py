from kedro.pipeline import Pipeline, node

from .nodes import transform_data_hp
from .nodes import hp_tuning


def create_pipeline(**kwargs):
    return Pipeline(
        [
            #node(
            #    func=transform_data_hp,
            #    inputs="model_input",
            #    outputs=["hp_train", "hp_test", "hp_valid", "hp_y_train", "hp_y_test", "hp_y_valid"],
            #    name="prepare_transformed_data_hp"
            #),
            node(
                func=hp_tuning,
                inputs=["hp_train", "hp_test", "hp_y_train", "hp_y_test",
                        "params:hp_params"],
                outputs="model_lgb_params",
                name="hp_tuning"
            )
        ]
    )
