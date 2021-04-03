from kedro.pipeline import Pipeline, node

from .nodes import normalize


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=normalize,
                inputs="flatsjsonlines",
                outputs="preprocessed_flats",
                name="preprocessed_flats_node",
            )
        ]
    )
