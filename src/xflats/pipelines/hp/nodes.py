from typing import Dict, Tuple

import re
from unidecode import unidecode
import random
from unidecode import unidecode
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Week

import scipy
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import category_encoders as ce
import logging

from functools import partial
from kedro.config import ConfigLoader
from hyperopt import fmin, hp, tpe
import lightgbm as lgb
import mlflow
from mlflow import log_metric, log_params
from mlflow.lightgbm import log_model
from mlflow.tracking import MlflowClient

from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.metrics import r2_score, median_absolute_error, \
    mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


logger = logging.getLogger(__name__)


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def transform_data_hp(df: pd.DataFrame) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray]:

    X_train = df.loc[df.split == 'train'].drop(columns=['split', 'price']). \
        reset_index(drop=True)
    X_test = df.loc[df.split == 'test'].drop(columns=['split', 'price']). \
        reset_index(drop=True)
    X_valid = df.loc[df.split == 'valid'].drop(columns=['split', 'price']). \
        reset_index(drop=True)

    y_train = df.loc[df.split == 'train'].price.reset_index(drop=True)
    y_test = df.loc[df.split == 'test'].price.reset_index(drop=True)
    y_valid = df.loc[df.split == 'valid'].price.reset_index(drop=True)

    cols_ce_oh = ['producer_name', 'market', 'building_type',
                  'building_material', 'property_form', 'offeror']

    cols_ce_te = ['GC_addr_suburb', 'GC_addr_postcode']

    cols_numeric = ['flat_size', 'rooms', 'floor', 'number_of_floors',
                    'year_of_building', 'GC_latitude', 'GC_longitude']

    cols_prices_in_neighbourhood = ['price_median_03w', 'price_median_08w',
                                    'price_median_12w', 'price_mean_03w',
                                    'price_mean_08w', 'price_mean_12w']

    stop_words = ['ale', 'oraz', 'lub', 'sie', 'and', 'the', 'jest', 'do',
                  'od', 'with', 'mozna']

    token_pattern = r'[A-Za-z]\w{2,}'

    def preProcess(s):
        return unidecode(s).lower()

    pipe = make_pipeline(
        ColumnTransformer([
            ('ce_oh', ce.OneHotEncoder(return_df=True, use_cat_names=True), cols_ce_oh),
            ('ce_GC', ce.TargetEncoder(return_df=True), cols_ce_te),
            ('numeric', 'passthrough', cols_numeric + cols_prices_in_neighbourhood),
            ('txt_description', TfidfVectorizer(lowercase=True,
                                                ngram_range=(1, 3),
                                                stop_words=stop_words,
                                                max_features=1000,
                                                token_pattern=token_pattern,
                                                preprocessor=preProcess,
                                                dtype=np.float32,
                                                use_idf=True
                                                ), 'description'),
            ('txt_name', TfidfVectorizer(lowercase=True,
                                         ngram_range=(1, 1),
                                         stop_words=stop_words,
                                         max_features=500,
                                         token_pattern=token_pattern,
                                         dtype=np.float32,
                                         binary=True,
                                         preprocessor=preProcess,
                                         use_idf=False
                                         ), 'name'),
        ]),
    )

    X_train_transformed = pipe.fit_transform(X_train, y_train)
    X_test_transformed = pipe.transform(X_test)
    X_valid_transformed = pipe.transform(X_valid)

    return X_train_transformed, X_test_transformed, X_valid_transformed, \
        y_train, y_test, y_valid


def lgb_hp(X_train: np.ndarray,
           X_test: np.ndarray,
           y_train: np.ndarray,
           y_test: np.ndarray) -> Dict:

    return 1


def _get_experiment() -> str:
    conf_paths = ["./conf/local", "./conf/base"]
    conf_loader = ConfigLoader(conf_paths=conf_paths)
    conf_mlflow = conf_loader.get("mlflow.yml")
    experiment_name = conf_mlflow\
        .get("experiment").get("name")
    client = MlflowClient(tracking_uri=conf_mlflow.get("tracking_uri"))
    experiments = client.list_experiments()
    lista = list(filter(lambda x: x.name == experiment_name, experiments))
    return lista[0].experiment_id if len(lista) > 0 else 0


def _get_tracking_uri() -> str:
    conf_paths = ["./conf/local", "./conf/base"]
    conf_loader = ConfigLoader(conf_paths=conf_paths)
    conf_mlflow = conf_loader.get("mlflow.yml")
    experiment_name = conf_mlflow\
        .get("experiment").get("name")
    return conf_mlflow.get("mlflow_tracking_uri")


def _objective(
        params: Dict,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        parameters) -> float:

    #experiment_id = _get_experiment()

    mlflow.lightgbm.autolog(log_input_examples=False,
                            log_model_signatures=False,
                            log_models=True,
                            disable=False,
                            exclusive=False,
                            disable_for_unsupported_versions=False,
                            silent=False)

    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        params['deterministic'] = True
        params['objective'] = "regression_l2"
        params['boosting'] = "gbdt"
        params['metric'] = ['l1', 'mape']
        params['seed'] = '666'

        train_params = {
            'num_boost_round': parameters['num_boost_round'],
            'verbose_eval': parameters['verbose_eval'],
            'early_stopping_rounds': parameters['early_stopping_rounds'],
        }

        train_data = lgb.Dataset(X_train, label=y_train, params={'verbose': -1})
        test_data = lgb.Dataset(X_test, label=y_test, params={'verbose': -1})

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'valid'],
            **train_params,
        )

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = median_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        return mae


def hp_tuning(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        parameters: Dict) -> Dict:

    space = {
        "learning_rate": hp.uniform("learning_rate", parameters["learning_rate"][0], parameters["learning_rate"][1]),
        "max_bin": hp.randint("max_bin", parameters["max_bin"][0], parameters["max_bin"][1]),
        "max_depth": hp.randint("max_depth", parameters["max_depth"][0], parameters["max_depth"][1]),
        "min_data_in_leaf": hp.randint("min_data_in_leaf", parameters["min_data_in_leaf"][0], parameters["min_data_in_leaf"][1]),
        "num_leaves": hp.randint("num_leaves", parameters["num_leaves"][0], parameters["num_leaves"][1]),
        "lambda_l1": hp.uniform("lambda_l1", parameters["lambda_l1"][0], parameters["lambda_l1"][1]),
        "lambda_l2": hp.uniform("lambda_l2", parameters["lambda_l2"][0], parameters["lambda_l2"][1]),
        "bagging_fraction": hp.uniform("bagging_fraction", parameters["bagging_fraction"][0], parameters["bagging_fraction"][1]),
        "bagging_freq": hp.randint("bagging_freq", parameters["bagging_freq"][0], parameters["bagging_freq"][1]),
        "feature_fraction": hp.uniform("feature_fraction", parameters["feature_fraction"][0], parameters["feature_fraction"][1]),
        }

    best = fmin(
        partial(_objective, X_train=X_train, X_test=X_test, y_train=y_train,
                y_test=y_test, parameters=parameters),
        space,
        algo=tpe.suggest,
        max_evals=parameters["hp_number_of_experiments"])

    for k in best:
        if type(best[k]) != str:
            best[k] = str(best[k])

    return best
