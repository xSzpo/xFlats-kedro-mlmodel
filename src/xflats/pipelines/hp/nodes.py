from typing import Dict, Tuple

import re
from unidecode import unidecode
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Week
import random
import scipy

from unidecode import unidecode

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import category_encoders as ce

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def split_data(df: pd.DataFrame,
               df_aggr: pd.DataFrame,
               parameters: Dict) -> pd.DataFrame:
    """Splits data into features and targets training, test, valid sets.
    Args:
        df: Data containing features and target.
        df_aggr: Data containing edditional features, join on=['GC_addr_suburb','market','date_offer_w']
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    df['date_offer_d'] = df.date_offer.dt.floor("D")
    df['date_offer_w'] = df['date_offer_d'] + Week(weekday=6)
    df = df.merge(df_aggr, how='left',
                  on=['GC_addr_suburb', 'market', 'date_offer_w'])

    random.seed(666)

    ds_tr = parameters["data_set_hp_train_size"]
    ds_te = parameters["data_set_hp_test_size"]
    ds_val = parameters["data_set_hp_valid_size"]

    idx_valid = list(df.sort_values('date_offer').index[:ds_val])
    idx_tt = (df.sort_values('date_offer').index[ds_val:])
    idx_tt = random.sample(set(idx_tt), ds_tr+ds_te)
    idx_train = idx_tt[:ds_tr]
    idx_test = idx_tt[ds_tr:]

    df_train = df.loc[idx_train].reset_index(drop=True)
    df_test = df.loc[idx_test].reset_index(drop=True)
    df_valid = df.loc[idx_valid].reset_index(drop=True)

    drop_cols = ['tracking_id', 'price_m2', 'download_date',
                 'download_date_utc', 'date_created', 'date_modified',
                 'GC_boundingbox', 'GC_addr_road', 'GC_addr_city',
                 'GC_addr_state', 'GC_addr_country', 'GC_addr_country_code',
                 'prediction', 'date_offer_d', 'date_offer_w']

    X_train = df_train.drop(columns=drop_cols)
    X_train['split'] = 'train'
    X_test = df_test.drop(columns=drop_cols)
    X_test['split'] = 'test'
    X_valid = df_valid.drop(columns=drop_cols)
    X_valid['split'] = 'valid'

    return pd.concat([X_train, X_test, X_valid], axis=0)


def transform_data_hp(df: pd.DataFrame) \
        -> Tuple[scipy.sparse.csr.csr_matrix, scipy.sparse.csr.csr_matrix, scipy.sparse.csr.csr_matrix]:

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
                    'year_of_building', 'GC_latitude', 'GC_longitude',
                    'price_median_08w']

    stop_words = ['ale', 'oraz', 'lub', 'sie', 'and', 'the', 'jest', 'do',
                  'od', 'with', 'mozna']

    token_pattern = r'[A-Za-z]\w{2,}'

    def preProcess(s):
        return unidecode(s).lower()

    pipe = make_pipeline(
        ColumnTransformer([
            ('ce_oh', ce.OneHotEncoder(return_df=True, use_cat_names=True), cols_ce_oh),
            ('ce_GC', ce.TargetEncoder(return_df=True), cols_ce_te),
            ('numeric', 'passthrough', cols_numeric),
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

    return X_train_transformed, X_test_transformed, X_valid_transformed
