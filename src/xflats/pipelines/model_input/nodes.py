from typing import Dict

import re
from unidecode import unidecode
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Week
import random


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

    ds_tr = parameters["data_set_train_size"]
    ds_te = parameters["data_set_test_size"]
    ds_val = parameters["data_set_valid_size"]

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
