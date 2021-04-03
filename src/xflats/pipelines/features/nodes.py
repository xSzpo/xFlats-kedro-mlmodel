from typing import Any, Dict

import re
from unidecode import unidecode
import numpy as np
import pandas as pd


def aggregates_prices_in_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
    """Node for calcuation of averages prices in districts.
    """

    _districts = list(pd.DataFrame(df.GC_addr_suburb.value_counts()).
                      query("GC_addr_suburb > 1000").index)
    _districts = [i for i in _districts if i not in ['Warszawa']]

    df.GC_addr_suburb = [i if i in _districts else np.nan for i in
                         df.GC_addr_suburb]
    df['date_offer_d'] = df.date_offer.dt.floor("D")

    df.set_index('date_offer_d', inplace=True)

    # period_length
    tmp01 = df[['GC_addr_suburb', 'market', 'price']]. \
        groupby(['GC_addr_suburb', 'market']).resample('W')

    _kind = ['median','mean']
    _weeks = [1, 2, 3, 4, 8, 12]
    results = list()

    _start = None

    for _k in _kind:
        for _w in _weeks:
            tmp01_agr = tmp01.aggregate(_k)

            # calculation not including week 0
            tmp01_agr_last_x_weeks = tmp01_agr.rolling(_w, min_periods=0). \
                aggregate(_k).shift(1).round(0)

            tmp01_agr_last_x_weeks = tmp01_agr_last_x_weeks.reset_index()

            _col_name = f'price_{_k}_{str(_w).zfill(2)}w'

            tmp01_agr_last_x_weeks.columns = ['GC_addr_suburb', 'market',
                                              'date_offer_w', _col_name]
            if _start:
                df_aggr[_col_name] = tmp01_agr_last_x_weeks[_col_name]
            else:
                _start = 1
                df_aggr = tmp01_agr_last_x_weeks.copy()

    return df_aggr
