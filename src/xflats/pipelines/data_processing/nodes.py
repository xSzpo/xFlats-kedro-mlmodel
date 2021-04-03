from typing import Any, Dict

import re
from unidecode import unidecode
import numpy as np
import pandas as pd


def _market(x):
    if x.lower().startswith('w'):
        return 'wtorny'
    else:
        return'pierwotny'


def _building_type(x):
    _tmp = unidecode(x).lower()
    if any([i in _tmp for i in ('blok','mieszkanie')]):
        return 'blok'
    elif any([i in _tmp for i in ('dom','szeregowiec', 'segment', 'blizniak', 'willa', 'rezydencja')]):
        return 'dom'
    elif any([i in _tmp for i in ('apartamentowiec', 'wiezowiec', 'loft')]):
        return 'apartamentowiec'
    elif any([i in _tmp for i in ('kamienica','historyczny')]):
        return 'kamienica'
    else:
        return 'n/a'


def _building_material(x):
    _tmp = unidecode(x).lower()
    if 'cegla' in _tmp:
        return 'cegla'
    elif 'plyta' in _tmp:
        return 'plyta'
    elif 'zelbet' in _tmp:
        return 'zelbet'
    elif 'pustak' in _tmp:
        return 'pustak'
    elif 'rama' in _tmp:
        return 'rama'
    elif 'silikat' in _tmp:
        return 'silikat'
    elif 'beton' in _tmp:
        return 'beton'
    else:
        return 'n/a'


def _property_form(x):
    _tmp = re.sub(r'\s+', ' ', unidecode(x).lower())
    if ('spol' in _tmp) and ('wl' in _tmp) and ('bez' in _tmp) and ('kw' in _tmp):
        return 'spol_wlasnosc'
    elif ('spol' in _tmp) and ('wl' in _tmp) and ('kw' in _tmp):
        return 'spol_wlasnosc_kw'
    elif ('spol' in _tmp) and ('wl' in _tmp):
        return 'spol_wlasnosc'
    elif ('spol' in _tmp) and ('kw' in _tmp):
        return 'spol_kw'
    elif ('wlasn' in _tmp):
        return 'wlasnosc'
    elif ('umowa' in _tmp) and ('develop' in _tmp):
        return 'umowa_z_developerem'
    elif ('hipo' in _tmp):
        return 'hipoteczne'
    elif ('udzial' in _tmp):
        return 'udzial'
    else:
        return 'n/a'


def _offeror(x):
    _tmp = unidecode(x).lower()
    if 'pryw' in _tmp:
        return 'prywatna'
    if 'dew' in _tmp:
        return 'deweloper'
    if 'dev' in _tmp:
        return 'deweloper'
    else:
        return 'agencja'


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Node for normalizing and cleaning data.
       normalize:
       * offeror,
       * property_form,
       * building_material,
       * market,
       create:
       * date_offer,
       filter:
       * date_offer >= '2020-03',
       * price between (10e4 and 15e5),
       * flat_size < 250
    """

    df['date_offer'] = df['date_modified'].combine_first(df['date_created']). \
        combine_first(df['download_date'])
    df.offeror = df.offeror.fillna('n/a').apply(_offeror)
    df.property_form = df.property_form.fillna('n/a').apply(_property_form)
    df.building_material = df.building_material.fillna('n/a'). \
        apply(_building_material)
    df.building_type = df.building_type.fillna('n/a').apply(_building_type)
    df.market = df.market.fillna('n/a').apply(_market)

    query = "date_offer >= '2020-03' and price > 10e4 and price < 15e5 \
        and flat_size < 250  and GC_addr_city=='Warszawa'"

    return df.query(query)