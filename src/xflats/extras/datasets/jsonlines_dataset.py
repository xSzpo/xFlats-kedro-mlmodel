from pathlib import PurePosixPath
from typing import Any, Dict

from kedro.io import AbstractVersionedDataSet, Version
from kedro.io.core import get_protocol_and_path

import fsspec

from google.cloud import storage
import os
import types
import logging
import warnings
import datetime

import jsonlines
from jsonschema import validate, Draft3Validator, SchemaError, ValidationError
import jsonschema
import json
import pandas as pd
import numpy as np
import re
import codecs


logger = logging.getLogger(__name__)


class JSONLineDataSet(AbstractVersionedDataSet):
    """``ManyJSONLineDataSet`` loads / save multiple JSONLine files.

    Example:
    ::

        >>> ImageDataSet(filepath='/img/file/path.png')
    """
    def __init__(self,
                 filepath: str,
                 version: Version = None,
                 file_mask: str = '.+.[jsonline|jsonl]',
                 encoding: str = 'utf-8',
                 drop_columns: list = None,
                 schema_path: str = None
                 ):
        """Creates a new instance of ManyJSONLineDataSet to load / save
        JSONLine fies, with specified mask, from given directory,

        Args:
            filepath : string
                Path to local directory where you store jsonlines.
            file_mask : string, default='.+.[jsonline|jsonl]'
                Use only files that matches REGEX pattern
            encoding : str, default='utf-8'
                Apply encoding when read jsonlines.
            drop_columns : str, list, default=None
                Columns to drop when reading json line.
            schema_path : str, default=None
                Path to json shema file, if provided - all jsons are validated on read,
                pandas dtypes structure is generated and applied to DataFrame.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._file_mask = file_mask
        self._encoding = encoding
        self._drop_columns = drop_columns
        self._schema_path = schema_path
        self._fs = fsspec.filesystem(self._protocol)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _load(self) -> pd.DataFrame:
        """Loads all JSONLine files into pandas DF

        Returns:
            Data from the JSONLines as pd.DataFrame
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = self._get_load_path()

        jr = ReadJsonline(
                dir_json_local=load_path,
                file_mask=self._file_mask,
                drop_columns=self._drop_columns,
                encoding=self._encoding,
                schema_path=self._schema_path
                )
        return jr.alljson2df()

    def _save(self, data: pd.DataFrame) -> None:
        """Saves data as parquet"""
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = self._get_save_path()
        data.to_parquet(save_path, compression='gzip')

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath, version=self._version, protocol=self._protocol
        )


class ReadJsonline:
    """Reads jsonlines and convert to Pandas DF.

       Reads all jsonline files from directory that matches provided regex
       expression mask.
       Validates each json in line with json schema.
       Generates dtypes dictionary from json schema and use it in Pandas DF.
       Convert columns to datetime64 dtype base on json schema structure.

    Parameters
    ----------
    dir_json_local : string
        Path to local directory where you store jsonlines.
    file_mask : string, default='.+.[jsonline|jsonl]'
        Use only files that matches REGEX pattern
    encoding : str, default='utf-8'
        Apply encoding when read jsonlines.
    drop_columns : str, list, default=None
        Columns to drop when reading json line.
    schema_path : str, default=None
        Path to json shema file, if provided - all jsons are validated on read,
        pandas dtypes structure is generated and applied to DataFrame.

    Attributes
    ----------
    schema : dict (JSON Schema)
        JSON Schema from provided path.
    dtypes : dict
        Dtype dictionary genrated fron schema and passed to pandas
        `pd.DataFrame(data, columns=self.dtypes)`.
    date : list
        A list of json columns that contains data/datatime values created from
        JSON Schema and applied to pandas `pd.to_datetime(tmp_df[date])`

    Examples
    --------
    >>> import Xszpo
    >>> import logging
    >>> import sys

    >>> logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
    >>>        format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'")

    >>> jr = Xszpo.DataSources.ReadJsonline(
    >>>     dir_json_local = "/Users/xszpo/GoogleDrive/01_Projects/202003_xFlats_K8S/mlflow/data",
    >>>     file_mask = r'flats_\w+_20200\d\d\d.jsonline',
    >>>     drop_columns='body',
    >>>     encoding='utf-8',
    >>>     schema_path="/Users/xszpo/GoogleDrive/01_Projects/202003_xFlats_K8S/data_structure/schema.json")

    >>> x = jr.alljson2df()

    Notes
    -----
    No notes
    """

    def __init__(self, dir_json_local, file_mask='.+.[jsonline|jsonl]',
                 encoding='utf-8', drop_columns=None,
                 schema_path=None):
        self.dir_json_local = dir_json_local
        self.drop_columns = drop_columns
        self.schema_path = schema_path
        self.encoding = encoding
        self.file_mask = file_mask
        self.file_mask_recompile = re.compile(file_mask, re.IGNORECASE)
        if self.schema_path:
            self.schema = self._load_schema()
            self.dtypes, self.dates = self._get_pd_dtypes_from_schema()
        else:
            self.schema = None
            self.dtypes, self.dates = None, None

    def _load_schema(self):

        with open(self.schema_path, "r") as file:
            schema = json.load(file)
        return schema

    def _get_pd_dtypes_from_schema(self):

        dtypes = dict()
        dates = list()

        for key in self.schema['properties'].keys():
            tmp = self.schema['properties'][key]

            if isinstance(tmp['type'], list):
                if 'string' in tmp['type']:
                    if 'format' in tmp:
                        if tmp['format'] in ('date-time', 'date'):
                            dtypes[key] = object
                            dates += [key]
                    else:
                        dtypes[key] = str
                elif 'number' in tmp['type']:
                    dtypes[key] = np.float32
                elif 'integer' in tmp['type']:
                    dtypes[key] = np.float32
                else:
                    dtypes[key] = str

            else:
                if tmp['type'] == 'string':
                    if 'format' in tmp:
                        if tmp['format'] in ('date-time', 'date'):
                            dtypes[key] = object
                            dates += [key]
                    else:
                        dtypes[key] = str
                elif tmp['type'] == 'number':
                    dtypes[key] = np.float32
                elif tmp['type'] == 'integer':
                    dtypes[key] = np.float32
                else:
                    dtypes[key] = str

        return dtypes, dates

    def _valid_json_line(self, json_line_path):

        valid = Draft3Validator(self.schema)
        with jsonlines.open(json_line_path) as reader:
            for obj in reader:
                if not valid.is_valid(obj):
                    return False
        return True

    def _valid_json(self, json):

        valid = Draft3Validator(self.schema)
        return valid.is_valid(json)

    def _read_one_jsonline(self, filename):

        data = list()
        file_path = os.path.join(self.dir_json_local, filename)

        with codecs.open(file_path, mode='r', encoding=self.encoding) as file:

            for i, line in enumerate(file):

                try:
                    tmp = json.loads(line)

                    if self.drop_columns:
                        if isinstance(self.drop_columns, list):
                            for col in self.drop_columns:
                                if col in tmp.keys():
                                    _ = tmp.pop(col)
                        else:
                            _ = tmp.pop(self.drop_columns)

                    if self.schema_path:
                        if self._valid_json(tmp):
                            data += [tmp]
                        else:
                            logger.debug(("file {filename}, line {i} hasn't " +
                                         "pass schema validation").format(
                                          filename=filename, i=i))
                    else:
                        data += [tmp]

                except BaseException as e:
                    logger.debug(("file {filename}, line {i} error: " +
                                 "{e}").format(filename=filename, i=i, e=e))

        tmp_dtypes = self.dtypes.copy()
        tmp_col = pd.DataFrame.from_dict(data[:1]).columns

        for c in self.dtypes:
            if c not in tmp_col:
                tmp_dtypes.pop(c)

        tmp_df = pd.DataFrame.from_dict(data).astype(tmp_dtypes)

        if self.dates:
            for date in self.dates:
                if date in tmp_col:
                    tmp_df[date] = pd.to_datetime(tmp_df[date]). \
                        astype("datetime64[ms]")

        return tmp_df

    def __read_one_jsonline_deprecated(self, filename):
        try:
            df = pd.read_json(os.path.join(self.dir_json_local, filename),
                              orient='table',
                              lines=True,
                              dtype=self.dtypes,
                              encoding=self.encoding,
                              convert_dates=self.dates).drop(
                                self.drop_columns, axis=1)
        except BaseException as e:
            print(filename, logger.error(e))
            df = None
        return df

    def json2df(self, json_name):
        file_path = os.path.join(self.dir_json_local, json_name)
        if self.schema_path:
            if self._valid_json_line(file_path):
                return self._read_one_jsonline(json_name)
            else:
                logger.debug(("file %s couldn't be loaded - it" +
                               " hasn't pass schema validation.") % json_name)
                return None
        else:
            logger.info(("No json schema provided - file %s will not be " +
                         "validated. Pandas will attempt to guess columns " +
                         "data type.") % json_name)
            return self._read_one_jsonline(json_name)

    def alljson2df(self):

        if not self.schema_path:
            logger.info(("No json schema provided - files will not be " +
                         "validated. Pandas will attempt to guess columns " +
                         "data type."))

        pd_list = list()

        for jsonline_file in os.listdir(self.dir_json_local):
            if self.file_mask_recompile.match(jsonline_file):
                pd_list += [self._read_one_jsonline(jsonline_file)]

        return pd.concat(pd_list, ignore_index=True)


class GoogleGCS:
    """Google Cloud Data Storage source.
       Download data from your Google Storage bucket.
       It's able to synch directory - download only files that are not
       available in local directory.
       Provides basic filtering methods.

    Parameters
    ----------
    dir_json_local : string
        Path to local directory where you want to download data from GCP
        Storage.
    dir_json_bucket : string
        GCP Sorage bucket name ex. `flats_jsonlines`
    file_mask : string, default='.+.[jsonline|jsonl]'
        Download only files that matches REGEX pattern
    exclude_file_name : string or None (default)
        Download only files that name does not contain provided text value.
    secrets : string, default=None
        Provide path to secrets (json key), as a default if should be exported
        to ENV "GOOGLE_APPLICATION_CREDENTIALS", if not just provide a path
        to json file with key.

    Attributes
    ----------
    storage_client_ : class `google.cloud.storage.client.Client`
        Google cloud api storage class.
    blobs_ : list of objebcts class:`google.cloud.storage.Blob`
        Google cloud api blob class.

    Examples
    --------
    >>> from datetime import date
    >>> import logging
    >>> import sys
    >>> import Xszpo

    >>> logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
    >>>        format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'")

    >>> gcs = Xszpo.DataSources.GoogleGCS(
    >>>     dir_json_local = "/Users/xszpo/GoogleDrive/01_Projects/202003_xFlats_K8S/mlflow/data",
    >>>     dir_json_bucket = "flats_jsonlines",
    >>>     file_mask = 'flats_\w+_202004\d\d.jsonline',
    >>>     exclude_file_name = date.today().strftime("%Y%m%d")
    >>> )

    >>> gcs.download_gcs(testonefile=True, overrideall=False)
    '2020-04-29 23:05:18,761 - Xszpo.DataSources._source - INFO - Blob flats_gratka_20200420.jsonline downloaded to /Users/xszpo/GoogleDrive/01_Projects/202003_xFlats_K8S/mlflow/data.'

    Notes
    -----
    No Notes
    """

    def __init__(self, dir_json_local, dir_json_bucket,
                 file_mask='.+.[jsonline|jsonl]',
                 exclude_file_name=None,
                 secrets=None
                 ):
        self.dir_json_local = dir_json_local
        self.dir_json_bucket = dir_json_bucket
        self.file_mask = file_mask
        self.exclude_file_name = exclude_file_name
        self.secrets = secrets

        self.file_mask_recompile = re.compile(file_mask, re.IGNORECASE)
        if self.secrets:
            self.storage_client_ = storage.Client. \
                from_service_account_json(self.secrets)
        else:
            self.storage_client_ = storage.Client()

        self._get_all_blobs()

    def _download_blob(self, blob, destination_dir):
        destination_file_name = os.path.join(destination_dir, blob.name)
        blob.download_to_filename(destination_file_name)
        result = "Blob {} downloaded to {}.".format(blob.name, destination_dir)
        return result

    def _get_all_blobs(self):
        if hasattr(self, "blobs_"):
            pass
        else:
            self.blobs_ = list(self.storage_client_.list_blobs(
                                self.dir_json_bucket))

    def _filer_blob(self, name):
        """ apply filters: extension, include text and exclude text """
        if self.file_mask_recompile.match(name):
            if self.exclude_file_name:
                if self.exclude_file_name in name:
                    return False
                else:
                    return True
            else:
                return True
        else:
            return False

    def download_gcs(self, overrideall=False, testonefile=False):
        """Download data from your Google Storage bucket.

        Parameters
        ----------
        overrideall : bool, default=False
            If set to False - do not download files that already in local
            direcory.
        testonefile : bool, default=False
            If set to True - download only one file.
        """

        list_files_local = list()
        list_files_gcp = list()

        if not overrideall:
            for file in os.listdir(self.dir_json_local):
                if file.endswith("jsonline"):
                    list_files_local += [file]

        for i, blob in enumerate(self.blobs_):
            if self._filer_blob(blob.name) and \
                    any([blob.name not in list_files_local]):
                logger.info(self._download_blob(blob, self.dir_json_local))
                if testonefile:
                    break
