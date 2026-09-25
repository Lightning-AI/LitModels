from litlogger import models as litlogger_models
from litlogger.models import cloud as litlogger_cloud
from litlogger.models import serialization as litlogger_serialization

import litmodels
from litmodels import io as litmodels_io
from litmodels.io import cloud as litmodels_cloud
from litmodels.io import gateway as litmodels_gateway
from litmodels.io import utils as litmodels_utils


def test_top_level_exports_are_reexported_from_litlogger():
    assert litmodels.download_model is litlogger_models.download_model
    assert litmodels.load_model is litlogger_models.load_model
    assert litmodels.save_model is litlogger_models.save_model
    assert litmodels.upload_model is litlogger_models.upload_model
    assert litmodels.upload_model_files is litlogger_models.upload_model_files


def test_io_exports_are_reexported_from_litlogger():
    assert litmodels_io.download_model is litlogger_models.download_model
    assert litmodels_io.download_model_files is litlogger_models.download_model_files
    assert litmodels_io.load_model is litlogger_models.load_model
    assert litmodels_io.save_model is litlogger_models.save_model
    assert litmodels_io.upload_model is litlogger_models.upload_model
    assert litmodels_io.upload_model_files is litlogger_models.upload_model_files
    assert litmodels_gateway.download_model is litlogger_models.download_model
    assert litmodels_gateway.load_model is litlogger_models.load_model
    assert litmodels_gateway.save_model is litlogger_models.save_model
    assert litmodels_gateway.upload_model is litlogger_models.upload_model


def test_cloud_exports_are_reexported_from_litlogger():
    assert litmodels_cloud.delete_model_version is litlogger_cloud.delete_model_version
    assert litmodels_cloud.download_model_files is litlogger_cloud.download_model_files
    assert litmodels_cloud.upload_model_files is litlogger_cloud.upload_model_files


def test_serialization_exports_are_reexported_from_litlogger():
    assert litmodels_utils._JOBLIB_AVAILABLE is litlogger_serialization._JOBLIB_AVAILABLE
    assert litmodels_utils._KERAS_AVAILABLE is litlogger_serialization._KERAS_AVAILABLE
    assert litmodels_utils._PYTORCH_AVAILABLE is litlogger_serialization._PYTORCH_AVAILABLE
    assert litmodels_utils._TENSORFLOW_AVAILABLE is litlogger_serialization._TENSORFLOW_AVAILABLE
    assert litmodels_utils.dump_pickle is litlogger_serialization.dump_pickle
    assert litmodels_utils.load_pickle is litlogger_serialization.load_pickle
