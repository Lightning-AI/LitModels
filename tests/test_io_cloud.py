from litlogger import models as litlogger_models

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


def test_cloud_and_utils_dependencies_are_reexported():
    assert litmodels_cloud.download_model_files is litlogger_models.download_model_files
    assert litmodels_cloud.upload_model_files is litlogger_models.upload_model_files
    assert callable(litmodels_cloud.delete_model_version)
    assert callable(litmodels_cloud._list_available_teamspaces)
    assert callable(litmodels_utils.dump_pickle)
    assert callable(litmodels_utils.load_pickle)
