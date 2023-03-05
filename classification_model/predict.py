import logging
import typing as t
import pandas as pd

from classification_model import __version__ as _version
from classification_model.config.core import config
from classification_model.processing.data_manager import load_model, pre_processing

logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.model_save_file}{_version}.pkl"
_titanic_pipe = load_model(file_name=pipeline_file_name)

def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
  """Make a prediction using a saved model pipeline."""
  logger.info("make_prediction calling ...")
  results = {"predictions": None, "version": _version}
  data = pre_processing(input_data)
  predictions = _titanic_pipe.predict(data)
  results = {
    "predictions": predictions,
    "version": _version,
  }
  return results
