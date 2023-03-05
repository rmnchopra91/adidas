import logging
import pytest

from classification_model.config.core import config
from classification_model.processing.data_manager import load_raw_dataset

logger = logging.getLogger(__name__)

@pytest.fixture
def sample_input_data():
  logger.info("sample_input_data calling.......................................")
  data = load_raw_dataset(file_name=config.app_config.test_data_file)
  return data
