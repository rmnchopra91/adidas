"""
Note: These tests will fail if you have not first trained the model.
"""

import logging
import numpy as np

from sklearn.metrics import accuracy_score
from classification_model.predict import make_prediction

logger = logging.getLogger(__name__)

def test_make_prediction(sample_input_data):
  logger.info("test_make_prediction calling.........................................")
  # Given
  # When
  result = make_prediction(input_data = sample_input_data)
  # Then
  predictions = result.get("predictions")
  assert isinstance(predictions, np.ndarray)
  assert len(predictions) == sample_input_data.shape[0]
  assert result.get("errors") is None
