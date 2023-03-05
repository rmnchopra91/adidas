import logging
import numpy as np
import pandas as pd
import re

from pathlib import Path
from typing import Any, List, Union
from sklearn.pipeline import Pipeline
import joblib

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

logger = logging.getLogger(__name__)

def missingvalue(df):
  cols=df.select_dtypes('object').columns
  cols=cols.tolist()
  df['HomePlanet'].fillna(df['HomePlanet'].value_counts().index[0],inplace=True)
  df['CryoSleep'].fillna(df['CryoSleep'].value_counts().index[0],inplace=True)
  df['Destination'].fillna(df['Destination'].value_counts().index[0],inplace=True)
  df['VIP'].fillna(df['VIP'].value_counts().index[0],inplace=True)
  cols1=df.select_dtypes('float64').columns
  cols1=cols1.tolist()
  for i in cols1:
      df[i]=df[i].fillna(df[i].mean())
  return df

def Onehotencoding(df1):
  df = df1.copy()
  df1=df1.join(pd.get_dummies(df['HomePlanet'],prefix='HomePlanet',prefix_sep='_'))
  df1=df1.join(pd.get_dummies(df['CryoSleep'],prefix='CryoSleep',prefix_sep='_'))
  df1=df1.join(pd.get_dummies(df['Destination'],prefix='Destination',prefix_sep='_'))
  df1=df1.join(pd.get_dummies(df['VIP'],prefix='VIP',prefix_sep='_'))
  df1.drop(['HomePlanet','CryoSleep','Destination','VIP'],axis=1,inplace=True)
  return df1

def pre_processing(df):
  df.drop(['PassengerId','Name','Cabin'],axis=1,inplace=True)
  df=missingvalue(df)
  df=Onehotencoding(df)
  return df

def load_dataset(*, file_name: str) -> pd.DataFrame:
  logger.info("load_dataset calling...")
  dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
  transformed = pre_processing(dataframe)
  return transformed

def save_model(*, model_to_persist: Pipeline) -> None:
  logger.info("save_model calling...")
  # Prepare versioned save file name
  save_file_name = f"{config.app_config.model_save_file}{_version}.pkl"
  save_path = TRAINED_MODEL_DIR / save_file_name
  remove_old_model(files_to_keep=[save_file_name])
  joblib.dump(model_to_persist, save_path)

def remove_old_model(*, files_to_keep: List[str]) -> None:
  logger.info("remove_old_model calling...")
  do_not_delete = files_to_keep + ["__init__.py"]
  for model_file in TRAINED_MODEL_DIR.iterdir():
    if model_file.name not in do_not_delete:
      model_file.unlink()

def load_model(*, file_name: str):
  """Load a persisted model."""
  file_path = TRAINED_MODEL_DIR / file_name
  return joblib.load(filename=file_path)

def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
  logger.info("load_raw_dataset calling ....")
  dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
  return dataframe
