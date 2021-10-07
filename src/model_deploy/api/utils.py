# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is a boilerplate pipeline 'score'
generated using Kedro 0.16.6
"""
import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

PROJ_DIR = Path(__file__).parent.parent.parent.parent


class ModelInput(BaseModel):
    uuid: int or str
    engines: float
    passenger_capacity: int
    crew: float
    d_check_complete: bool
    moon_clearance_complete: bool


class MultiModelInput(BaseModel):
    model_inputs: List[ModelInput]


class ModelOutput(BaseModel):
    uuid: int or str
    score: float


class MultiModelOutput(BaseModel):
    model_outputs: List[ModelOutput]


def _parse_latest_version(filepath: str or Path):
    original_filepath = filepath
    _filepath = filepath
    _filepath = Path(_filepath) / "*" / "*"
    _filepath = str(_filepath)
    versions = glob.glob(_filepath)

    if versions == []:
        return original_filepath
    else:
        latest_version = max(versions)
        return latest_version


def get_regressor(
    filepath: str = "data/06_models/regressor.pickle", root: Path = PROJ_DIR
):
    filepath = root / filepath
    if not filepath.exists():
        return None

    latest_version = _parse_latest_version(filepath)
    regressor = pd.read_pickle(latest_version)
    return regressor


def get_features(filepath: str = "conf/base/parameters.yml", root: Path = PROJ_DIR):
    filepath = root / filepath
    if not filepath.exists():
        return None

    parameters = yaml.safe_load(open(filepath, "r"))
    features = parameters["features"]
    return features


def score_pandas(
    regressor: LinearRegression, df: pd.DataFrame, features: list
) -> np.ndarray:
    """Simple method to score pandas dataframe with serialized model

    Args:
        regressor (LinearRegression): Serialized model
        df (pd.DataFrame): Data with features in the column names
        features (list): Columns to subset data, in order, for scoring by regressor

    Returns:
        pd.DataFrame: Single column dataframe with scores
    """
    X = df[features]
    y_hat = regressor.predict(X)
    return y_hat


def score_json(
    regressor: LinearRegression, model_input: dict, features: list
) -> np.ndarray:
    """Take a model regressor object, model input data, and make prediction

    Args:
        regressor (LinearRegression): Serialized model
        model_input (dict): Singleton
        features (list): Column names of regressor, in order

    Returns:
        pd.DataFrame: Single column dataframe with scores
    """
    df = _json_to_pandas(model_input)
    y_hat = score_pandas(regressor, df, features)
    return y_hat


def _json_to_pandas(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    return df
