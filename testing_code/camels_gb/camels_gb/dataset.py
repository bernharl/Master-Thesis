from typing import List, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import numba


class CamelsTXT(Dataset):
    """Torch Dataset for basic use of data from the CAMELS data set.

    This data set provides meteorological observations and discharge of a given
    basin from the CAMELS data set.
    """

    def __init__(
        self,
        basin: str,
        features: List[str],
        camels_root: Path,
        seq_length: int = 365,
        period: str = None,
        dates: List = None,
        means: pd.Series = None,
        stds: pd.Series = None,
    ):
        """Initialize Dataset containing the data of a single basin.

        :param basin: 8-digit code of basin as string.
        :param seq_length: (optional) Length of the time window of
            meteorological input provided for one time step of prediction.
        :param period: (optional) One of ['train', 'eval']. None loads the
            entire time series.
        :param dates: (optional) List of pd.DateTimes of the start and end date
            of the discharge period that is used.
        :param means: (optional) Means of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_means() on the data set.
        :param stds: (optional) Stds of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_stds() on the data set.
        """
        self.basin = basin
        self.features = features
        self.camels_root = camels_root
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means
        self.stds = stds

        # load data into memory
        self.x, self.y = self._load_data()

        # store number of samples as class attribute
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """Load input and output data from text files."""
        df, area = load_forcing(self.basin, self.camels_root, self.features)
        df["discharge_spec"] = load_discharge(
            self.basin, self.camels_root, area, df.index.to_series()
        )
        # df.dropna(inplace=True)
        # print(df)

        if self.dates is not None:
            # If meteorological observations exist before start date
            # use these as well. Similiar to hydrological warmup period.
            if self.dates[0] - pd.DateOffset(days=self.seq_length) > df.index[0]:
                start_date = self.dates[0] - pd.DateOffset(days=self.seq_length)
            else:
                start_date = self.dates[0]
            df = df[start_date : self.dates[1]]
            # print(df)

        # if training period store means and stds
        if self.period == "train":
            self.means = df.mean()
            self.stds = df.std()

        # extract input and output features from DataFrame
        """x = np.array(
            [
                df["precipitation"].values,
                df["temperature"].values,
                df["peti"].values,
                df["humidity"].values,
                df["shortwave_rad"].values,
                df["longwave_rad"].values,
                df["windspeed"].values,
            ]
        ).T"""
        x = np.array([df[key].values for key in self.features]).T
        y = np.array([df["discharge_spec"].values]).T

        # normalize data, reshape for LSTM training and remove invalid samples
        x = self._local_normalization(x, variable="inputs")
        x, y = self.reshape_data(x, y, self.seq_length)

        if self.period == "train":
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # normalize discharge
            y = self._local_normalization(y, variable="output")

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == "inputs":
            # print(self.means)
            """means = np.array(
                [
                    self.means["precipitation"],
                    self.means["temperature"],
                    self.means["peti"],
                    self.means["humidity"],
                    self.means["shortwave_rad"],
                    self.means["longwave_rad"],
                    self.means["windspeed"],
                ]
            )
            stds = np.array(
                [
                    self.stds["precipitation"],
                    self.stds["temperature"],
                    self.stds["peti"],
                    self.stds["humidity"],
                    self.stds["shortwave_rad"],
                    self.stds["longwave_rad"],
                    self.stds["windspeed"],
                ]
            )"""
            means = np.array([self.means[key] for key in self.features])
            stds = np.array([self.stds[key] for key in self.features])
            feature = (feature - means) / stds
        elif variable == "output":
            feature = (feature - self.means["discharge_spec"]) / self.stds[
                "discharge_spec"
            ]
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == "inputs":
            """means = np.array(
                [
                    self.means["precipitation"],
                    self.means["temperature"],
                    self.means["peti"],
                    self.means["humidity"],
                    self.means["shortwave_rad"],
                    self.means["longwave_rad"],
                    self.means["windspeed"],
                ]
            )
            stds = np.array(
                [
                    self.stds["precipitation"],
                    self.stds["temperature"],
                    self.stds["peti"],
                    self.stds["humidity"],
                    self.stds["shortwave_rad"],
                    self.stds["longwave_rad"],
                    self.stds["windspeed"],
                ]
            )"""
            means = np.array([self.means[key] for key in self.features])
            stds = np.array([self.stds[key] for key in self.features])
            feature = feature * stds + means
        elif variable == "output":
            feature = (
                feature * self.stds["discharge_spec"] + self.means["discharge_spec"]
            )
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    @staticmethod
    @numba.njit
    def reshape_data(
        x: np.ndarray, y: np.ndarray, seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape matrix data into sample shape for LSTM training.

        :param x: Matrix containing input features column wise and time steps row wise
        :param y: Matrix containing the output feature.
        :param seq_length: Length of look back days for one day of prediction

        :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
        """
        num_samples, num_features = x.shape

        x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
        y_new = np.zeros((num_samples - seq_length + 1, 1))

        for i in range(0, x_new.shape[0]):
            x_new[i, :, :num_features] = x[i : i + seq_length, :]
            y_new[i, :] = y[i + seq_length - 1, 0]

        return x_new, y_new


def load_forcing(
    basin: str, camels_root: Path, features: List[str]
) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data of a specific basin.

    :param basin: 8-digit code of basin as string.

    :return: pd.DataFrame containing the meteorological forcing data and the
        area of the basin as integer.
    """
    path = (
        camels_root
        / "timeseries"
        / f"CAMELS_GB_hydromet_timeseries_{basin}_19701001-20150930.csv"
    )
    exclude = ["pet", "discharge_vol", "discharge_spec"]
    df = pd.read_csv(path).dropna()
    columns = df.columns.values
    for feature in columns:
        if feature not in features and feature != "date":
            exclude.append(feature)
    df = df.drop(exclude, axis=1)
    dates = pd.to_datetime(df["date"])
    year = []
    day = []
    month = []
    hour = np.ones(len(dates)) * 12
    for date in df["date"]:
        date_split = date.split("-")
        year.append(int(date_split[0]))
        month.append(int(date_split[1]))
        day.append(int(date_split[2]))
    df["Year"] = np.array(year)
    df["Mnth"] = np.array(month)
    df["Day"] = np.array(day)
    df["Hr"] = hour
    df["Date"] = dates
    df.drop("date", axis=1, inplace=True)
    df.set_index("Date", inplace=True)
    return df, 1


def load_discharge(
    basin: str, camels_root: Path, area: int, dates: pd.Series
) -> pd.Series:
    """Load the discharge time series for a specific basin.

    :param basin: 8-digit code of basin as string.
    :param area: int, area of the catchment in square meters

    :return: A pd.Series containng the catchment normalized discharge.
    """

    discharge_path = (
        camels_root
        / "timeseries"
        / f"CAMELS_GB_hydromet_timeseries_{basin}_19701001-20150930.csv"
    )
    df = pd.read_csv(discharge_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df["discharge_spec"]
    df.fillna(0, inplace=True)
    df = pd.to_numeric(df)
    df = df[dates[0] : dates[-1]]
    return df
