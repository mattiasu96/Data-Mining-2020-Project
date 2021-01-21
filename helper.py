import pandas as pd
import datetime as dt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class data_container:
    """container class for a dataset"""

    def __init__(self, dataset, drop=True):

        self._dataset = dataset

        df = self._dataset.copy()

        df['DATE'] = pd.to_datetime(df['DATE'])

        weather_names = ['_temperature_', '_rain_mm_', '_humidity_', '_wind_speed_', '_pressure_']

        for name in weather_names:
            df['max' + name + 'prev14d'] = df['max' + name + 'prev7d']
            df['max' + name + 'prev7d_shift'] = df['max' + name + 'prev7d'].shift(7, axis=0)
            df.loc[df.DATE > '2019-04-16', 'max' + name + 'prev14d'] = df[df['DATE'] > '2019-04-16'][
                ['max' + name + 'prev7d', 'max' + name + 'prev7d_shift']].max(axis=1)
            df = df.drop(columns=['max' + name + 'prev7d_shift'])

            df['min' + name + 'prev14d'] = df['min' + name + 'prev7d']
            df['min' + name + 'prev7d_shift'] = df['min' + name + 'prev7d'].shift(7, axis=0)
            df.loc[df.DATE > '2019-04-16', 'min' + name + 'prev14d'] = df[df['DATE'] > '2019-04-16'][
                ['min' + name + 'prev7d', 'min' + name + 'prev7d_shift']].min(axis=1)
            df = df.drop(columns=['min' + name + 'prev7d_shift'])

            df['mean' + name + 'prev14d'] = df['mean' + name + 'prev7d']
            df['mean' + name + 'prev7d_shift'] = df['mean' + name + 'prev7d'].shift(7, axis=0)
            df.loc[df.DATE > '2019-04-16', 'mean' + name + 'prev14d'] = df[df['DATE'] > '2019-04-16'][
                ['mean' + name + 'prev7d', 'mean' + name + 'prev7d_shift']].mean(axis=1)
            df = df.drop(columns=['mean' + name + 'prev7d_shift'])

        persistance_names = ['equipment', 'fire/smoke', 'ge', 'power', 'temperature']

        for name in persistance_names:
            df[name + '_max_persistance_prev14d'] = df[name + '_max_persistance_prev7d']
            df[name + '_max_persistance_prev7d_shift'] = df[name + '_max_persistance_prev7d'].shift(7, axis=0)
            df.loc[df.DATE > '2019-04-16', name + '_max_persistance_prev14d'] = df[df['DATE'] > '2019-04-16'][
                [name + '_max_persistance_prev7d', name + '_max_persistance_prev7d_shift']].max(axis=1)
            df = df.drop(columns=[name + '_max_persistance_prev7d_shift'])

            df[name + '_min_persistance_prev14d'] = df[name + '_min_persistance_prev7d']
            df[name + '_min_persistance_prev7d_shift'] = df[name + '_min_persistance_prev7d'].shift(7, axis=0)
            df.loc[df.DATE > '2019-04-16', name + '_min_persistance_prev14d'] = df[df['DATE'] > '2019-04-16'][
                [name + '_min_persistance_prev7d', name + '_min_persistance_prev7d_shift']].min(axis=1)
            df = df.drop(columns=[name + '_min_persistance_prev7d_shift'])

            df[name + '_mean_persistance_prev14d'] = df[name + '_mean_persistance_prev7d']
            df[name + '_mean_persistance_prev7d_shift'] = df[name + '_mean_persistance_prev7d'].shift(7, axis=0)
            df.loc[df.DATE > '2019-04-16', name + '_mean_persistance_prev14d'] = df[df['DATE'] > '2019-04-16'][
                [name + '_mean_persistance_prev7d', name + '_mean_persistance_prev7d_shift']].mean(axis=1)
            df = df.drop(columns=[name + '_mean_persistance_prev7d_shift'])

        self._temperature_cols = [col for col in df.columns if 'temperature' in col and 'alarms' not in col and 'persistance' not in col]
        self._humidity_cols = [col for col in df.columns if 'humidity' in col]
        self._wind_cols = [col for col in df.columns if 'wind' in col]
        self._rain_cols = [col for col in df.columns if 'rain' in col]
        self._alarm_cols = [col for col in df.columns if 'alarm' in col]
        self._pressure_cols = [col for col in df.columns if 'pressure' in col]
        self._persistance_cols = [col for col in df.columns if 'persistance' in col]
        self._aircon_cols = [col for col in df.columns if 'wo_prev' in col]

        self._numerical_features = self._temperature_cols + self._humidity_cols + \
                                   self._wind_cols + self._rain_cols + self._alarm_cols + \
                                   self._pressure_cols + self._persistance_cols + self._aircon_cols

        # process data feature
        df['month'] = df['DATE'].dt.month
        df['day'] = df['DATE'].dt.day
        df['day' + '_sin'] = np.sin(2 * np.pi * df['day'] / df['day'].max())
        df['day' + '_cos'] = np.cos(2 * np.pi * df['day'] / df['day'].max())

        cell_type = [col for col in df.columns if 'CELL_TYPE' in col]
        df = df.drop(columns=cell_type)
        df = df.drop(columns='day')

        if drop == True:
            # drop unused columns
            df = df.drop(columns=['DATE', 'SITE_ID'])

        self._prepared_dataset = df

    @property
    def prepared_dataset(self):
        """returns a copy of the dataset ready to be passed to the model"""
        return self._prepared_dataset

    @property
    def dataset(self):
        return self._dataset

    @property
    def numerical_features(self):
        return self._numerical_features

    def features_by_type(self, features=None, dataset_type='prepared'):
        """returns a DataFrame with the selected features (temperature, humidity, wind, rain)"""
        if features == None:
            return None

        elif features in ['temperature', 'humidity', 'wind', 'rain', 'pressure']:
            cols = [i for i in self.dataset.columns if features in i]

            if dataset_type == 'prepared':
                return self.prepared_dataset[cols]

            elif dataset_type == 'raw':
                return self.dataset[cols]

        return None

    def run_PCA(self, features=None, components=12, append=False):
        """run PCA on the requested normalized features features (temperature, humidity, wind, rain) with the requested number of components, returns the principals components and the explained variance"""
        normalized_features = self.features_by_type(features)

        pca = PCA(n_components=components)
        pca.fit(normalized_features)
        PCA_components = pca.transform(normalized_features)
        explained_variance = pca.explained_variance_ratio_

        if append == True:
            columns = [features + '_pca_%i' % i for i in range(components)]
            print(columns)
            to_drop = [i for i in self.dataset.columns if features in i]
            self._prepared_dataset = self._prepared_dataset.drop(columns=to_drop)
            print(columns)

            self._prepared_dataset[columns] = PCA_components
            print(self._prepared_dataset[columns].shape)
        return PCA_components, explained_variance

    def normalize_dataset(self):
        scaler = StandardScaler()
        features = self.numerical_features
        scaler.fit(self.prepared_dataset[features])
        normalized_features = scaler.transform(self.prepared_dataset[features])
        self.prepared_dataset[features] = normalized_features
        return self.prepared_dataset

    def compute_14d_features(self):
        df_14d = self.prepared_dataset[
            [col for col in self.prepared_dataset.columns if '3d' not in col and '7d' not in col]]
        return df_14d


if __name__ == "__main__":
    # change the file path with the needed one
    train = pd.read_csv('/kaggle/input/dmtm-dataset/train.csv')
    container = data_container(train)
    df14 = container.compute_14d_features()
    print(df14.columns)