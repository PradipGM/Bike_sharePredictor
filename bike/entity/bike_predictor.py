import os
import sys

from bike.exception import bikeException
from bike.util.util import load_object

import pandas as pd


class HousingData:

    def __init__(self,
                season: object,
                yr: object,
                mnth: object,
                hr: int,
                holiday: object,
                weekday: object,
                workingday: object,
                weathersit: object,
                temp: float,
                atemp: float,
                hum: float,
                windspeed: float,
                cnt: int= None
                 ):
        try:
            self.season =season
            self.yr=yr
            self.mnth=mnth
            self.hr=hr
            self.holiday=holiday
            self.weekday=weekday
            self.workingday=workingday
            self.weathersit=weathersit
            self.temp=temp
            self.atemp=atemp
            self.hum=hum
            self.windspeed=windspeed
            self.cnt = cnt

        except Exception as e:
            raise bikeException(e, sys) from e

    def get_housing_input_data_frame(self):

        try:
            housing_input_dict = self.get_housing_data_as_dict()
            return pd.DataFrame(housing_input_dict)
        except Exception as e:
            raise bikeException(e, sys) from e

    def get_housing_data_as_dict(self):
        try:
            input_data = {
                "season": [self.season],
                "year": [self.yr],
                "month": [self.mnth],
                "hour": [self.hr],
                "holiday": [self.holiday],
                "weekday": [self.weekday],
                "workingday": [self.workingday],
                "weathersit": [self.weathersit],
                "temp": [self.temp],
                "atemp": [self.atemp],
                "humidity": [self.hum],
                "windspeed": [self.windspeed],
            }
            return input_data
        except Exception as e:
            raise bikeException(e, sys)


class HousingPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise bikeException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise bikeException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            cnt = model.predict(X)
            return cnt
        except Exception as e:
            raise bikeException(e, sys) from e