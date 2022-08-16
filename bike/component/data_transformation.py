from cgi import test
from sklearn import preprocessing
from bike.exception import bikeException
from bike.logger import logging
from bike.entity.config_entity import DataTransformationConfig
from bike.entity.artifact_entity import DataIngestionArtifact, \
    DataValidationArtifact, DataTransformationArtifact
import sys, os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.impute import SimpleImputer
import pandas as pd
from bike.constant import *
from bike.util.util import read_yaml_file, save_object, save_numpy_array_data, load_data


#   longitude: float
#   latitude: float
#   housing_median_age: float
#   total_rooms: float
#   total_bedrooms: float
#   population: float
#   households: float
#   median_income: float
#   median_house_value: float
#   ocean_proximity: category
#   income_cat: float


'''class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 dteday_ix=0,
                 season_ix=1,
                 yr_ix=2,
                 mnth_ix=3,
                 hr_ix=4,
                 holiday_ix=5,
                 weekday_ix=6,
                 workingday_ix=7,
                 weathersit_ix=8,
                 temp_ix=9,
                 atemp_ix=10,
                 hum_ix=11,
                 windspeed_ix=12, columns=None):
        
        FeatureGenerator Initialization
        add_bedrooms_per_room: bool
        total_rooms_ix: int index number of total rooms columns
        population_ix: int index number of total population columns
        households_ix: int index number of  households columns
        total_bedrooms_ix: int index number of bedrooms columns
        
        try:
            self.columns = columns
            if self.columns is not None:
                #dteday_ix = self.columns.index(COLUMN_dteday)
                season_ix = self.columns.index(COLUMN_season)
                yr_ix = self.columns.index(COLUMN_yr)
                mnth_ix = self.columns.index(COLUMN_mnth)
                hr_ix = self.columns.index(COLUMN_hr)
                holiday_ix = self.columns.index(COLUMN_holiday)
                weekday_ix = self.columns.index(COLUMN_weekday)
                workingday_ix = self.columns.index(COLUMN_workingday)
                weathersit_ix = self.columns.index(COLUMN_weathersit)
                temp_ix = self.columns.index(COLUMN_temp)
                atemp_ix = self.columns.index(COLUMN_atemp)
                hum_ix = self.columns.index(COLUMN_hum)
                windspeed_ix = self.columns.index(COLUMN_windspeed)
            #self.dteday_ix = dteday_ix
            self.season_ix = season_ix
            self.yr_ix = yr_ix
            self.mnth_ix = mnth_ix
            self.hr_ix = hr_ix
            self.holiday_ix = holiday_ix
            self.weekday_ix = weekday_ix
            self.workingday_ix = workingday_ix
            self.weathersit_ix = weathersit_ix
            self.temp_ix= temp_ix
            self.atemp_ix = atemp_ix
            self.hum_ix = hum_ix
            self.windspeed_ix = windspeed_ix
        except Exception as e:
            raise bikeException(e, sys) from e
    def fit(self, X, y=None):
        return self
    def dtype_change(self,df):
        try:
            logging.info(f"{'=' * 20}Ashutosh log started.{'=' * 20} ")
            df['self.season_ix'] = df['self.season_ix'].astype('object')
            df['self.mnth_ix'] = df['self.mnth_ix'].astype('object')
            df['self.weekday_ix'] = df['self.weekday_ix'].astype('object')
            df['self.weathersit_ix'] = df['self.weathersit_ix'].astype('object')
            df['self.workingday_ix'] = df['self.workingday_ix'].astype('object')
            df['yr_ix'] = df['yr_ix'].astype('object')
            df['self.holiday_ix'] = df['self.holiday_ix'].astype('object')
            return df
        except Exception as e:
            raise bikeException(e, sys) from e
    def transform(self, X, y=None):
        try:
            room_per_household = X[:, self.total_rooms_ix] / \
                                 X[:, self.households_ix]
            population_per_household = X[:, self.population_ix] / \
                                       X[:, self.households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, self.total_bedrooms_ix] / \
                                    X[:, self.total_rooms_ix]
                generated_feature = np.c_[
                    X, room_per_household, population_per_household, bedrooms_per_room]
            else:
                generated_feature = np.c_[
                    X, room_per_household, population_per_household]
            return generated_feature
        except Exception as e:
            raise bikeException(e, sys) from e'''
class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'=' * 20}Data Transformation log started.{'=' * 20} ")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise bikeException(e, sys) from e

    '''def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path
            dataset_schema = read_yaml_file(file_path=schema_file_path)
            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
            num_pipeline = Pipeline(steps=[
               #('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ]
            )
            cat_pipeline = Pipeline(steps=[
                #('impute', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ])
            return preprocessing
        except Exception as e:
            raise bikeException(e, sys) from e
   # def fit(self, X, y=None):
      #  return self'''

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Description: Function is used to standardize the data(train and test both) with standard scaler and
                     saved it in array form
        return: is_transformed: Ture for transform and False for not transform
                message: message after data transform completed
                transformed_train_file_path: Path of transformed train file
                transformed_test_file_path: Path of transformed test file
                preprocessed_object_file_path: Save preprocessed object for predict data
        """
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = StandardScaler()

            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)

            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)


            target_column_name = schema[TARGET_COLUMN_KEY]
           # drop = schema[DROP]
            #drop1 = schema[DROP1]
            #drop2 = schema[DROP2]
            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
           # input_feature_train_df1= input_feature_train_df.drop(columns=[drop], axis=1)
            #input_feature_train_df2=input_feature_train_df.drop(columns=[drop1],axis=1)
            #input_feature_train_df3=input_feature_train_df2.drop(columns=[drop2],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            #input_feature_test_df1= input_feature_test_df.drop(columns=[drop], axis=1)
            #input_feature_test_df2=input_feature_test_df.drop(columns=[drop1],axis=1)
            #input_feature_test_df3=input_feature_test_df2.drop(columns=[drop2],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df,target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv", ".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")

            save_numpy_array_data(file_path=transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path, array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data transformation successfull.",
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      transformed_test_file_path=transformed_test_file_path,
                                                                      preprocessed_object_file_path=preprocessing_obj_file_path

                                                                      )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise bikeException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Data Transformation log completed.{'=' * 20} \n\n")