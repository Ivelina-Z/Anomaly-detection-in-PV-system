import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# preprocessing - dataset


def delete_additional_header(data):
    """
    If additional headers are present in the dataframe those will be dropped.
    The function takes one argument - the dataframe and returns it without the additional records.

    :param data: DataFrame
    :return: DataFrame with only one line of headers.
    """
    data = data.drop(data[data['SiteID'] == 'SiteID'].index)
    return data


def convert_data_types(data, columns_type):
    """
    Converts the datatype of all columns in `columns_type` if it's different.
    If there's missing data in a column it's converted to float.

    :param data: DataFrame
    :param columns_type: dictionary with columns' name as key and desired datatype as value
    :return: DataFrame with converted data types
    """
    for column, column_type in columns_type.items():
        try:
            if columns_type[column] != data[column].dtypes:
                data[column] = data[column].astype(columns_type[column])
        except ValueError:
            data[column] = data[column].astype(float)
    return data


def missing_values_percent(dataframe, year_column):
    """
    Calculates the missing values percentage by year for each column in the data.

    :param dataframe: DataFrame
    :param year_column: the column with year parameter to group by
    :return: DataFrame with missing values percentage per year
    """
    missing_values = dataframe.groupby(dataframe[year_column].dt.year).apply(
        lambda x: (x.isna().sum() / (x.size / dataframe.shape[1])) * 100)
    return missing_values


# preprocessing - pipeline 

def encoding_error_code(data):
    """
    Encode with -1, where data is not equal to 0, and 1 - where it is.

    :param data: numpy array
    :return: DataFrame with encoded data
    """
    encoded_data = np.where(data != 0, -1, 1)
    return pd.DataFrame(encoded_data)


def encoding_sensor_record(data):
    """
    Encode with 1, where data is not equal to 0, and 0 - where it is.

    :param data: numpy array
    :return: DataFrame with encoded data
    """
    encoded_data = np.where(data != 0, 1, 0)
    return pd.DataFrame(encoded_data)


def temperature_difference_module(data):
    """
    Calculates difference between the last element(ambient temperature) of data and
    the rest of the data.

    :param data: numpy array with last column - ambient temperature
    :return DataFrame with recalculated values
    """
    ambient_temp = data[:, -1]
    result = {}
    for idx in range(len(data[0]) - 1):
        module_temp = data[:, idx]
        transformed_module_temp = module_temp - ambient_temp
        result[f'transformed_module{idx}_temp'] = transformed_module_temp
    return pd.DataFrame.from_dict(result)


def temperature_difference_inverter(data):
    """
    Calculates difference between the last element(ambient temperature) of data and
    the rest of the data. Returns the resulting temperature difference and the ambient 
    temperature

    :param data: numpy array with last column - ambient temperature
    :return DataFrame with the ambient temperature and inverters' temperature differences
    """
    ambient_temp = data[:, -1]
    result = {'ambient_temp': ambient_temp}
    for idx in range(len(data[0]) - 1):
        inverter_temp = data[:, idx]
        transformed_inverter_temp = inverter_temp - ambient_temp
        result[f'transformed_inv{idx}_temp'] = transformed_inverter_temp
    return pd.DataFrame.from_dict(result)
