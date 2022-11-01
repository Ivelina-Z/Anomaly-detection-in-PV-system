import pandas as pd
import matplotlib.pyplot as plt
import datetime 
from math import ceil


# DATA CLEANING
def delete_additional_header(dataframe):
    ''' 
    If additional headers are present in the dataframe those will be dropped.
    The function takes one argument - the dataframe and returns it without the additional records.
    '''
    dataframe = dataframe.drop(dataframe[dataframe['SiteID'] == 'SiteID'].index)
    return dataframe


def convert_data_types(dataframe, columns_type: dict):
    for column, column_type in columns_type.items():
        try:
            if columns_type[column] != dataframe[column].dtypes:
                dataframe[column] = dataframe[column].astype(columns_type[column])
        except ValueError: #raise Error instead except
            dataframe[column] = dataframe[column].astype(float)
    return dataframe


def missing_values_percent(dataframe, year_column):
    missing_values = dataframe.groupby(dataframe[year_column].dt.year).apply(lambda x: (x.isna().sum() / (x.size / dataframe.shape[1])) * 100)
    return missing_values

 
def drop_years(dataframe, years: list):
    for year in years:
        dataframe = dataframe.drop(dataframe[dataframe['timestamp'].dt.year == year].index)
        dataframe = dataframe.reset_index(drop=True)
    return dataframe
    

# PLOT 

def plot_time_series(dataframe, time_range: tuple, y: list, step):
    if step == 'hour':
        time = dataframe['timestamp'][dataframe['timestamp'].dt.date == datetime.date(*time_range)]
        y_feature = [dataframe[feature][dataframe['timestamp'].dt.date == datetime.date(*time_range)] for feature in y]
    elif step == 'day':
        year, month = time_range
        time = dataframe['timestamp'][(dataframe['timestamp'].dt.year == year) & (dataframe['timestamp'].dt.month == month)]
        y_feature = [dataframe[feature][(dataframe['timestamp'].dt.year == year) & (dataframe['timestamp'].dt.month == month)] for feature in y]
    elif step == 'month':
        time = dataframe['timestamp'][dataframe['timestamp'].dt.year == time_range]
        y_feature = [dataframe[feature][dataframe['timestamp'].dt.year == time_range] for feature in y]

    [plt.plot(time, feature, alpha = 0.8) for feature in y_feature]
    plt.title(f'Time series of {", ".join(y)} by {step}')
    plt.xlabel(step)
    plt.xticks(rotation=90)
    
    if len(y) > 1:
        plt.legend(y)
        plt.ylabel('y')
    else:
        plt.ylabel(", ".join(y))

    plt.show()
    
    
def subplot_boxplots(dataframe, columns):
    rows = ceil(len(columns) / 3)
    fig, ax = plt.subplots(rows, 3)
    fig.set_figheight(30)
    fig.set_figwidth(30)
    for row in range(rows):
        for col in range(3):
            try:
                ax[row, col].boxplot(dataframe[columns[col + row * 3]], showmeans = True, meanline=True)
                ax[row, col].set_title(f'Boxplot for {columns[col + row * 3]}')
            except IndexError:
                break
    plt.show()



# DATA EXAMINATION

def get_extreme_values(dataframe, columns: list):
    wiskers = {}
    for column in columns:
        interquartile_range = dataframe[column].describe()['75%'] - dataframe[column].describe()['25%']
        upper_wisker = dataframe[column].describe()['75%'] + 1.5 * interquartile_range
        lower_wisker = dataframe[column].describe()['25%'] - 1.5 * interquartile_range
        wiskers[column] = [upper_wisker, lower_wisker]
    return wiskers


def set_extreme_threshold(wiskers: dict, threshold: dict):
    for attribute in threshold:
        if 'lower' in threshold[attribute]:
            wiskers[attribute][1] = threshold[attribute]['lower']
        elif 'upper' in threshold[attribute]:
            wiskers[attribute][0] = threshold[attribute]['upper']
    return wiskers


def get_outlier_index(dataframe, columns: list, wiskers: dict, outlier_indexes=None, directions={}):
    """If directions=None both values below lower wisker and above upper wisker will be considered outliers."""
    if not outlier_indexes:
        outlier_indexes = set()
    for column in columns:
        upper_wisker, lower_wisker = wiskers[column]
        if column in directions and directions[column] == 'upper':
            outlier_index = dataframe[dataframe[column] > upper_wisker].index
        elif column in directions and directions[column] == 'lower':
            outlier_index = dataframe[dataframe[column] < lower_wisker].index
        elif column not in directions:
            outlier_index = dataframe[(dataframe[column] < lower_wisker) | (dataframe[column] > upper_wisker)].index
        outlier_indexes.update(outlier_index)
    return outlier_indexes