import datetime 
import matplotlib.pyplot as plt

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

    [plt.plot(time, feature) for feature in y_feature]
    plt.title(f'Time series of {", ".join(y)} by {step}')
    plt.xlabel(step)
    plt.xticks(rotation=90)
    
    if len(y) > 1:
        plt.legend(y)
        plt.ylabel('y')
    else:
        plt.ylabel(", ".join(y))

    plt.show()