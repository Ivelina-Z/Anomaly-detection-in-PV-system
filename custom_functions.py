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
    
