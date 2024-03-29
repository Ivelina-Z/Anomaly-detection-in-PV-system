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


# Label anomalies

def get_whiskers_values(data):
    """
    data: array
    """
    whiskers = {}
    for column in range(len(data[0])):
        interquartile_range = np.quantile(data[:, column], 0.75) - np.quantile(data[:, column], 0.25)
        upper_extreme = np.quantile(data[:, column], 0.75) + 1.5 * interquartile_range
        lower_extreme = np.quantile(data[:, column], 0.25) - 1.5 * interquartile_range
        whiskers[column] = [upper_extreme, lower_extreme]
    return whiskers


def get_outlier_index(data, whiskers: dict, threshold_conditions={}):
    outlier_indexes = set()
    
    for column in range(len(data[0])):
        if column in threshold_conditions:
            outlier_index = np.where(threshold_conditions[column])[0]
        else: 
            upper_whisker, lower_whisker = whiskers[column]
            outlier_index = np.where((data[:, column] < lower_whisker) | (data[:, column] > upper_whisker))[0]
                                     
        outlier_indexes.update(list(outlier_index))
    return outlier_indexes


def labeling(data, threshold_conditions):
    anomaly_indexes = get_outlier_index(
        data,
        get_whiskers_values(data),
        threshold_conditions)
    expected_anomaly = np.ones([len(data), ])
    expected_anomaly[np.array(list(anomaly_indexes))] = -1
    return expected_anomaly



def threshold_conditions(data, timestamp, columns):
    threshold_conditions = {
             columns['ambient_temp']: (data[:, columns['ambient_temp']] < -10),
             columns['inverter_temp_diff']: (data[:, columns['inverter_temp_diff']] < 0),
             columns['inverter_error_code']: (data[:, columns['inverter_error_code']] == -1),
             columns['relative_humidity_record_available']: (data[:, columns['relative_humidity_record_available']] == 0),
             columns['wind_direction_record_available']: (data[:, columns['wind_direction_record_available']] == 0),
             columns['wind_speed_record_available']: (data[:, columns['wind_speed_record_available']] == 0),
             columns['ac_current']: ((data[:, columns['ac_current']] <= 0) & (data[:, columns['poa_irradiance']] >= 50)),
             columns['ac_voltage']: ((data[:, columns['ac_voltage']] <= 0) & (data[:, columns['poa_irradiance']] >= 50)),
             columns['dc_current']: ((data[:, columns['dc_current']] <= 0) & (data[:, columns['poa_irradiance']] >= 50)),
             columns['dc_voltage']: ((data[:, columns['dc_voltage']] <= 0) & (data[:, columns['poa_irradiance']] >= 50)),
             columns['poa_irradiance']: ((data[:, columns['poa_irradiance']] <= 0) & (timestamp.dt.hour >= 8) & (timestamp.dt.hour < 17))
    }
    return threshold_conditions

# Modelling

def year_data_split(data, year, target_column=None, drop_time = True):
    data_attributes = data[data['timestamp'].dt.year == year]    
    labels = data_attributes[target_column]
    data_attributes = data_attributes.drop([target_column], axis=1)
    if drop_time:
        data_attributes = data_attributes.drop(['timestamp'], axis=1)
    return data_attributes, labels


def contamination(labels, anomaly_label = -1):
    """
        Calculates the amount of anomalies present in the data based on the automatic annotation       
    """
    return np.count_nonzero(labels == anomaly_label) / len(labels)
    
# plots


def subplot_boxplots(dataframe, columns, system_id=None):
    """
    Plot multiple boxplots. 
    
    :param dataframe: data
    :param columns: features to plot 
    """
    rows = int(len(columns) / 3)
    fig, ax = plt.subplots(rows, 3)
    if system_id:
        fig.suptitle(f'System {system_id} boxplots')
    fig.set_figheight(30)
    fig.set_figwidth(30)
    for row in range(rows):
        for col in range(3):
            try:
                ax[row, col].boxplot(dataframe[columns[col + row * 3]].dropna())
                ax[row, col].set_title(f'Boxplot for {columns[col + row * 3]}')
            except IndexError:
                break
    plt.show()

    
def plot_day(data, day, features: list, ax=None):
    if ax is None:
        ax = plt.gca()
        ax.legend(x)   
    time = data['timestamp'][data['timestamp'].dt.date == day]
    y = [data[feature][data['timestamp'].dt.date == day] for feature in features]
    [ax.plot(time, feature) for feature in y]
    ax.tick_params(
        axis='x',         
        which='both',      
        bottom=False,      
        top=False,         
        labelbottom=False)    
    ax.set_xticks(time, rotation = 90)
    ax.set_xlabel('time')
    ax.set_ylabel(", ".join(features))
    return(ax)


def plot_year(data, years: list, feature, ax=None):
    if ax is None:
        ax = plt.gca()
        ax.legend(x)   
    time = list(range(1, 13))
    y = [data[feature][data['timestamp'].dt.year == year].groupby(data['timestamp'].dt.month).mean() for year in years]
    [ax.plot(time, feature) for feature in y]
    ax.set_xlabel('time')
    ax.set_ylabel(feature)
    return(ax)


def plot_cv_results(cv_results: pd.DataFrame, hyperparameter, x=['mean_test_score'], std=True, ax=None):
    if ax is None:
        ax = plt.gca()
        ax.legend(x)    
    [ax.plot(cv_results[hyperparameter].unique(), cv_results.groupby(hyperparameter)[feature].mean()) for feature in x]
    ax.set_xticks(cv_results[hyperparameter].unique())
    ax.set_xlabel(hyperparameter)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel('f2-score, anomaly class')    
    if std:
        [ax.fill_between(
            cv_results[hyperparameter].unique(),
            cv_results.groupby(hyperparameter)[feature].mean() - cv_results.groupby(hyperparameter)['std_test_score'].mean(),
            cv_results.groupby(hyperparameter)[feature].mean() + cv_results.groupby(hyperparameter)['std_test_score'].mean(),
            alpha=0.1
    ) for feature in x] 
    return(ax)


def plot_validation_curve(train_scores, test_scores, param_range, param, std=False, ax=None):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if ax is None:
        ax = plt.gca()
        ax.legend(x)    
    ax.set_xlabel(param)
    ax.set_xticks(param_range)
    ax.set_ylabel("f2 score, anomaly class")
    ax.set_ylim(0.0, 1.1)
    ax.plot(param_range, train_scores_mean, label="Training score", color="darkorange")
    ax.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy")
    if std:
        ax.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.2,
            color="darkorange"
        )
        
        ax.fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.2,
            color="navy"
        )
    return(ax)


def plot_learning_curve_time(fit_times, train_size, std=False, ax=None):
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    if ax is None:
        ax = plt.gca()
        ax.legend(x)    

    if std:
        ax.fill_between(
            train_size,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.1,
            color="r"
        )
    ax.plot(train_size, fit_times_mean, "o-", color="r")
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('fit time')
    return(ax)
    
    
def plot_learning_curve(train_scores, test_scores, train_size, std=False, ax=None):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if ax is None:
        ax = plt.gca()
        ax.legend(x) 
        
    if std:
        ax.fill_between(
            train_size,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r"
        )
        ax.fill_between(
                train_size,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.1,
                color="g"
        )
        
    ax.plot(train_size, train_scores_mean, "o-", color="r", label="Training score")
    ax.plot(train_size, test_scores_mean, "o-", color="g", label="Cross-validation score")
    
    ax.set_xlabel('Number of samples')
    ax.set_xticks(train_size)

    ax.set_ylabel('f2-score of the anomaly class')
    ax.set_ylim(0.0, 1.1)
    return(ax)
    

def plot_test_results(years, results:dict, model_contamination=0.5):
    plt.title(f'{", ".join(results.keys())} percentage on testing data')
    [plt.plot(years, score, label=metrics_name ) for metrics_name, score in results.items()]
    plt.hlines(model_contamination, xmin = years[0], xmax = years[-1], linestyles = '--', color = 'red', label = 'train contamination')

    plt.xticks(years)
    plt.xlabel('year')

    plt.ylabel('score/contamination')
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=5)
    plt.show()


def plot_confusion_matrices(conf_matrix, classes, data_contamination):
    fig, axes = plt.subplots(1, len(conf_matrix), figsize=(20, 5))
    for idx, result in enumerate(conf_matrix):
        disp = ConfusionMatrixDisplay(result, display_labels=classes)
        disp.plot(ax=axes[idx], xticks_rotation=90, cmap='Reds')
        disp.im_.colorbar.remove()
        disp.ax_.set_title(f'{data_contamination[idx]:.2f}% contamination')
        if idx != 0:
            disp.ax_.set_ylabel('')
        if idx != len(conf_matrix) // 2:
            disp.ax_.set_xlabel('')

    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)
    plt.show()
