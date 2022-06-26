# File is currently being used to predict staggered PRI.
from xmlrpc.client import MAXINT
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def adjust_answers(answers, unique_array):
    """
    Method to take the answer vecotrs and make convert to numerical values. The method will find the index of the highest value 
    in the answer vector and find its index. That index corresponds to the numerical value in the unique_array at the same index.

    Parameters: 
        - answers: array of all answer vectors.
        - unique_array: array of x locations from get_x_locs function 
    Returns: 
        - adjusted: array or list like iterable of all the asnwer vectors transformed to have 0's and 1's
    """
    adjusted = np.zeros(len(answers))
    for i in range(len(answers)):
        # ans is a vector
        adjusted[i] = unique_array[np.argmax(answers[i][:])]

    return adjusted


def gather_metrics(history):
    """
    Helper method to grab relevant metrics from tensorflow history object

    Parameters: 
    - history: tensorflow history object

    Returns: 
    - relevant metrics
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    eps = range(1, len(acc) + 1)
    return acc, val_acc, loss, val_loss, eps


def get_x_locs(corrupted, plot=False, num_bins=1000, prom=0.012, dist=20):
    """
    Method to get the x locations for the peaks of the distribution histogram. Histogram is computed using observed values 

    Parameters: 
        - corrupted - array -  corrupted data
        - plot - boolean - tells method to plot the histogram with peaks highlighted. Default = False
        - num_buns - int -  number of bins for histogram. Default = 1000
        - prom - float - prominence level for a value to be considered a peak. Default = 0.012
        - dist - int - minimum distance between 2 peaks. Default = 20 

    returns: 
        - x_locs: array of x locs
    """
    counts, edges = np.histogram(corrupted, bins=num_bins, density=True)
    peaks = find_peaks(counts, prominence=prom, distance=dist)[0]
    x_locs = []
    for j in peaks:
        x_locs.append(edges[j])

    if plot:
        plt.bar(edges[:-1], counts)
        for loc in x_locs:
            plt.axvline(loc, color='r')
        print(f"Number of captured peaks: {len(x_locs)}")
    return np.asarray(x_locs)


def interpolate(x, map_vals):
    """
    Method to map values of x to their closest value in the map values list 

    Parameters: 
        - x: list or array like iterable that is to be mapped
        - map_vals: list or array like iterable that serves as the mapping for X

    Returns: 
        - new_data: numpy array 
    """
    new_data = []
    # for each number in x
    for num in x:
        min_dist = MAXINT
        val = num
        for i in map_vals:
            dist = abs(num - i)
            if dist < min_dist:
                min_dist = dist
                val = i
        new_data.append(val)

    return np.asarray(new_data)


def load_data(path, dtype=int):
    """
    Method to load data from the text file
    Parameters:
        - path: string of path to the text file
        - dtype: data type for the data 

    Returns: 
        - data: data array
    """
    f = open(path, "r")
    # define list of pri points and read from file
    if dtype == str:
        data = ''
    else:
        data = []
    for line in f:
        if dtype == str:
            data = data + line
        else:
            data.append(dtype(line))

    if dtype != str:
        data = np.asarray(data)
    return data


def preprocess(corrupted, wsz, scaler, x_locs):
    """
    Method to preprocess the data for classification models
    Parameters: 
        - pri_data: numpy array array of pri_data
        - wsz: int, window size
        - scaler: Scaler object, example: StandardScaler, MinMaxScaler

    Returns 
        - x_train, x_test, y_train, y_test: arrays or list like iterables of all the data
    """
    # interpolate the corrupted array to their closest peak
    interp_data = interpolate(corrupted, x_locs)

    # transform the array from high values to indicies of where their value occurs in x_locs
    transformed_1 = transform_list(interp_data, list(x_locs))

    # crete Y data one-hot encoded
    y_list = []
    for i in range(len(transformed_1) - (wsz)):
        y_list.append(transformed_1[i+wsz])
    y_data_cat = to_categorical(y_list, num_classes=len(x_locs))
    y_data = np.asarray(y_data_cat)

    # Fit and transform to the scaler object
    corrupted = corrupted.reshape(-1, 1)
    scaler.fit(corrupted)
    corrupted = scaler.transform(corrupted)

    # create X data
    x_data = []
    # Slide window over data and save the window as X and the next PRI as Y
    for i in range(len(corrupted)-(wsz)):
        x_data.append(corrupted[i:i+wsz])
    x_data = np.asarray(x_data)

    # fix x data structure
    x_data = x_data.reshape(-1, wsz)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def preprocess_proper(corrupted, wsz, scaler, x_locs):
    """
    Method to preprocess the data for classification models
    Parameters: 
        - pri_data: numpy array array of pri_data
        - wsz: int, window size
        - scaler: Scaler object, example: StandardScaler, MinMaxScaler

    Returns 
        - x_train, x_test, y_train, y_test: arrays or list like iterables of all the data
    """
    # interpolate the corrupted array to their closest peak
    interp_data = interpolate(corrupted, x_locs)

    # transform the array from high values to indicies of where their value occurs in x_locs
    transformed_1 = transform_list(interp_data, list(x_locs))

    # crete Y data one-hot encoded
    y_list = []
    for i in range(len(transformed_1) - (wsz)):
        y_list.append(transformed_1[i+wsz])
    y_data_cat = to_categorical(y_list, num_classes=len(x_locs))
    y_data = np.asarray(y_data_cat)

    # create X data
    x_data = []
    # Slide window over data and save the window as X and the next PRI as Y
    for i in range(len(corrupted)-(wsz)):
        x_data.append(corrupted[i:i+wsz])
    x_data = np.asarray(x_data)

    # fix x data structure
    x_data = x_data.reshape(-1, wsz)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=0.3, random_state=42)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test, y_train, y_test


def preprocessing_validation(clean, corrupted, wsz, scaler, x_locs):
    """
    This method is used for preprocessing the data for validation. We need a separate method
    because we want the X data to be the same but need the Y data to be the clean data so that
    we know how far off our classification is. 

    Parameters: 
        - clean: clean data array 
        - corrupted: corrupted data array 
        - wsz: window size (int)
        - scaler: scaler object, typically StandarScaler() from sklearn
        - x_locs: x locations of peaks on from corrupted histogram

    returns
        - x and y train and test arrays. Since this is validation you do not need to retrain
    """
    # first we need to interpolate
    interpolated_data = interpolate(corrupted, x_locs)

    interpolated_data = interpolated_data.reshape(-1, 1)
    # Fit and transform to the scaler object
    scaler.fit(interpolated_data)
    interpolated_data = scaler.transform(interpolated_data)
    # list to hold x_data
    x_data = []
    y_data = []
    # Slide window over data and save the window as X and the next PRI as Y
    for i in range(len(interpolated_data)-(wsz)):
        x_data.append(interpolated_data[i:i+wsz])
        y_data.append(clean[i+wsz])
    x_data = np.asarray(x_data)
    # fix x data structure
    x_data = x_data.reshape(-1, wsz)
    y_data = np.asarray(y_data)

    xtrain, xtest, ytrain, ytest = train_test_split(
        x_data, y_data, random_state=42, train_size=0.3)
    return xtrain, xtest, ytrain, ytest


def preprocessing_validation_proper(clean, corrupted, wsz, scaler, x_locs):
    """
    This method is used for preprocessing the data for validation. We need a separate method
    because we want the X data to be the same but need the Y data to be the clean data so that
    we know how far off our classification is. 

    Parameters: 
        - clean: clean data array 
        - corrupted: corrupted data array 
        - wsz: window size (int)
        - scaler: scaler object, typically StandarScaler() from sklearn
        - x_locs: x locations of peaks on from corrupted histogram

    returns
        - x and y train and test arrays. Since this is validation you do not need to retrain
    """
    # first we need to interpolate
    interpolated_data = interpolate(corrupted, x_locs)

    # list to hold x_data
    x_data = []
    y_data = []
    # Slide window over data and save the window as X and the next PRI as Y
    for i in range(len(interpolated_data)-(wsz)):
        x_data.append(interpolated_data[i:i+wsz])
        y_data.append(clean[i+wsz])
    x_data = np.asarray(x_data)
    # fix x data structure
    x_data = x_data.reshape(-1, wsz)
    y_data = np.asarray(y_data)

    xtrain, xtest, ytrain, ytest = train_test_split(
        x_data, y_data, random_state=42, train_size=0.3)

    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)

    return xtrain, xtest, ytrain, ytest


def save_data(data, path):
    """
    Generic Method to save any data array to a text file. 

    Parameters: 
    - data: list or arary like iterable.
    - path: string, path to text file to create and save data to.
    """
    dataFile = open(path, 'w+')
    for elm in data:
        dataFile.write(str(elm))
        dataFile.write("\n")
    dataFile.close()


def transform_list(arr, unique_vals):
    """
     Method to transform the list from high values to indicies of where those occur in the uniques values array. 
     This method assumes that the input array arr has been interpolated to the unique values.
     Parameters: 
        - arr: array or list like iterable 
        - unique_vals: array or list like iterable that contains all the unique values of arr
    Return: 
        - arr/list
"""
    new_list = []
    for elm in arr:
        new_list.append(np.where(unique_vals == elm))
    new_list = np.asarray(new_list)
    new_list = new_list.reshape(-1)
    return new_list
