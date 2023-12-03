from __future__ import print_function

from builtins import range
# from six.moves import cPickle as pickle
import pickle
import numpy as np
import os
import platform
from imageio import imread
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding = 'bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(cifar10_dir='data/cifar-10-batches-py', num_training=49000, num_validation=1000, num_test=1000,
                     normalize=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image and divide by the std
    if normalize:
        mean_image = np.mean(X_train, axis=0)
        std_image = np.std(X_train, axis=0)
        X_train = (X_train - mean_image) / std_image
        X_val = (X_val - mean_image) / std_image
        X_test = (X_test - mean_image) / std_image


    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.

    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (3, 64, 64) array giving mean training image
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d'
                  % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * \
                        np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
        ## grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]]
                  for img_file in img_files]
        y_test = np.array(y_test)

    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]

    return {
      'class_names': class_names,
      'X_train': X_train,
      'y_train': y_train,
      'X_val': X_val,
      'y_val': y_val,
      'X_test': X_test,
      'y_test': y_test,
      'class_names': class_names,
      'mean_image': mean_image,
    }


def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt)
    will be skipped.

    Inputs:
    - models_dir: String giving the path to a directory containing model files.
      Each model file is a pickled dictionary with a 'model' field.

    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = load_pickle(f)['model']
            except pickle.UnpicklingError:
                continue
    return models


def load_imagenet_val(num=None):
    """Load a handful of validation images from ImageNet.

    Inputs:
    - num: Number of images to load (max of 25)

    Returns:
    - X: numpy array with shape [num, 224, 224, 3]
    - y: numpy array of integer image labels, shape [num]
    - class_names: dict mapping integer label to class name
    """
    imagenet_fn = 'cs682/datasets/imagenet_val_25.npz'
    if not os.path.isfile(imagenet_fn):
      print('file %s not found' % imagenet_fn)
      print('Run the following:')
      print('cd cs682/datasets')
      print('bash get_imagenet_val.sh')
      assert False, 'Need to download imagenet_val_25.npz'
    f = np.load(imagenet_fn)
    X = f['X']
    y = f['y']
    class_names = f['label_map'].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names

# First we need to add a function to read in our ECG data (from BIH dataset)
def load_ecg_data(data_folder):
    train_path = os.path.join(data_folder, 'mitbih_train.csv')
    test_path = os.path.join(data_folder, 'mitbih_test.csv')

    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

    # Splitting features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return X_train, y_train, X_test, y_test

# For testing its always a good idea to create a mini dataset
def create_mini_ecg(X_train, y_train, num_samples_per_class=100):
    unique_classes = np.unique(y_train)
    mini_X, mini_y = [], []

    for class_label in unique_classes:
        # Extracting samples for each class
        class_indices = np.where(y_train == class_label)[0]
        sampled_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
        mini_X.extend(X_train[sampled_indices])
        mini_y.extend(y_train[sampled_indices])

    return np.array(mini_X), np.array(mini_y)

# Functions to preprocess ECG data
# first is a set of bandwith filters
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def preprocess_ecg(data, fs=125, lowpass_cutoff=40, highpass_cutoff=0.5):
    # Apply low-pass filter
    b, a = butter_lowpass(lowpass_cutoff, fs)
    data = filtfilt(b, a, data, axis=1)

    # Apply high-pass filter
    b, a = butter_highpass(highpass_cutoff, fs)
    data = filtfilt(b, a, data, axis=1)

    # Normalization
    data = 2 * ((data - np.min(data)) / (np.max(data) - np.min(data))) - 1

    return data

def preprocess_and_plot_ecg(data, sample_index=0, fs=125, lowpass_cutoff=40, highpass_cutoff=0.5):
    # Original Data
    original_data = data[sample_index]

    # Apply low-pass filter
    b, a = butter_lowpass(lowpass_cutoff, fs)
    low_passed_data = filtfilt(b, a, original_data)

    # Apply high-pass filter
    b, a = butter_highpass(highpass_cutoff, fs)
    filtered_data = filtfilt(b, a, low_passed_data)

    # Normalization
    normalized_data = 2 * ((filtered_data - np.min(filtered_data)) / (np.max(filtered_data) - np.min(filtered_data))) - 1

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(original_data)
    plt.title('Original Data')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(1, 3, 2)
    plt.plot(filtered_data)
    plt.title('Filtered Data')
    plt.xlabel('Sample')

    plt.subplot(1, 3, 3)
    plt.plot(normalized_data)
    plt.title('Normalized Data')
    plt.xlabel('Sample')

    plt.tight_layout()
    plt.show()

    return normalized_data

# Function to read in EEG data (UCI EEG database)
def process_eeg_data(file_path):
    # Initialize lists to store data and labels
    data = []
    labels = []

    # Iterate over each file in the directory
    for filename in os.listdir(file_path):
        with open(os.path.join(file_path, filename), 'r') as file:
            lines = file.readlines()

            # Parse the header to determine the label
            subject_type = lines[0].split()[0][3]
            label = 1 if subject_type == 'a' else 0  # 1 for alcoholic, 0 for control

            # Extract sensor data
            sensor_data = []
            for line in lines[5:]:  # Data starts from line 5
                _, _, _, value = line.split()
                sensor_data.append(float(value))
            
            # Add to data lists
            data.append(sensor_data)
            labels.append(label)

    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Preprocess EEG
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=1)
    return y

def preprocess_eeg_data(eeg_data, lowcut, highcut, fs, baseline_idx):
    scaler = StandardScaler()

    for key in ['X_train', 'X_val', 'X_test']:
        # Apply bandpass filter
        eeg_data[key] = bandpass_filter(eeg_data[key], lowcut, highcut, fs, order=5)

        # Baseline correction
        baseline = np.mean(eeg_data[key][:, :, baseline_idx], axis=2, keepdims=True)
        eeg_data[key] -= baseline

        # Normalization - reshape for scaler and then reshape back to original dimensions
        original_shape = eeg_data[key].shape
        eeg_data[key] = scaler.fit_transform(eeg_data[key].reshape(-1, original_shape[-1])).reshape(original_shape)

    return eeg_data

import matplotlib.pyplot as plt

def plot_sample_comparison(original_data, preprocessed_data, trial_index, channel_index):
    #Plots a single sample of EEG data before and after preprocessing.

    # Extract the specific trial and channel data
    original_sample = original_data[trial_index, :, channel_index]
    preprocessed_sample = preprocessed_data[trial_index, :, channel_index]

    # Create a plot with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f'EEG Data Comparison for Trial {trial_index} and Channel {channel_index}')

    # Plot original data
    axs[0].plot(original_sample)
    axs[0].set_title('Original Data')
    axs[0].set_xlabel('Sample Number')
    axs[0].set_ylabel('Amplitude (uV)')

    # Plot preprocessed data
    axs[1].plot(preprocessed_sample)
    axs[1].set_title('Preprocessed Data')
    axs[1].set_xlabel('Sample Number')
    axs[1].set_ylabel('Amplitude (uV)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
