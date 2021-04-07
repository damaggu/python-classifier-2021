#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

twelve_lead_model_filename = '12_lead_model.sav'
six_lead_model_filename = '6_lead_model.sav'
three_lead_model_filename = '3_lead_model.sav'
two_lead_model_filename = '2_lead_model.sav'

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x))  # Sort classes numerically if numbers.
    else:
        classes = sorted(classes)  # Sort classes alphanumerically otherwise.
    num_classes = len(classes)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    # data, labels = gen_data(num_recordings=num_recordings,
    #                         num_classes=num_classes,
    #                         classes=classes,
    #                         header_files=header_files,
    #                         recording_files=recording_files)
    data, labels = gen_my_data(num_recordings=num_recordings,
                            num_classes=num_classes,
                            classes=classes,
                            header_files=header_files,
                            recording_files=recording_files)

    # Train models.

    # Define parameters for random forest classifier.
    n_estimators = 200  # Number of trees in the forest.
    max_leaf_nodes = 3  # Maximum number of leaf nodes in each tree.
    random_state = 0  # Random state; set for reproducibility.

    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    leads = twelve_leads
    filename = os.path.join(model_directory, twelve_lead_model_filename)

    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices, :]
    features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))

    # imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
    #                                     random_state=random_state).fit(features, labels)
    print('yo')
    classifier = MLPClassifier(hidden_layer_sizes=(100,50,20),random_state=random_state, verbose=10).fit(features, labels)
    # save_model(filename, classes, leads, imputer, classifier)
    save_model(filename, classes, leads, scaler, classifier)

    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    leads = six_leads
    filename = os.path.join(model_directory, six_lead_model_filename)

    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices,:]
    features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))

    # imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
    #                                     random_state=random_state).fit(features, labels)

    classifier = MLPClassifier(hidden_layer_sizes=(100,50,20),random_state=random_state, verbose=1).fit(features, labels)
    # save_model(filename, classes, leads, imputer, classifier)
    save_model(filename, classes, leads, scaler, classifier)
    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    leads = three_leads
    filename = os.path.join(model_directory, three_lead_model_filename)

    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices,:]
    features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))

    # imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
    #                                     random_state=random_state).fit(features, labels)

    classifier = MLPClassifier(hidden_layer_sizes=(100,50,20),random_state=random_state, verbose=1).fit(features, labels)
    # save_model(filename, classes, leads, imputer, classifier)
    save_model(filename, classes, leads, scaler, classifier)

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    leads = two_leads
    filename = os.path.join(model_directory, two_lead_model_filename)

    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices,:]
    features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))

    # imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
    #                                     random_state=random_state).fit(features, labels)

    classifier = MLPClassifier(hidden_layer_sizes=(100,50,20),random_state=random_state, verbose=1).fit(features, labels)
    # save_model(filename, classes, leads, imputer, classifier)
    save_model(filename, classes, leads, scaler, classifier)

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
    joblib.dump(d, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    return load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    # return run_model(model, header, recording)
    return run_my_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    # return run_model(model, header, recording)
    return run_my_model(model, header, recording)
# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    # return run_model(model, header, recording)
    return run_my_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    # return run_model(model, header, recording)
    return run_my_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    num_leads = len(leads)
    data = np.zeros(num_leads + 2, dtype=np.float32)
    age, sex, rms = get_features(header, recording, leads)
    data[0:num_leads] = rms
    data[num_leads] = age
    data[num_leads + 1] = sex

    # Impute missing data.
    features = data.reshape(1, -1)
    features = imputer.transform(features)

    # Predict labels and probabilities.
    labels = classifier.predict(features)
    labels = np.asarray(labels, dtype=np.int)[0]

    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities

def run_my_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    num_leads = len(leads)
    data = np.zeros(num_leads + 2, dtype=np.float32)
    # age, sex, rms = get_features(header, recording, leads)
    # data[0:num_leads] = rms
    # data[num_leads] = age
    # data[num_leads + 1] = sex

    recording, samples, frequency = get_ts_features(header, recording, twelve_leads)
    time = calcLength(samples, frequency)

    # in seconds
    segment_length = 5

    segments = int(time / segment_length)
    for segment in range(segments):
        recording_segment = recording[:, int(segment * segment_length * frequency):
                                         int((segment + 1) * segment_length * frequency)]
        new_segments = []
        for lead in range(12):
            lead_segment = recording_segment[lead, :]
            new_segment = resample_segment(lead_segment, len(lead_segment), frequency, 200)
            new_segments.append(new_segment)
        recording_segment = np.stack(new_segments)

    data = recording_segment

    # Impute missing data.
    features = data.reshape(1, -1)
    features = imputer.transform(features)

    # Predict labels and probabilities.
    labels = classifier.predict(features)
    labels = np.asarray(labels, dtype=np.int)[0]

    probabilities = classifier.predict_proba(features)
    # probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]
    probabilities = np.asarray(probabilities, dtype=np.float32)[0, :]

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x ** 2) / np.size(x))

    return age, sex, rms

def get_ts_features(header, recording, leads):
    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x ** 2) / np.size(x))

    return recording, len(recording[0, :]), get_frequency(header)


def gen_data(num_recordings, num_classes, classes, header_files, recording_files):
    data = np.zeros((num_recordings, 14),
                    dtype=np.float32)  # 14 features: one feature for each lead, one feature for age, and one feature for sex
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)  # One-hot encoding of classes

    for i in range(num_recordings):
        print('    {}/{}...'.format(i + 1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])

        # Get age, sex and root mean square of the leads.
        age, sex, rms = get_features(header, recording, twelve_leads)
        data[i, 0:12] = rms
        data[i, 12] = age
        data[i, 13] = sex

        current_labels = get_labels(header)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1

    return data, labels

def calcLength(samples, frequency):
    return samples / frequency

def resample_segment(recording_segment, samples, old_frequency, new_frequency):
    ratio = new_frequency / old_frequency
    new_samples = int(samples * ratio)
    recording_segment = signal.resample(recording_segment, new_samples)
    return recording_segment

def gen_my_data(num_recordings, num_classes, classes, header_files, recording_files):
    data = []
    labels = []
    for i in range(num_recordings):
        recording_data = []
        recording_label = []
        print('    {}/{}...'.format(i + 1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])

        # Get age, sex and root mean square of the leads.
        recording, samples, frequency = get_ts_features(header, recording, twelve_leads)
        # data[i, 0:12] = rms
        # data[i, 12] = age
        # data[i, 13] = sex

        # TODO: check if makes sense
        # if longer than 5 seconds, add each segment individually with same label

        time = calcLength(samples, frequency)

        # in seconds
        segment_length = 5

        segments = int(time / segment_length)
        for segment in range(segments):
            recording_segment = recording[:, int(segment * segment_length * frequency):
                                             int((segment + 1) * segment_length * frequency)]
            new_segments = []
            for lead in range(12):
                lead_segment = recording_segment[lead, :]
                new_segment = resample_segment(lead_segment, len(lead_segment), frequency, 200)
                new_segments.append(new_segment)
            recording_segment = np.stack(new_segments)
            recording_data.append(recording_segment)

            current_labels = get_labels(header)
            this_label = np.zeros((num_classes,), dtype=np.bool)
            for label in current_labels:
                if label in classes:
                    j = classes.index(label)
                    this_label[j] = 1
            recording_label.append(this_label)

        recording_data = np.stack(recording_data)
        data.append(recording_data)
        labels.append(np.stack(recording_label))

    data = np.vstack(data)
    labels = np.vstack(labels)
    print('yo')

    return data, labels