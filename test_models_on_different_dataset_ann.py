#!/usr/bin/python

######################
# OS Utility imports #
######################
import time, sys, os, subprocess
import shutil, glob, argparse, random
import numpy as np
import pandas as pd
from hashlib import md5
######################
# Keras imports      #
######################
seed = 42
np.random.seed(seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
        
def load_dataset(dataset_path):
    dataset_path_md5 = md5(dataset_path.encode('utf-8')).hexdigest()
    features_csv = '{0}_features.csv'.format(dataset_path_md5[:6])
    labels_csv = '{0}_labels.csv'.format(dataset_path_md5[:6])
    if os.path.exists(features_csv) and os.path.exists(labels_csv):
        X = pd.read_csv(features_csv, header=None)
        Y = pd.read_csv(labels_csv, header=None)
        return X, Y
        
    all_tfidf_files = glob.glob('{0}\\*.tfidfobjs_cross'.format(dataset_path))
    X = None
    Y = None
    class_labels = []
    for i,tfidf_file in enumerate(all_tfidf_files):
        if i % 100 == 0:
            print('loaded {0} files.'.format(i))
        label_file = os.path.splitext(tfidf_file)[0] + '.label'
        if not os.path.exists(label_file):
            print("Could not find a label file for \"{0}\". Skipping".format(tfidf_file))
            continue

        data = pd.DataFrame(data=[float(s) for s in open(tfidf_file).read()[2:-2].split(',')]).T
        if X is not None:
            X = pd.concat([X, data])
        else:
            X = data
        label = open(label_file).read()
        if label not in class_labels:
            class_labels.append(label)
        label_index = class_labels.index(label)
        df = pd.DataFrame(data={'label': [label_index]})
        if Y is not None:
            Y = pd.concat([Y, df])
        else:
            Y = df
    
    X.to_csv(features_csv, header=False, index=False)
    Y.to_csv(labels_csv, header=False, index=False)
    return X, Y

def main():
    train_dir = 'D:\\BGU\\dataset\\home\\vagrant\\random_programs'
    test_dir = 'D:\\BGU\\Oedipus\\simple_programs'
    test_dir_2 = 'D:\\BGU\\Oedipus\\another_simple_programs'
    
    # Load data from source directory
    X_train, Y_train = load_dataset(test_dir_2)
    print('training data: {0}, {1}'.format(X_train.shape, Y_train.shape))
    Y_train = np_utils.to_categorical(Y_train, num_classes=7)
    X_test, Y_test = load_dataset(test_dir)
    Y_test = np_utils.to_categorical(Y_test, num_classes=7)
    print('test data: {0}, {1}'.format(X_test.shape, Y_test.shape))
                    
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=1000))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(x=X_train, y=Y_train, epochs=10, verbose=1, validation_data=(X_test, Y_test))
    print(model.evaluate(x=X_test, y=Y_test))
    X_test, Y_test = load_dataset(train_dir)
    Y_test = np_utils.to_categorical(Y_test, num_classes=7)
    print('validation data: {0}, {1}'.format(X_test.shape, Y_test.shape))
    print(model.evaluate(x=X_test, y=Y_test))

if __name__ == "__main__":
    main()


