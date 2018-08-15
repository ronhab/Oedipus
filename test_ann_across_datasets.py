import os
import numpy as np
import pandas as pd
seed = 42
np.random.seed(seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils        

def load_dataset(dataset1, dataset2):
    dataset_prefix = '{0}_{1}'.format(dataset1, dataset2)
    features_csv = '{0}_tfidf.csv'.format(dataset_prefix)
    labels_csv = '{0}_labels.csv'.format(dataset_prefix)
    if os.path.exists(features_csv) and os.path.exists(labels_csv):
        X = pd.read_csv(features_csv, header=None)
        Y = pd.read_csv(labels_csv, header=None)
        return X, Y

def main():
    
    datasets_names = [
        'random',
        'simple1',
        'simple2'
    ]
    
    for dataset1 in datasets_names:
        for dataset2 in datasets_names:
            print('Classifing: {0} - {1}'.format(dataset1, dataset2))
            X_train, Y_train = load_dataset(dataset1, dataset1)
            X_test, Y_test = load_dataset(dataset1, dataset2)
            Y_train = np_utils.to_categorical(Y_train, num_classes=7)
            Y_test = np_utils.to_categorical(Y_test, num_classes=7)
            print('train data: {0}, {1}'.format(X_train.shape, Y_train.shape))
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

if __name__ == "__main__":
    main()


