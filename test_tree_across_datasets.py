from Oedipus.gadgets import classification
from Oedipus.gadgets import data_visualization
import os
import numpy as np
import pandas as pd

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
            X_train, y_train = load_dataset(dataset1, dataset1)
            X_test, y_test = load_dataset(dataset1, dataset2)
            accuracies, timings, allDepths = [], [], [4,6,8,10,12]
            for maxDepth in allDepths:
                accuracyRates, allTimings, allProbabilities, predictedLabels = classification.classifyTree(X_train, y_train, X_test, y_test, 'gini', int(maxDepth), visualizeTree=False)
                print("Classification accuracy: %.2f" % (accuracyRates*100.0))
                accuracies.append(accuracyRates)
                timings.append(allTimings)

            # Plot accuracies graph
            print("Plotting accuracies")
            data_visualization.plotAccuracyGraph(allDepths, accuracies, "Maximum Tree Depth", "Classification Accuracy", "Classification Accuracy: gini (tfidfs_cross)", "accuracy_{0}_{1}.pdf".format(dataset1, dataset2))
            print(timings)

if __name__ == "__main__":
    main()


