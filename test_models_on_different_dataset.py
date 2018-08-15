#!/usr/bin/python

###########################
# Oedipus Utility imports #
###########################
from Oedipus.utils.misc import *
from Oedipus.utils.graphics import *
from Oedipus.utils.data import *
###########################
# Oedipus Service imports #
###########################
from Oedipus.gadgets import classification
from Oedipus.gadgets import clustering
from Oedipus.gadgets import feature_extraction
from Oedipus.gadgets import data_visualization
######################
# OS Utility imports #
######################
#from __future__ import division
import time, sys, os, subprocess
import shutil, glob, argparse, random
import numpy

def write_checkpoint(stage):
    with open('checkpoint3.txt', 'w') as checkpoint_file:
        checkpoint_file.write(str(stage))

def checkpoint(stage):
    last_stage = 0
    result = False
    if os.path.exists('checkpoint3.txt'):
        with open('checkpoint3.txt', 'r') as checkpoint_file:
            last_stage = int(checkpoint_file.read().strip())
    if last_stage > stage:
        result = True
    else:
        write_checkpoint(stage)
    return result

def main():
    train_dir = 'D:\\BGU\\Oedipus\\another_simple_programs'
    test_dir = 'D:\\BGU\\dataset\\home\\vagrant\\random_programs'
    max_features = 1000
    
    if not checkpoint(1):
        flavor = 'objdumps'
        tfidf_flavor = 'tfidfobjs'
        filter = 'both'
        filtered_input_ext = flavor + '_' + filter
        output_ext = tfidf_flavor + '_cross'
        train_files = glob.glob('%s%s*.%s' % (train_dir, os.sep, flavor)) 
        test_files = glob.glob('%s%s*.%s' % (test_dir, os.sep, flavor))
        # vectorizer_file = 'random_programs_vectorizer.pkl'
        vectorizer_file = 'another_simple_vectorizer.pkl'
        if os.path.exists(vectorizer_file):
            if feature_extraction.extractTFIDFWithSavedVectorizer(vectorizer_file, test_files, output_ext):
                prettyPrint("Successfully extracted %s TF-IDF features from traces with \"%s\" extension" % (max_features, filtered_input_ext))
            else:
                prettyPrint("Some error occurred during TF-IDF feature extraction", "error")
                return
        else:            
            if feature_extraction.extractTFIDFWithVectorizer(train_files, test_files, max_features, output_ext, save_train=True, vectorizer_file=vectorizer_file):
                prettyPrint("Successfully extracted %s TF-IDF features from traces with \"%s\" extension" % (max_features, filtered_input_ext))
            else:
                prettyPrint("Some error occurred during TF-IDF feature extraction", "error")
                return
    
    if not checkpoint(2):
        # Load data from source directory
        X_train, y_train, allClasses, originalPrograms = loadFeaturesFromDir(train_dir, 'tfidfobjs_cross', 'label')
        X_test, y_test, allClasses, originalPrograms = loadFeaturesFromDir(test_dir, 'tfidfobjs_cross', 'label')
        accuracies, timings, allDepths = [], [], [4,6,8,10,12]
        for maxDepth in allDepths:
            accuracyRates, allTimings, allProbabilities, predictedLabels = classification.classifyTree(X_train, y_train, X_test, y_test, 'gini', int(maxDepth), visualizeTree=False)
            prettyPrint("Classification accuracy: %.2f" % (accuracyRates*100.0), "output")
            accuracies.append(accuracyRates)
            timings.append(allTimings)

        # Plot accuracies graph
        prettyPrint("Plotting accuracies")
        data_visualization.plotAccuracyGraph(allDepths, accuracies, "Maximum Tree Depth", "Classification Accuracy", "Classification Accuracy: gini (tfidfs_cross)", "accuracy_cross_datasets.pdf")
        print(timings)

if __name__ == "__main__":
    main()


