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
from Oedipus.gadgets import program_generation
from Oedipus.gadgets import random_programs
######################
# OS Utility imports #
######################
#from __future__ import division
import time, sys, os, subprocess
import shutil, glob, argparse, random
import numpy

def main():
    source_dir = '/home/vagrant/random_programs'
    number_of_programs = 2000
    tigress_dir = '/oedipus/tigress-2.2'
    obfuscation_level = 1
    obfuscation_function = 'SECRET'
    max_features = 1000
    kfold = 10

    if random_programs.generate_random_programs(source_dir, number_of_programs, obfuscation_function):
        prettyPrint("Successfully generated %d random programs" % number_of_programs)
    else:
        prettyPrint("Some error occurred during random program generation", "warning")
    
    # Get programs from source directory [random/pre-existent]
    sourceFiles = sorted(glob.glob("%s/*.c" % source_dir))
    if len(sourceFiles) < 1:
        prettyPrint("No files were found in \"%s\". Exiting" % source_dir, "error")
        return

    generationStatus = program_generation.generateObfuscatedPrograms(sourceFiles, tigress_dir, obfuscation_level, obfuscation_function)
    prettyPrint("Successfully generated obfuscated programs")
        
    if not os.path.exists(source_dir):
        prettyPrint("Unable to locate \"%s\". Exiting" % source_dir, "error")
        return
    sourceFiles = sorted(glob.glob("%s/*.c" % source_dir))
    if len(sourceFiles) < 1:
        prettyPrint("No files were found in \"%s\". Exiting" % source_dir)
    
    for targetFile in sourceFiles:            
        if not os.path.exists(targetFile.replace(".c", ".label")):
            prettyPrint("File \"%s\" does not have a label/metadata file. Removing" % targetFile, "warning")
            sourceFiles.pop( sourceFiles.index(targetFile) )

    prettyPrint("Generating static and dynamic traces")
    if not feature_extraction.extractTFIDF(source_dir, sourceFiles):
        prettyPrint("Could not generate traces from source files. Exiting", "error")
        return

    prettyPrint("Successfully generated traces")
    cleanUp()

    flavors = ['dyndis', 'dyndiss', 'objdump', 'objdumps']
    tfidf_flavors = ['tfidf', 'tfidfs', 'tfidfobj', 'tfidfobjs']
    for i,flavor in enumerate(flavors):
        filter_modes = ['raw', 'both']
        for filter in filter_modes:
            filtered_input_ext = flavor + '_' + filter
            output_ext = tfidf_flavors[i] + '_both' if filter == 'both' else ''
            if filterTraces(source_dir, flavor, filter, filtered_input_ext, obfuscation_function):
                prettyPrint("Successfully filtered \"%s\" traces to \"%s\" traces using the \"%s\" filter" % (filter, filtered_input_ext, filter))
            else:
                prettyPrint("Some error occurred during filteration", "warning")
            if feature_extraction.extractTFIDFMemoryFriendly(source_dir, filtered_input_ext, max_features, output_ext):
                prettyPrint("Successfully extracted %s TF-IDF features from traces with \"%s\" extension" % (max_features, filtered_input_ext))
            else:
                prettyPrint("Some error occurred during TF-IDF feature extraction", "warning")
    
    data_flavors = ['tfidf', 'tfidfs', 'tfidfobj', 'tfidfobjs', 'tfidf_both', 'tfidfs_both', 'tfidfobj_both', 'tfidfobjs_both']
    for flavor in data_flavors:
        algorithms = ["bayes", "tree"]
        for algo in algorithms:
            if algo == 'bayes':
                X, y, allClasses = loadFeaturesFromDir(source_dir, flavor, 'label')
                for reduction_method in ['selectkbest', 'pca', 'none']:
                    classificationLog = open("classificationlog_%s_exp1_%s_%s.txt" % (flavor, reduction_method, algo), "a") # A file to log all classification labels
                    classificationLog.write("Experiment 1 - Algorithm: %s, Datatype: %s\n" % (algo, flavor))
                    if reduction_method == "selectkbest":
                        accuracies, timings = [], []
                        targetDimensions = [8, 16, 32, 64, 128]#[64, 128, 256, 512, 1000]
                        for dimension in targetDimensions:
                            accuracyRates, allProbabilities, allTimings, groundTruthLabels, predictedLabels = classification.classifyNaiveBayesKFold(X, y, kFold=kfold, reduceDim=reduction_method, targetDim=dimension)
                            prettyPrint("Average classification accuracy: %s%%" % (averageList(accuracyRates)*100.0), "output")
                            accuracies.append(averageList(accuracyRates))
                            timings.append(averageList(allTimings))
                            # Log classifications
                            for foldIndex in range(len(predictedLabels)):
                                classificationLog.write("Target Dimensionality: %s\n" % dimension)
                                for labelIndex in range(len(predictedLabels[foldIndex])):
                                    classificationLog.write("Class:%s,Predicted:%s\n" % (allClasses[groundTruthLabels[foldIndex][labelIndex]], allClasses[predictedLabels[foldIndex][labelIndex]]))
                        
                        classificationLog.close()
                        # Plot accuracies graph
                        prettyPrint("Plotting accuracies")
                        data_visualization.plotAccuracyGraph(targetDimensions, accuracies, "Number of Selected Features", "Classification Accuracy", "Classification Accuracy: Selected Features (%s)" % flavor, "accuracy_%s_exp1_%s_selectkbest.pdf" % (flavor, algo)) 
                        # Plot performance graph
                        print(timings)
                    elif reduction_method == "pca":
                        accuracies, timings = [], []
                        targetDimensions = [8, 16, 32, 64, 128]#[2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
                        for dimension in targetDimensions:
                            accuracyRates, allProbabilities, allTimings, groundTruthLabels, predictedLabels = classification.classifyNaiveBayesKFold(X, y, kFold=kfold, reduceDim=reduction_method, targetDim=dimension)
                            prettyPrint("Average classification accuracy: %s%%" % (averageList(accuracyRates)*100.0), "output")
                            accuracies.append(averageList(accuracyRates))
                            timings.append(averageList(allTimings))
                            # Log classifications
                            for foldIndex in range(len(predictedLabels)):
                                classificationLog.write("Target Dimensionality: %s\n" % dimension)
                                for labelIndex in range(len(predictedLabels[foldIndex])):
                                    classificationLog.write("Class:%s,Predicted:%s\n" % (allClasses[groundTruthLabels[foldIndex][labelIndex]], allClasses[predictedLabels[foldIndex][labelIndex]]))

                        classificationLog.close()
                        # Plot accuracies graph
                        prettyPrint("Plotting accuracies")
                        data_visualization.plotAccuracyGraph(targetDimensions, accuracies, "Number of Extracted Features", "Classification Accuracy", "Classification Accuracy: PCA (%s)" % flavor, "accuracy_%s_exp1_%s_pca.pdf" % (flavor, algo))
                        # Plot performance graph
                        print(timings)
                    else:    
                        accuracyRates, allProbabilities, allTimings, predictedLabels = classification.classifyNaiveBayes(X, y, kFold=kfold)
                        prettyPrint("Average classification accuracy: %s%%, achieved in an average of %s seconds" % (averageList(accuracyRates)*100.0, averageList(allTimings)), "output")
            ####################
            # Using CART trees #
            ####################
            elif algo == "tree":
               # Load data from source directory
               X, y, allClasses = loadFeaturesFromDir(source_dir, flavor, 'label')
               for splitting_criterion in ['gini', 'entropy']:
                    classificationLog = open("classificationlog_%s_exp1_%s_%s.txt" % (flavor, splitting_criterion, algo), "a") # A file to log all classification labels
                    classificationLog.write("Experiment 1 - Algorithm: %s, Datatype: %s\n" % (algo, flavor))
                    accuracies, timings, allDepths = [], [], [2,3,4,5,6,7,8,10,12,14,16]#,32,64]
                    for maxDepth in allDepths:
                        accuracyRates, allProbabilities, allTimings, groundTruthLabels, predictedLabels = classification.classifyTreeKFold(X, y, kfold, splitting_criterion, int(maxDepth), visualizeTree=False)
                        #print accuracyRates, allProbabilities
                        prettyPrint("Average classification accuracy: %s%%" % (averageList(accuracyRates)*100.0), "output")
                        accuracies.append(averageList(accuracyRates))
                        timings.append(averageList(allTimings))
                        # Log classifications
                        for foldIndex in range(len(predictedLabels)):
                            classificationLog.write("Tree Depth: %s\n" % maxDepth)
                            for labelIndex in range(len(predictedLabels[foldIndex])):
                                classificationLog.write("Class:%s,Predicted:%s\n" % (allClasses[groundTruthLabels[foldIndex][labelIndex]], allClasses[predictedLabels[foldIndex][labelIndex]]))

                    classificationLog.close()
                    # Plot accuracies graph
                    prettyPrint("Plotting accuracies for \"%s\" criterion" % splitting_criterion)
                    data_visualization.plotAccuracyGraph(allDepths, accuracies, "Maximum Tree Depth", "Classification Accuracy", "Classification Accuracy: %s (%s)" % (splitting_criterion, flavor), "accuracy_%s_exp1_%s_%s.pdf" % (flavor, splitting_criterion, algo))
                    print(timings)

if __name__ == "__main__":
    main()

