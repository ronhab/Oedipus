#!/usr/bin/python

###################
# Library imports #
###################
from Oedipus.utils.data import *
from Oedipus.utils.misc import *
import subprocess, time, os, traceback
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.externals import joblib

####################
# Defining Methods #
####################
def generateObjdumpDisassembly(outFile, inExt=".out", outExt=".objdump"):
    """ Generates an Objdump of an executable """
    # Check whether file is executable using "file"
    fileOutput = subprocess.Popen(["file", outFile], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    if fileOutput.lower().find("executable") == -1:
        prettyPrint("The file \"%s\" is not an executable" % outFile, "warning")
        return False
    # Generate the objdump disassembly 
    objdumpFile = open(outFile.replace(inExt, outExt), "w")
    objdumpArgs = ["objdump", "--disassemble", outFile]
    objdumpOutput = subprocess.Popen(objdumpArgs, stderr=subprocess.STDOUT, stdout=objdumpFile).communicate()[0]
    # Check if the file has been generated and not empty
    if not os.path.exists(outFile.replace(inExt, outExt)) or os.path.getsize(outFile.replace(inExt, outExt)) < 1:
        prettyPrint("Could not find a (non-empty) objdump disassembly file for \"%s\"" % outFile, "warning")
        return False

    return True

def compileFile(targetFile):
    """ Compiles a source files for feature extraction """
    outFile = targetFile.replace(".c",".outs")
    gccArgs = ["gcc", "-Wl,--unresolved-symbols=ignore-in-object-files","-std=c99", targetFile, "-o", outFile]
    stripArgs = ["strip", "-s", "-K", "main", "-K", "SECRET", outFile]
    print('Compiling "{0}"'.format(targetFile))
    subprocess.Popen(gccArgs, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    print('Stripping "{0}"'.format(targetFile))
    subprocess.Popen(stripArgs, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    # Check if compilation succeeded by checking for existence of "a.out"
    if not os.path.exists(outFile):
        print('Compiling {0} failed. Skipping file'.format(targetFile))
        return ""
        
    return outFile

def generateTraces(sourceDir, sourceFiles):
    try:
        for targetFile in sourceFiles:
            outFile = compileFile(os.path.join(sourceDir, targetFile))
            if outFile == "":
                print('Unable to compile "{0}". Skipping'.format(targetFile))
                continue
            if generateObjdumpDisassembly(outFile, ".outs", ".objdumps"):
                print("{0} has been successfully generated".format(outFile))
    except Exception as e:
        print('Error encountered in "generateTraces": {0}'.format(traceback.format_exc()))
        return False

    return True

class DocumentsIterator(object):
    def __init__(self, docs):
        self.documents = docs

    def  __iter__(self):
        count = 0
        size = len(self.documents)
        for doc in self.documents:
            count += 1
            print('file no. {0} requested - out of {1} files'.format(count, size))
            yield open(doc).read()

def createTFIDFVectorizer(traces, max_features):
    print('Creating TFIDF Vectorizer')
    train_iter = DocumentsIterator(traces)
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=max_features, token_pattern=r'\S+', norm='l2', smooth_idf=True, use_idf=True, sublinear_tf=False)
    vectorizer.fit(train_iter)
    return vectorizer

def extractTFIDFWithVectorizer(vectorizer, traces):
    prettyPrint("Extracting TF-IDF features")
    train_iter = DocumentsIterator(traces)
    X = vectorizer.transform(train_iter)
    return X
