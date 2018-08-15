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
        print('Compiling "%s" failed. Skipping file'.format(targetFile))
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
            print 'file no. %d requested - out of %d files' % (count, size)
            yield open(doc).read()

def extractTFIDFWithVectorizer(train_files, test_files, max_features=128, out_extension="tfidf_vec", save_train=True, vectorizer_file=None):
    try:
        prettyPrint("Extracting TF-IDF features")
        train_iter = DocumentsIterator(train_files)
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=max_features, token_pattern='\S+', norm='l2', smooth_idf=True, use_idf=True, sublinear_tf=False)
        X = vectorizer.fit_transform(train_iter)
        if save_train:
            for row, train in enumerate(train_files):
                out_filename = os.path.splitext(train)[0] + '.%s' % out_extension
                with open(out_filename, 'w') as out_file:
                    out_file.write(numpy.array2string(X[row, :], separator=','))
        if vectorizer_file:
            joblib.dump(vectorizer, vectorizer_file)

        test_iter = DocumentsIterator(test_files)
        X = vectorizer.transform(test_iter)
        for row, test in enumerate(test_files):
            out_filename = os.path.splitext(test)[0] + '.%s' % out_extension
            with open(out_filename, 'w') as out_file:
                out_file.write(numpy.array2string(X[row, :], separator=','))

    except Exception as e:
        prettyPrint("Error encountered in \"extractTFIDFWithVectorizer\": %s" % traceback.format_exc(), "error")
        return False

    return True

def extractTFIDFWithSavedVectorizer(vectorizer_file, test_files, out_extension="tfidf_vec"):
    try:
        prettyPrint("Loading vectorizer")
        vectorizer = joblib.load(vectorizer_file)

        test_iter = DocumentsIterator(test_files)
        X = vectorizer.transform(test_iter)
        for row, test in enumerate(test_files):
            out_filename = os.path.splitext(test)[0] + '.%s' % out_extension
            with open(out_filename, 'w') as out_file:
                out_file.write(numpy.array2string(X[row, :], separator=','))

    except Exception as e:
        prettyPrint("Error encountered in \"extractTFIDFWithVectorizer\": %s" % traceback.format_exc(), "error")
        return False

    return True