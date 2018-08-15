from Oedipus.gadgets.feature_extraction import extractTFIDFWithVectorizer
from Oedipus.utils.data import loadLabelFromFile
import sys, os, glob
from sklearn.externals import joblib
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) != 5:
        print('Usage: python generate_dataset_csv.py <traces_dir> <vectorizer_file> <features_csv_filename> <labels_csv_filename>')
        return
    
    traces_dir = sys.argv[1]
    if not os.path.exists(traces_dir):
        print('Error: traces_dir {0} does not exists.'.format(traces_dir))
        return

    vectorizer_file = sys.argv[2]
    if not os.path.exists(vectorizer_file):
        print('Error: vectorizer_file {0} does not exists.'.format(vectorizer_file))
        return

    features_csv = sys.argv[3]
    labels_csv = sys.argv[4]

    trace_files = sorted(glob.glob("{0}/*.objdumps_both".format(traces_dir)))
    if len(trace_files) < 1:
        print('Error: No *.objdumps_both trace files were found in {0}'.format(traces_dir))
        return

    label_files = []
    valid_trace_files = []
    for trace in trace_files:
        label_file = trace.replace(".objdumps_both", ".label")
        if os.path.exists(label_file):
            label_files.append(label_file)
            valid_trace_files.append(trace)

    vectorizer = joblib.load(vectorizer_file)
    X = extractTFIDFWithVectorizer(vectorizer, valid_trace_files)
    X = pd.DataFrame(data=X.toarray())
    Y_list = []
    all_classes = []
    for label_file in label_files:
        current_class, current_params = loadLabelFromFile(label_file)
        if current_class not in all_classes:
            all_classes.append(current_class)
        class_index = all_classes.index(current_class)
        Y_list.append(class_index)
    Y = pd.DataFrame(Y_list)
    X.to_csv(features_csv, header=False, index=False)
    Y.to_csv(labels_csv, header=False, index=False)
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    top_n = 10
    top_features = [features[i] for i in indices[:top_n]]
    print ('top features: {0}'.format(top_features))

if __name__ == "__main__":
    main()
