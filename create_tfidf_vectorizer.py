from Oedipus.gadgets.feature_extraction import createTFIDFVectorizer
import sys, os, glob
from sklearn.externals import joblib

def main():
    if len(sys.argv) != 4:
        print('Usage: python create_tfidf_vectorizer.py <traces_dir> <max_features> <vectorizer_file>')
        return
    
    traces_dir = sys.argv[1]
    if not os.path.exists(traces_dir):
        print('Error: traces_dir {0} does not exists.'.format(traces_dir))
        return

    try:
        max_features = int(sys.argv[2])
    except ValueError:
        print('Error: max_features {0} is not a number.'.format(sys.argv[2]))
        return

    vectorizer_file = sys.argv[3]

    trace_files = sorted(glob.glob("{0}/*.objdumps_both".format(traces_dir)))
    if len(trace_files) < 1:
        print('Error: No *.objdumps_both trace files were found in {0}'.format(traces_dir))
        return

    vectorizer = createTFIDFVectorizer(trace_files, max_features)
    joblib.dump(vectorizer, vectorizer_file)
    print('Vectorizer saved to {0}'.format(vectorizer_file))

if __name__ == "__main__":
    main()
