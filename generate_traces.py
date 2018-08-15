from Oedipus.gadgets.feature_extraction import generateTraces
from Oedipus.utils.misc import cleanUp
from Oedipus.utils.data import filterTraces
import sys, os, glob

def main():
    if len(sys.argv) != 3:
        print('Usage: python generate_traces.py <programs_dir> <obfuscated_function>')
        return
    
    programs_dir = sys.argv[1]
    if not os.path.exists(programs_dir):
        print('Error: programs_dir {0} does not exists.'.format(programs_dir))
        return

    obfuscated_function = sys.argv[2]

    source_files = sorted(glob.glob("{0}/*.c".format(programs_dir)))
    if len(source_files) < 1:
        print('No C files were found in {0}'.format(programs_dir))
        
    for i,program in enumerate(source_files):
        if not os.path.exists(program.replace(".c", ".label")):
            print('File {0} does not have a label file. Removing'.format(program))
            source_files.pop(i)

    print("Generating traces")
    if not generateTraces(programs_dir, source_files):
        print("Could not generate traces from source files. Exiting")
        return
    print("Successfully generated traces")
    print("Filtering traces")
    if filterTraces(programs_dir, 'objdumps', 'both', 'tfidfobjs_both', obfuscated_function):
        print('Successfully filtered traces')
    else:
        print('Some error occurred during filteration')
    cleanUp()

if __name__ == "__main__":
    main()
