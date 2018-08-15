from Oedipus.gadgets import program_generation
import sys, os, glob

def main():
    if len(sys.argv) != 5:
        print('Usage: python obfuscate_programs.py <programs_dir> <tigress_dir> <function_to_obfuscate> <obfuscation_level>')
        return
    
    programs_dir = sys.argv[1]
    if not os.path.exists(programs_dir):
        print('Error: programs_dir {0} does not exists.'.format(programs_dir))
        return

    tigress_dir = sys.argv[2]
    if not os.path.exists(tigress_dir):
        print('Error: tigress_dir {0} does not exists.'.format(tigress_dir))
        return

    function_to_obfuscate = sys.argv[3]

    try:
        obfuscation_level = int(sys.argv[4])
    except ValueError:
        print('Error: obfuscation_level {0} is not a number.'.format(sys.argv[4]))
        return

    programs_to_obfuscate = sorted(glob.glob("{0}/*.c".format(programs_dir)))
    if len(programs_to_obfuscate) < 1:
        print('Error: No C files were found in {0}'.format(programs_dir))
        return

    program_generation.generateObfuscatedPrograms(programs_to_obfuscate, tigress_dir, obfuscation_level, function_to_obfuscate)
    print("Successfully generated obfuscated programs")

if __name__ == "__main__":
    main()
