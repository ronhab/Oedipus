from Oedipus.gadgets import random_programs
import sys, os

def main():
    if len(sys.argv) != 4:
        print('Usage: python generate_random_programs.py <destination_dir> <number_of_programs> <random_function_name>')
        return
    
    dest_dir = sys.argv[1]
    if not os.path.exists(dest_dir):
        print('Error: destination_dir {0} does not exists.'.format(dest_dir))
        return

    try:
        number_of_programs = int(sys.argv[2])
    except ValueError:
        print('Error: number_of_programs {0} is not a number.'.format(sys.argv[2]))
        return

    random_function_name = sys.argv[3]

    if random_programs.generate_random_programs(dest_dir, number_of_programs, random_function_name):
        print("Successfully generated {0} random programs".format(number_of_programs))
    else:
        print("Some error occurred during random program generation")

if __name__ == "__main__":
    main()
