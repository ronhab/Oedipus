import os, subprocess, sys, random

def generate_random_program(source_dir, seed, random_function):
    seed_str = '{0:0{1}X}'.format(seed,8)
    log_path = os.path.join(source_dir, '%s.log' % seed_str)
    generated_program_path = os.path.join(source_dir, '%s_before.c' % seed_str)
    final_program_path = os.path.join(source_dir, '%s.c' % seed_str)
    if os.path.exists(final_program_path):
        print('source file %s already exists' % final_program_path)
        return None
    
    empty_file = os.path.join(source_dir, 'empty.c')
    with open(empty_file, 'w'):
        tigress_cmd = ['tigress']
        tigress_cmd.append('--Verbosity=1')
        tigress_cmd.append('--Seed={0}'.format(seed))
        tigress_cmd.append('--Transform=RandomFuns')
        tigress_cmd.append('--RandomFunsName={0}'.format(random_function))
        tigress_cmd.append('--RandomFunsInputSize=1')
        tigress_cmd.append('--RandomFunsStateSize=1')
        tigress_cmd.append('--RandomFunsOutputSize=1')
        tigress_cmd.append('--FilePrefix=rnd')
        tigress_cmd.append('--Environment=x86_64:Linux:Gcc:4.6')
        tigress_cmd.append('--out={0}'.format(generated_program_path))
        tigress_cmd.append('empty.c')
        tigress_output = subprocess.Popen(tigress_cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, cwd=source_dir).communicate()[0]
        with open(log_path, 'w') as log_file:
            log_file.write(str(tigress_output))
        with open(generated_program_path, 'r') as generated_file:
            file_content = generated_file.read()
            # fix issues with function declarations not matching standard lib declarations
            problematic_func_declarations = [
                'extern int fclose(',
                'extern unsigned long write(',
                'extern long strtol(',
                'extern unsigned long read(',
                'extern float strtof(',
                'extern double strtod(',
                'extern void *fopen(',
                'extern void signal(',
                'extern unsigned long strtoul('
            ]
            for func_decl in problematic_func_declarations:
                file_content = file_content.replace(func_decl, '//' + func_decl, 1)
            with open(final_program_path, 'w') as output_file:
                output_file.write(file_content)
    os.remove(empty_file)
    os.remove(os.path.join(source_dir, 'a.out'))
    os.remove(generated_program_path)
    return final_program_path

def generate_random_programs(source_dir, number_of_programs, random_function):
    random.seed(42)
    for i in range(number_of_programs):
        seed = random.randint(0, 2**32)
        generate_random_program(source_dir, seed, random_function)
        print('random program #{0} generated'.format(i))
    return True