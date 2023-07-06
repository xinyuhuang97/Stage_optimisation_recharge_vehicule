import numpy as np
#from parameters import *
def print_objective_function(lp_filename):
    with open(lp_filename, 'r') as f:
        read_obj = False
        for line in f:
            if 'Minimize' in line:
                read_obj = True
            if 'Subject To' in line:
                read_obj = False
            if read_obj:
                print(line.strip())