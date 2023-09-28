import numpy as np
import pandas as pd
import json
import time
import sys
import os
from copy import deepcopy
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


def cplex_to_file(my_instance, time ,optimal_value):
    instance = my_instance.split("/")[-1]
    instance_name = instance.replace(".json", "")
    filename="../result/cplex/" +instance_name+"_cplex" + ".txt"
    # Format the values as a string
    output_string = f"Time: {time}, Optimal_value: {optimal_value}"
    # Save the formatted string to a file
    with open(filename, "w") as file:
        file.write(output_string)
    print("Result saved in file", filename)

def read_cplexfile(my_instance):
    instance = my_instance.split("/")[-1]
    instance_name = instance.replace(".json", "")
    filename="../result/cplex/" + "/"+instance_name+"_cplex" + ".txt"
    with open(filename, "r") as file:
        content = file.read()
    time_str, optimal_val_str = content.split(", ")
    time = float(time_str.split(": ")[1])
    optimal_value = float(optimal_val_str.split(": ")[1])
    return time, optimal_value

def execute_frontal(algo, actual_time,my_instance):
    instance = my_instance.split("/")[-1]
    instance_name = instance.replace(".json", "")
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    folder_path = os.path.join(parent_directory+"/log/", instance_name)
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{instance_name}' created successfully at '{folder_path}'")
    filename=folder_path+"/" + "cplex" + "_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    start_time = time.time()
    original_stdout = sys.stdout
    print("Output is saved in file", filename)
    with open(filename,"w") as file:    
        sys.stdout=file
        result, s_t_min, soc_max, optimal_value=algo.Frontal_solve(actual_time)
        sys.stdout.flush()
    end_time = time.time()
    sys.stdout = original_stdout
    save=deepcopy(result)
    save["c_bl"]=save["c_bl"].tolist()
    save["c_up"]=save["c_up"].tolist()
    save["d_bl"]=save["d_bl"].tolist()
    save["d_up"]=save["d_up"].tolist()
    save["s_bl"]=save["s_bl"].tolist()
    save["s_up"]=save["s_up"].tolist()
    save_cplex_data(my_instance, save)
    cplex_to_file(instance, end_time-start_time ,optimal_value)
    return optimal_value, end_time-start_time

def save_cplex_data(my_instance, data):
    instance = my_instance.split("/")[-1]
    instance_name = instance.replace(".json", "")
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    folder_path = os.path.join(parent_directory+"/result/cplex/", instance_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{instance_name}' created successfully at '{folder_path}'")
    filename="../result/cplex/" + "/"+instance_name+"/" + "cplex"  + "_"  + time.strftime("%Y%m%d-%H%M%S") + ".json"
    #data.to_json(filename, orient="records")
    with open(filename, "w") as json_file:
        json.dump(data, json_file)

#def execute_cplex(algo, acutal_time):
def execute_algo(algo, actual_time, K, n_k, n_pre=0, analyse=True, optimize=False,gap_calculate=True,pb=0):
    if pb==0:
        name_pb="pb_original"
    elif pb==1:
        name_pb="pb_booster"
    else:
        name_pb="pb_reduit"
    instance = algo.instance.split("/")[-1]
    instance_name = instance.replace(".json", "")
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    folder_path = os.path.join(parent_directory+"/log/", instance_name)
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{instance_name}' created successfully at '{folder_path}'")
    if n_pre!=0:
        filename="../log/" +instance_name+"/" + str(algo.algo_name) +  "_" + name_pb + "_" + str(n_pre) + "_" + str(K) + "_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    else:
        filename="../log/" +instance_name+"/" + str(algo.algo_name)+  "_" + name_pb + "_" + str(K) + "_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    original_stdout = sys.stdout
    print("Output is saved in file", filename)
    start_time = time.time()
    with open(filename,"w") as file:
        sys.stdout=file
        if n_pre!=0:
            x_bar_k,df=algo.solve(actual_time, K=K, nk=n_k, n_pre=n_pre, analyse=analyse, optimize=optimize,gap_calculate=gap_calculate)
        else:
            x_bar_k,df=algo.solve(actual_time, K=K, nk=n_k, analyse=analyse, optimize=optimize,gap_calculate=gap_calculate)
        sys.stdout.flush()
    end_time = time.time()
    sys.stdout = original_stdout
    return x_bar_k,df, end_time-start_time
    
def show_file_list_by_instance(pb, instance):
    instance = instance.split("/")[-1]
    instance_name = instance.replace(".json", "")
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    
    file_CFW1=[]
    file_CFW2=[]
    file_CFW3=[]
    file_SFW1=[]
    file_SFW2=[]
    file_Robust=[]
    path=""
    #folder_path = os.path.join(parent_directory+"/result/"), instance_name)
    if pb==1:
        path=parent_directory+"/result/pb_original/"+instance_name
        file_list=os.listdir(path)
        print(path)
    elif pb==2:
        path=parent_directory+"/result/pb_original&booster/"+instance_name
        file_list=os.listdir(path)
        print(path)
    elif pb==3:
        path=parent_directory+"/result/pb_reduced/"+instance_name
        file_list=os.listdir(path)
        print(path)
    for item in file_list:
        print(item)
        if pb !=3:
            if item.startswith("CFW1"):
                file_CFW1.append(path+"/"+item)
            elif item.startswith("CFW2"):
                file_CFW2.append(path+"/"+item)
            elif item.startswith("CFW3"):
                file_CFW3.append(path+"/"+item)
            elif item.startswith("SFW1"):
                file_SFW1.append(path+"/"+item)
            elif item.startswith("SFW2"):
                file_SFW2.append(path+"/"+item)
        else:
            if item.startswith("PR_CFW1"):
                file_CFW1.append(path+"/"+item)
            elif item.startswith("PR_CFW2"):
                file_CFW2.append(path+"/"+item)
            elif item.startswith("PR_CFW3"):
                file_CFW3.append(path+"/"+item)
            elif item.startswith("PR_SFW1"):
                file_SFW1.append(path+"/"+item)
            elif item.startswith("PR_SFW2"):
                file_SFW2.append(path+"/"+item)
            elif item.startswith("PR_Robust"):
                file_Robust.append(path+"/"+item)
                
    return file_CFW1, file_CFW2, file_CFW3, file_SFW1, file_SFW2, file_Robust

def show_cplexfile_list_by_instance(instance):
    instance = instance.split("/")[-1]
    instance_name = instance.replace(".json", "")
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    
    file_cplex=[]
    path=""
    #folder_path = os.path.join(parent_directory+"/result/"), instance_name)
    path=parent_directory+"/result/cplex/"+instance_name
    file_list=os.listdir(path)
    print(path)

    for item in file_list:
        print(item)
        file_cplex.append(path+"/"+item)
    return file_cplex


def choose_file_by_novelty(file_names, position):
    def get_date_from_filename(file_name):
        # Split the file name and extract the date part
        date_part = file_name.split("_")[-1].split(".")[0]
        return date_part
    
    sorted_file_names = sorted(file_names, key=get_date_from_filename)
    
    selected_file = sorted_file_names[position]
    return selected_file

def choose_and_read(file_names, position):
    selected_file = choose_file_by_novelty(file_names, position)

    #df = pd.read_csv("your_file.csv", dtype=data_types)
    return pd.read_json(selected_file, orient="records")


def read_cplex_data(file_names):
    def get_date_from_filename(file_name):
        # Split the file name and extract the date part
        date_part = file_name.split("_")[-1].split(".")[0]
        return date_part
    
    sorted_file_names = sorted(file_names, key=get_date_from_filename)
    
    selected_file = sorted_file_names[0]
    return pd.read_json(selected_file, orient="records")