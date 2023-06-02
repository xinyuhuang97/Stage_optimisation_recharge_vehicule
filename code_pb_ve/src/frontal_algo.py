import sys
import time
import numpy as np
import pandas as pd
sys.path.append('../data')
from tqdm import tqdm
from generator import *
from tools import *
sys.path.append('/Applications/CPLEX_Studio221/cplex/python/3.7/x86-64_osx/cplex/_internal')
sys.path.insert(0, "/Applications/CPLEX_Studio221/cplex/python/3.7/x86-64_osx")
import cplex

from tools import *
if __name__ == "__main__":
    my_instance = "../data/instance_10.json"

actual_time=0
file_name="Frontal_problem.lp"
data=json.load(open(my_instance))
N=len(data["evses"])
T=data["optim_horizon"]

beta_min=data["penalties"]["beta_min"]
beta_max=data["penalties"]["beta_max"]
alpha=data["penalties"]["fcr_up"]



def Frontal(data, actual_time, verbose=False):
    
    N=len(data["evses"]) # Nombre d'evse
    T=data["optim_horizon"] # Nombre de pas de temps
    beta_min=data["penalties"]["beta_min"]
    beta_max=data["penalties"]["beta_max"]
    alpha=data["penalties"]["fcr_up"]
    # Create a new LP problem
    problem = cplex.Cplex()

    if not verbose:
        problem.set_log_stream(None)
        problem.set_error_stream(None)
        problem.set_warning_stream(None)
        problem.set_results_stream(None)

    # Prepare data
    T=data["optim_horizon"]
    N=len(data["evses"])
    soc_init=[data["evses"][i]["SOC_init"]*data["evses"][i]["capacity"] for i in range(N)]
    soc_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)]
    soc_min=[data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"] for i in range(N)]

    p_charge_max=[data["evses"][i]["p_charge_max"] for i in range(N)]
    p_charge_min=[data["evses"][i]["p_charge_min"] for i in range(N)]
    p_discharge_max=[data["evses"][i]["p_discharge_max"] for i in range(N)]
    p_discharge_min=[data["evses"][i]["p_discharge_min"] for i in range(N)]
    
    time_mesh_to_hour = data["time_mesh"]/60
    s_final = [data["evses"][i]["SOC_final"]*data["evses"][i]["capacity"] for i in range(N)]
    s_t_min = [[max(data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"],s_final[i]-p_discharge_max[i]*(T-t)*time_mesh_to_hour ) for t in range(1,T+1)] for i in range(N)]

    cost_electricity = data["cost_of_electricity"]
    penality_SOC_fin = data["penalties"]["SOC_fin"]

    # Create variables
    s_bl = [["s_bl_" + str(i) +"_" +str(t) for t in range(T+1)]  for i in range(N) ]
    c_bl = [["c_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    d_bl = [["d_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    u_bl = [["u_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    v_bl = [["v_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    n_bl = [["n_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    p_bl = [["p_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]

    s_up = [["s_up_" + str(i) +"_" +str(t) for t in range(T+1)]  for i in range(N) ]
    c_up = [["c_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    d_up = [["d_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    u_up = [["u_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    v_up = [["v_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    n_up = [["n_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    p_up = [["p_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]

    y = ["y_" + str(t) for t in range(1,T+1)]
    problem.variables.add(names=y,lb=[-cplex.infinity]*(T),ub=[cplex.infinity]*(T))
    for i in range(N):
        problem.variables.add(names=s_bl[i])
        problem.variables.add(names=s_up[i])
        problem.variables.add(names=c_bl[i])
        problem.variables.add(names=c_up[i])
        problem.variables.add(names=d_bl[i])
        problem.variables.add(names=d_up[i])
        problem.variables.add(types=['B']*T,names=u_bl[i])
        problem.variables.add(types=['B']*T,names=u_up[i])
        problem.variables.add(types=['B']*T,names=v_bl[i])
        problem.variables.add(types=['B']*T,names=v_up[i])
        problem.variables.add(names=n_bl[i])
        problem.variables.add(names=n_up[i])
        problem.variables.add(names=p_bl[i])
        problem.variables.add(names=p_up[i])
    
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(0)],val=[1])],senses=["E"],rhs=[soc_init[i]],names=["Initial SOC bl "+str(i)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(0)],val=[1])],senses=["E"],rhs=[soc_init[i]],names=["Initial SOC up "+str(i)])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[soc_max[i] for _ in range(1,T+1)],names=["SOC max bl "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[soc_max[i] for _ in range(1,T+1)],names=["SOC max up "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[soc_min[i] for _ in range(1,T+1)],names=["SOC min bl "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[soc_min[i] for _ in range(1,T+1)],names=["SOC min up "+str(i)+" "+str(t) for t in range(1,T+1)])
        
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bl_"+str(i)+"_"+str(t),"v_bl_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[1]*T,names=["charge or discharge bl "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_up_"+str(i)+"_"+str(t),"v_up_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[1]*T,names=["charge or discharge up "+str(i)+" "+str(t) for t in range(1,T+1)])
        
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_bl_"+str(i)+"_"+str(t),"u_bl_"+str(i)+"_"+str(t)],val=[1,-p_charge_max[i]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T,names=["charge upper bound bl "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_up_"+str(i)+"_"+str(t),"u_up_"+str(i)+"_"+str(t)],val=[1,-p_charge_max[i]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T,names=["charge upper bound up "+str(i)+" "+str(t) for t in range(1,T+1)])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_bl_"+str(i)+"_"+str(t),"v_bl_"+str(i)+"_"+str(t)],val=[1,-p_discharge_max[i]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T,names=["discharge upper bound bl "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_up_"+str(i)+"_"+str(t),"v_up_"+str(i)+"_"+str(t)],val=[1,-p_discharge_max[i]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T,names=["discharge upper bound up "+str(i)+" "+str(t) for t in range(1,T+1)])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bl_"+str(i)+"_"+str(t),"c_bl_"+str(i)+"_"+str(t)],val=[p_charge_min[i],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T,names=["charge lower bound bl "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_up_"+str(i)+"_"+str(t),"c_up_"+str(i)+"_"+str(t)],val=[p_charge_min[i],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T,names=["charge lower bound up "+str(i)+" "+str(t) for t in range(1,T+1)])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_bl_"+str(i)+"_"+str(t),"d_bl_"+str(i)+"_"+str(t)],val=[p_discharge_min[i],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T,names=["discharge lower bound bl "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_up_"+str(i)+"_"+str(t),"d_up_"+str(i)+"_"+str(t)],val=[p_discharge_min[i],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T,names=["discharge lower bound up "+str(i)+" "+str(t) for t in range(1,T+1)])
        
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t+1),"s_bl_"+str(i)+"_"+str(t),"c_bl_"+str(i)+"_"+str(t+1),"d_bl_"+str(i)+"_"+str(t+1)],\
                                                                  val=[1,-1,-time_mesh_to_hour,time_mesh_to_hour]) for t in range(T)],senses=["E"]*(T),rhs=[0]*(T),names=["Production balance bl"+str(i)+" "+str(t) for t in range(T)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t+1),"s_bl_"+str(i)+"_"+str(t),"c_up_"+str(i)+"_"+str(t+1),"d_up_"+str(i)+"_"+str(t+1)],\
                                                                  val=[1,-1,-time_mesh_to_hour,time_mesh_to_hour]) for t in range(T)],senses=["E"]*(T),rhs=[0]*(T),names=["Production balance up"+str(i)+" "+str(t) for t in range(T)])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["n_bl_"+str(i)+"_"+str(t),"s_bl_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["G"]*T,rhs=s_t_min[i],names=["Negative part bl "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["n_up_"+str(i)+"_"+str(t),"s_up_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["G"]*T,rhs=s_t_min[i],names=["Negative part up "+str(i)+" "+str(t) for t in range(1,T+1)])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["p_bl_"+str(i)+"_"+str(t),"s_bl_"+str(i)+"_"+str(t)],val=[1,-1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[-soc_max[i]]*T,names=["Positive part bl "+str(i)+" "+str(t) for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["p_up_"+str(i)+"_"+str(t),"s_up_"+str(i)+"_"+str(t)],val=[1,-1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[-soc_max[i]]*T,names=["Positive part up "+str(i)+" "+str(t) for t in range(1,T+1)])
    
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["y_"+str(t)]+[x for i in range(N) for x in ["c_bl_"+str(i)+"_"+str(t),"c_up_"+str(i)+"_"+str(t),"d_bl_"+str(i)+"_"+str(t),"d_up_"+str(i)+"_"+str(t)]],\
                                                              val=[1]+[x for _ in range(N) for x in [-1/N,1/N,1/N,-1/N]]) for t in range(1,T+1)],senses=["E"]*T,\
                                                              rhs=[-x/N for x in data["announced_capacity"]["up"][actual_time:actual_time+T]],names=["y_t facilitate calculations"+str(t) for t in range(1,T+1)])

    # Set the objective function
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Add function f 
    for t in range(1,T+1):
        problem.objective.set_quadratic_coefficients("y_"+str(t),"y_"+str(t),2*alpha)

    # Add terms concerning negative/positive parts
    
    for i in range(N):
        for t in range(1,T+1):
            problem.objective.set_quadratic_coefficients("n_bl_"+str(i)+"_"+str(t),"n_bl_"+str(i)+"_"+str(t),2*beta_min/N)
            problem.objective.set_quadratic_coefficients("n_up_"+str(i)+"_"+str(t),"n_up_"+str(i)+"_"+str(t),2*beta_min/N)
            problem.objective.set_quadratic_coefficients("p_bl_"+str(i)+"_"+str(t),"p_bl_"+str(i)+"_"+str(t),2*beta_max/N)
            problem.objective.set_quadratic_coefficients("p_up_"+str(i)+"_"+str(t),"p_up_"+str(i)+"_"+str(t),2*beta_max/N)
    
    # Add electricity cost 
    exprs_electricity_cost = [x for i in range(N) for t in range(1,T+1) for x in ["c_bl_"+str(i)+"_"+str(t),"d_bl_"+str(i)+"_"+str(t)]]
    val_electricity_cost = [x for _ in range(N) for t in range(1,T+1) for x in [time_mesh_to_hour*cost_electricity[t]/N,-time_mesh_to_hour*cost_electricity[t]/N]]
    
    # Add reward on final soc
    exprs_final_soc = ["s_bl_"+str(i)+"_"+str(T) for i in range(N)]
    val_final_soc = [-penality_SOC_fin*cost_electricity[T]/N]*N

    
    problem.objective.set_linear(zip(exprs_electricity_cost+exprs_final_soc,val_electricity_cost+val_final_soc))

    problem.write(file_name)
    problem.solve()
    
    #print("Valeur optimal retourne par cplex",problem.solution.get_objective_value()) 

    sample_c_bl=np.zeros((N,T))
    sample_d_bl=np.zeros((N,T))
    sample_c_up=np.zeros((N,T))
    sample_d_up=np.zeros((N,T))
    sample_s_bl=np.zeros((N,T))
    sample_s_up=np.zeros((N,T))
    sample_n_bl=np.zeros((N,T))
    sample_n_up=np.zeros((N,T))
    sample_p_bl=np.zeros((N,T))
    sample_p_up=np.zeros((N,T))
    for i in range(N):
        sample_c_bl[i]=(problem.solution.get_values(c_bl[i]))
        sample_d_bl[i]=(problem.solution.get_values(d_bl[i]))
        sample_c_up[i]=(problem.solution.get_values(c_up[i]))
        sample_d_up[i]=(problem.solution.get_values(d_up[i]))
        sample_s_bl[i]=(problem.solution.get_values(s_bl[i][1:]))
        sample_s_up[i]=(problem.solution.get_values(s_up[i][1:]))
        for t in range(1,T+1):
            sample_n_bl[i][t-1]=(problem.solution.get_values("n_bl_"+str(i)+"_"+str(t)))
            sample_n_up[i][t-1]=(problem.solution.get_values("n_up_"+str(i)+"_"+str(t)))
            sample_p_bl[i][t-1]=(problem.solution.get_values("p_bl_"+str(i)+"_"+str(t)))
            sample_p_up[i][t-1]=(problem.solution.get_values("p_up_"+str(i)+"_"+str(t)))

    
    print("n_bl",sample_n_bl)
    print("n_up",sample_n_up)
    print("p_bl",sample_p_bl)
    print("p_up",sample_p_up)
    print("c_bl",sample_c_bl)
    print("d_bl",sample_d_bl)
    print("c_up",sample_c_up)
    print("d_up",sample_d_up)
    print("s_bl",sample_s_bl)
    print("s_up",sample_s_up)
    print("s_t_min",s_t_min)
    print("s_t_max",soc_max)
    print("y",problem.solution.get_values(y))
    print([x**2 for x in (problem.solution.get_values(y))])
    
    print_objective_function(file_name)
    result ={"c_bl": sample_c_bl, "d_bl": sample_d_bl, \
            "c_up": sample_c_up, "d_up": sample_d_up,\
            "s_bl": sample_s_bl, "s_up": sample_s_up}
    #print("Valeur optimal retourne par la fonction",objective_function(data,result,s_t_min,soc_max))
    print("valeur optimal par function objective:", objective_function(data, result, s_t_min, soc_max))
    return result,s_t_min,soc_max


def objective_function(data, x, s_i_t_min, s_i_max):

    alpha=data["penalties"]["fcr_up"]
    penality_SOC_fin = data["penalties"]["SOC_fin"]
    # Recuperer les valeurs des variables
    c_bl=x["c_bl"]
    d_bl=x["d_bl"]
    c_up=x["c_up"]
    d_up=x["d_up"]
    s_bl=x["s_bl"]
    s_up=x["s_up"]
   
    f_val= alpha*np.sum([ (np.sum([c_bl[i][t]-d_bl[i][t]-c_up[i][t]+d_up[i][t] for i in range(N) ])/N-data["announced_capacity"]["up"][actual_time+t]/N)**2 for t in range(T)  ])
    s_T=s_bl[:,T-1]

    soc_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)]

    # Calculer la fonction objectif
    cost=0 # Valeur de la fonction objectif
    
    for i in range(N):
        for t in range(actual_time,actual_time+T):
            neg_pos_part =  (beta_min*(-min(s_bl[i][t]-s_i_t_min[i][t],0))**2 + \
                  beta_max*max(s_bl[i][t]-s_i_max[i],0)**2 + \
                  beta_min*(-min(s_up[i][t]-s_i_t_min[i][t],0))**2+ \
                  beta_max*max(s_up[i][t]-s_i_max[i],0)**2)/N
            cost_electricity = (c_bl[i][t]-d_bl[i][t])*data["cost_of_electricity"][t]*data["time_mesh"]/60/N
            cost+=neg_pos_part+cost_electricity
        # We want the level of c_bl to be the high at the end of the optimization horizon
        #print("s_T[i]",s_T[i])
        #print("s_i_max[i]",soc_max[i])
        exprs_final_soc=(soc_max[i]-s_T[i])*data["cost_of_electricity"][actual_time+T]*penality_SOC_fin/N 
        cost+=exprs_final_soc
    
    cost+=f_val #+ np.sum([1/N*data["announced_capacity"]["up"][actual_time]*data["cost_of_electricity"][actual_time]*x for x in soc_max])
    return cost

#x,s_i_t_min,s_i_max =Frontal(data, actual_time, verbose=False)
#print("valeur optimal retourne par la fonction",objective_function(data,x,s_i_t_min,s_i_max))
