import sys
import time
import numpy as np
import pandas as pd
sys.path.append('../data')
from tqdm import tqdm
from generator import *
from subpb_pl_cplex import *
from copy import deepcopy

# Generer une nouvelle instance a=instance_json(20)
#instance_json(3)
data=json.load(open('../data/instance_100.json'))
N=len(data["evses"]) # Nombre d'evse
T=data["optim_horizon"] # Nombre de pas de temps
actuel_time=0
def Frontal(data, actuel_time, verbose=False):
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
    s_t_min = [[max(data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"],s_final[i]-p_discharge_max[i]*(T-t) ) for t in range(1,T+1)] for i in range(N)]

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
    problem.variables.add(names=y)
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
    
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(0)],val=[1])],senses=["E"],rhs=[soc_init[i]],names=["constraint_s_"+str(0)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(0)],val=[1])],senses=["E"],rhs=[soc_init[i]])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[soc_max[i] for _ in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[soc_max[i] for _ in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[soc_min[i] for _ in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[soc_min[i] for _ in range(1,T+1)])
        
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bl_"+str(i)+"_"+str(t),"v_bl_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[1]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_up_"+str(i)+"_"+str(t),"v_up_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[1]*T)
        
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_bl_"+str(i)+"_"+str(t),"u_bl_"+str(i)+"_"+str(t)],val=[1,-p_charge_max[i]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_up_"+str(i)+"_"+str(t),"u_up_"+str(i)+"_"+str(t)],val=[1,-p_charge_max[i]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_bl_"+str(i)+"_"+str(t),"v_bl_"+str(i)+"_"+str(t)],val=[1,-p_discharge_max[i]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_up_"+str(i)+"_"+str(t),"v_up_"+str(i)+"_"+str(t)],val=[1,-p_discharge_max[i]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bl_"+str(i)+"_"+str(t),"c_bl_"+str(i)+"_"+str(t)],val=[p_charge_min[i],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_up_"+str(i)+"_"+str(t),"c_up_"+str(i)+"_"+str(t)],val=[p_charge_min[i],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_bl_"+str(i)+"_"+str(t),"d_bl_"+str(i)+"_"+str(t)],val=[p_discharge_min[i],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_up_"+str(i)+"_"+str(t),"d_up_"+str(i)+"_"+str(t)],val=[p_discharge_min[i],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[]*T)
        
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t+1),"s_bl_"+str(i)+"_"+str(t),"c_bl_"+str(i)+"_"+str(t+1),"d_bl_"+str(i)+"_"+str(t+1)],\
                                                                  val=[1,-1,-time_mesh_to_hour,time_mesh_to_hour]) for t in range(T)],senses=["E"]*(T),rhs=[0]*(T))
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t+1),"s_bl_"+str(i)+"_"+str(t),"c_up_"+str(i)+"_"+str(t+1),"d_up_"+str(i)+"_"+str(t+1)],\
                                                                  val=[1,-1,-time_mesh_to_hour,time_mesh_to_hour]) for t in range(T)],senses=["E"]*(T),rhs=[0]*(T))

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["n_bl_"+str(i)+"_"+str(t),"s_bl_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["G"]*T,rhs=s_t_min[i])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["n_up_"+str(i)+"_"+str(t),"s_up_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["G"]*T,rhs=s_t_min[i])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["p_bl_"+str(i)+"_"+str(t),"s_bl_"+str(i)+"_"+str(t)],val=[1,-1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[-soc_max[i]]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["p_up_"+str(i)+"_"+str(t),"s_up_"+str(i)+"_"+str(t)],val=[1,-1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[-soc_max[i]]*T)
    
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["y_"+str(t)]+[x for i in range(N) for x in ["c_bl_"+str(i)+"_"+str(t),"c_up_"+str(i)+"_"+str(t),"d_bl_"+str(i)+"_"+str(t),"d_up_"+str(i)+"_"+str(t)]],\
                                                              val=[1]+[x for _ in range(N) for x in [-1/N,1/N,1/N,-1/N]]) for t in range(1,T+1)],senses=["E"]*T,\
                                                              rhs=[-x/N for x in data["announced_capacity"]["up"][actuel_time:actuel_time+T]])

    # Set the objective function
    problem.objective.set_sense(problem.objective.sense.minimize)
    for i in range(N):
        for t in range(1,T+1):
            problem.objective.set_quadratic_coefficients("n_bl_"+str(i)+"_"+str(t),"n_bl_"+str(i)+"_"+str(t),2/N)
            problem.objective.set_quadratic_coefficients("n_up_"+str(i)+"_"+str(t),"n_up_"+str(i)+"_"+str(t),2/N)
            problem.objective.set_quadratic_coefficients("p_bl_"+str(i)+"_"+str(t),"p_bl_"+str(i)+"_"+str(t),2/N)
            problem.objective.set_quadratic_coefficients("p_up_"+str(i)+"_"+str(t),"p_up_"+str(i)+"_"+str(t),2/N)

    for t in range(1,T+1):
        problem.objective.set_quadratic_coefficients("y_"+str(t),"y_"+str(t),2)
    
    exprs_electric_cost = [x for i in range(N) for t in range(1,T+1) for x in ["c_bl_"+str(i)+"_"+str(t),"d_bl_"+str(i)+"_"+str(t)]]
    val_electric_cost = [x for _ in range(N) for t in range(1,T+1) for x in [time_mesh_to_hour*cost_electricity[t]/N,-time_mesh_to_hour*cost_electricity[t]/N]]
    
    exprs_final_soc = ["s_bl_"+str(i)+"_"+str(T) for i in range(N)]
    val_final_soc = [-1/N*penality_SOC_fin*cost_electricity[T]]*N

    problem.objective.set_linear(zip(exprs_electric_cost+exprs_final_soc,val_electric_cost+val_final_soc))

    problem.write("test_frontal.lp")
    problem.solve()
    
    print("valeur optimal retourne par cplex",problem.solution.get_objective_value()) 
    """
    for i in range(N):
        print(problem.solution.get_values(c_bl[i]))
        print(problem.solution.get_values(c_up[i]))
        print(problem.solution.get_values(d_bl[i]))
        print(problem.solution.get_values(d_up[i]))
        print("s")
        print(problem.solution.get_values(s_bl[i]))
        print(problem.solution.get_values(s_up[i]))
    for t in range(1,T+1):
        print("y",problem.solution.get_values("y_"+str(t)))
    print(data["announced_capacity"]["up"][actuel_time:actuel_time+T])
    print(s_t_min)
    print(soc_max)
    print("what",data["announced_capacity"]["up"][actuel_time:actuel_time+T])
    print(val_final_soc)
    """
    sample_c_bl=np.zeros((N,T))
    sample_d_bl=np.zeros((N,T))
    sample_c_up=np.zeros((N,T))
    sample_d_up=np.zeros((N,T))
    sample_s_bl=np.zeros((N,T))
    sample_s_up=np.zeros((N,T))
    for i in range(N):
        sample_c_bl[i]=(problem.solution.get_values(c_bl[i]))
        sample_d_bl[i]=(problem.solution.get_values(c_up[i]))
        sample_c_up[i]=(problem.solution.get_values(d_bl[i]))
        sample_d_up[i]=(problem.solution.get_values(d_up[i]))
        sample_s_bl[i]=(problem.solution.get_values(s_bl[i][1:]))
        sample_s_up[i]=(problem.solution.get_values(s_up[i][1:]))
    """
    print("c_bl",sample_c_bl)
    print("d_bl",sample_d_bl)
    print("c_up",sample_c_up)
    print("d_up",sample_d_up)
    print("s_bl",sample_s_bl)
    print("s_up",sample_s_up)
    print(s_t_min)
    """
    result ={"c_bl": sample_c_bl, "d_bl": sample_d_bl, \
            "c_up": sample_c_up, "d_up": sample_d_up,\
            "soc": sample_s_bl, "soc_bar": sample_s_up}
    return result,s_t_min,soc_max


def objective_function(data, x, s_i_t_min, s_i_max):
    beta_min=1
    beta_max=1
    penality_SOC_fin = data["penalties"]["SOC_fin"]
    # Recuperer les valeurs des variables
    c_bl=x["c_bl"]
    d_bl=x["d_bl"]
    c_up=x["c_up"]
    d_up=x["d_up"]
    s_bl=x["soc"]
    s_up=x["soc_bar"]
    y_val= np.sum([ (np.sum([c_bl[i][t]-d_bl[i][t]-c_up[i][t]+d_up[i][t] for i in range(N) ])/N-data["announced_capacity"]["up"][actuel_time+t]/N)**2 for t in range(T)  ])
    s_T=s_bl[:,T-1]
    # Calculer la fonction objectif
    cost=0 # Valeur de la fonction objectif
    
    for i in range(N):
        for j in range(actuel_time,actuel_time+data["optim_horizon"]):
            hi=  (beta_min*(-min(s_bl[i][j]-s_i_t_min[i][j],0))**2 + \
                  beta_max*max(s_bl[i][j]-s_i_max[i],0)**2 + \
                  beta_min*(-min(s_up[i][j]-s_i_t_min[i][j],0))**2+ \
                  beta_max*max(s_up[i][j]-s_i_max[i],0)**2)
            cost_electricity = (c_bl[i][j]-d_bl[i][j])*data["cost_of_electricity"][j]*data["time_mesh"]/60/N
            cost+=hi+cost_electricity
        # We want the level of c_bl to be the high at the end of the optimization horizon
        cost-=s_T[i]*data["cost_of_electricity"][actuel_time+data["optim_horizon"]]*penality_SOC_fin/N 
    #print(y_val,"y_val")
    cost+=y_val
    #print(cost1, cost2)
    return cost

x,s_i_t_min,s_i_max =Frontal(data, actuel_time, verbose=False)
print("valeur optimal retourne par la fonction",objective_function(data,x,s_i_t_min,s_i_max))