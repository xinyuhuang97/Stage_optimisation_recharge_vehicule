
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
data=json.load(open('../data/instance_3.json'))
N=len(data["evses"]) # Nombre d'evse
T=data["optim_horizon"] # Nombre de pas de temps
M=15 # Nombre de pas de temps dans un intervalle de temps
# Actuel_time \in {0,1,...,H-1}???
actuel_time=0

def Frontal(data, actuel_time, verbose=False):
    # Create a new LP problem
    problem = cplex.Cplex()

    #problem.parameters.mip.tolerances.mipgap.set(0.1)

    #optimality_target = 0.001  # Adjust the value as desired

    # Set the optimality target parameter
    #problem.parameters.mip.tolerances.mipgap.set(optimality_target)
    if not verbose:
        problem.set_log_stream(None)
        problem.set_error_stream(None)
        problem.set_warning_stream(None)
        problem.set_results_stream(None)

    # Prepare data
    T=data["optim_horizon"]
    N=len(data["evses"])

    soc_init=[data["evses"][k]["SOC_init"]*data["evses"][k]["capacity"] for k in range(N)]
    soc_max=[data["evses"][k]["SOC_max"]*data["evses"][k]["capacity"] for k in range(N)]
    soc_min=[data["evses"][k]["SOC_min"]*data["evses"][k]["capacity"] for k in range(N)]
    soc_final=[data["evses"][k]["SOC_final"]*data["evses"][k]["capacity"] for k in range(N)]
    p_charge_max=[data["evses"][k]["p_charge_max"] for k in range(N)]
    p_charge_min=[data["evses"][k]["p_charge_min"] for k in range(N)]
    p_discharge_max=[data["evses"][k]["p_discharge_max"] for k in range(N)]
    p_discharge_min=[data["evses"][k]["p_discharge_min"] for k in range(N)]
    
    time_mesh_to_hour = data["time_mesh"]/60
    s_final = [data["evses"][k]["SOC_final"]*data["evses"][k]["capacity"] for k in range(N)]
    s_t_min = [[max(data["evses"][k]["SOC_min"]*data["evses"][k]["capacity"],s_final[k]-p_discharge_max[k]*(T-t) ) for t in range(1,T+1)] for k in range(N)]
    s_max = [[-data["evses"][k]["SOC_max"]*data["evses"][k]["capacity"] for t in range(1,T+1)] for k in range(N)]
    #s_max = [[-data["evses"][k]["SOC_max"]*data["evses"][k]["capacity"] for t in range(1,T+1)]for k in range(N)]
    cost_electricity = data["cost_of_electricity"]
    penality_SOC_fin = data["penalties"]["SOC_fin"]

    # Create variables
    s = [["s_" + str(i) +"_" +str(t) for t in range(T+1)]for i in range(N) ]
    c = [["c_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    d = [["d_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    u = [["u_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    v = [["v_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    n = [["n_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    p = [["p_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]

    s_bar = [["s_bar_" + str(i) +"_" +str(t) for t in range(T+1)]for i in range(N) ]
    c_bar = [["c_bar_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    d_bar = [["d_bar_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    u_bar = [["u_bar_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    v_bar = [["v_bar_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    n_bar = [["n_bar_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
    p_bar = [["p_bar_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]

    y = [ "y_" + str(t) for t in range(1,T+1) ]
    #print(s)
    #print([[soc_min[k]]*(T+1) for k in range(N)])
    problem.variables.add(names=y)
    for k in range(N):
        problem.variables.add(lb=[soc_min[k]]*(T+1),ub=[soc_max[k]]*(T+1),names=s[k])
        problem.variables.add(lb=[soc_min[k]]*(T+1),ub=[soc_max[k]]*(T+1),names=s_bar[k])
        problem.variables.add(names=c[k])
        problem.variables.add(names=c_bar[k])
        problem.variables.add(names=d[k])
        problem.variables.add(names=d_bar[k])
        problem.variables.add(types=['B']*T,names=u[k])
        problem.variables.add(types=['B']*T,names=u_bar[k])
        problem.variables.add(types=['B']*T,names=v[k])
        problem.variables.add(types=['B']*T,names=v_bar[k])
        problem.variables.add(names=n[k])
        problem.variables.add(names=n_bar[k])
        problem.variables.add(names=p[k])
        problem.variables.add(names=p_bar[k])

    
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_"+str(k)+"_"+str(0)],val=[1])],senses=["E"],rhs=[soc_init[k]],names=["constraint_s_"+str(0)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bar_"+str(k)+"_"+str(0)],val=[1])],senses=["E"],rhs=[soc_init[k]])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_"+str(k)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[soc_max[k] for t in range(1,T+1)])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bar_"+ str(k)+"_"+str(t)],val=[1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[soc_max[k] for t in range(1,T+1)])

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_"+str(k)+"_"+str(t),"v_"+str(k)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[1]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bar_"+str(k)+"_"+str(t),"v_bar_"+str(k)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[1]*T)
        


        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_"+str(k)+"_"+str(t),"u_"+str(k)+"_"+str(t)],val=[1,-p_charge_max[k]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_bar_"+str(k)+"_"+str(t),"u_bar_"+str(k)+"_"+str(t)],val=[1,-p_charge_max[k]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_"+str(k)+"_"+str(t),"v_"+str(k)+"_"+str(t)],val=[1,-p_discharge_max[k]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_bar_"+str(k)+"_"+str(t),"v_bar_"+str(k)+"_"+str(t)],val=[1,-p_discharge_max[k]]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_"+str(k)+"_"+str(t),"c_"+str(k)+"_"+str(t)],val=[p_charge_min[k],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bar_"+str(k)+"_"+str(t),"c_bar_"+str(k)+"_"+str(t)],val=[p_charge_min[k],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_"+str(k)+"_"+str(t),"d_"+str(k)+"_"+str(t)],val=[p_discharge_min[k],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[0]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_bar_"+str(k)+"_"+str(t),"d_bar_"+str(k)+"_"+str(t)],val=[p_discharge_min[k],-1]) for t in range(1,T+1)],senses=["L"]*T,rhs=[]*T)

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_"+str(k)+"_"+str(t+1),"s_"+str(k)+"_"+str(t),"c_"+str(k)+"_"+str(t+1),"d_"+str(k)+"_"+str(t+1)],val=[1,-1,-time_mesh_to_hour,time_mesh_to_hour]) for t in range(T-1)],senses=["E"]*(T-1),rhs=[0]*(T-1))
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bar_"+str(k)+"_"+str(t+1),"s_bar_"+str(k)+"_"+str(t),"c_bar_"+str(k)+"_"+str(t+1),"d_bar_"+str(k)+"_"+str(t+1)],val=[1,-1,-time_mesh_to_hour,time_mesh_to_hour]) for t in range(T-1)],senses=["E"]*(T-1),rhs=[0]*(T-1))

        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["n_"+str(k)+"_"+str(t),"s_"+str(k)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["G"]*T,rhs=s_t_min[k])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["n_bar_"+str(k)+"_"+str(t),"s_bar_"+str(k)+"_"+str(t)],val=[1,1]) for t in range(1,T+1)],senses=["G"]*T,rhs=s_t_min[k])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["p_"+str(k)+"_"+str(t),"s_"+str(k)+"_"+str(t)],val=[1,-1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[-soc_final[k]]*T)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["p_bar_"+str(k)+"_"+str(t),"s_bar_"+str(k)+"_"+str(t)],val=[1,-1]) for t in range(1,T+1)],senses=["G"]*T,rhs=[-soc_final[k]]*T)
    t=1
    #print(len(["y_"+str(t)]+[x for k in range(N) for x in ["c_"+str(k)+"_"+str(t),"c_bar_"+str(k)+"_"+str(t),"d_"+str(k)+"_"+str(t),"d_bar_"+str(k)+"_"+str(t)]]))
    #print(len([1]+[x for _ in range(N) for x in [-1,1,1,-1]]))
    #print(len(["E"]*T))
    #print(len(data["announced_capacity"]["up"]))
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["y_"+str(t)]+[x for k in range(N) for x in ["c_"+str(k)+"_"+str(t),"c_bar_"+str(k)+"_"+str(t),"d_"+str(k)+"_"+str(t),"d_bar_"+str(k)+"_"+str(t)]],\
                                                              val=[1]+[x for _ in range(N) for x in [-1/N,1/N,1/N,-1/N]]) for t in range(1,T+1)],senses=["E"]*T,\
                                                              rhs=data["announced_capacity"]["up"][actuel_time:actuel_time+T])

        # Set the objective function
    problem.objective.set_sense(problem.objective.sense.minimize)
    for k in range(N):
        for t in range(1,T+1):
            problem.objective.set_quadratic_coefficients("n_"+str(k)+"_"+str(t),"n_"+str(k)+"_"+str(t),1/N)
            problem.objective.set_quadratic_coefficients("n_bar_"+str(k)+"_"+str(t),"n_bar_"+str(k)+"_"+str(t),1/N)
            problem.objective.set_quadratic_coefficients("p_"+str(k)+"_"+str(t),"p_"+str(k)+"_"+str(t),1/N)
            problem.objective.set_quadratic_coefficients("p_bar_"+str(k)+"_"+str(t),"p_bar_"+str(k)+"_"+str(t),1/N)

    for t in range(1,T+1):
        problem.objective.set_quadratic_coefficients("y_"+str(t),"y_"+str(t),1)
    #exprs_f = ["y_"+str(t) for t in range(1,T+1)]
    #val_f = [x for k in range(N) for x in [-2/N**2*data["announced_capacity"]["up"][k]]*2]
    exprs_electric_cost = [x for k in range(N) for t in range(1,T+1) for x in ["c_"+str(k)+"_"+str(t),"d_"+str(k)+"_"+str(t)]]
    val_electric_cost = [x for k in range(N) for t in range(1,T+1) for x in [1/N*time_mesh_to_hour*cost_electricity[t],-1/N*time_mesh_to_hour*cost_electricity[t]]]
    exprs_final_soc = [ "s_"+str(k)+"_"+str(T) for k in range(N)]
    val_final_soc = [1/N*penality_SOC_fin*cost_electricity[T]]*N

    problem.objective.set_linear(zip(exprs_electric_cost+exprs_final_soc,val_electric_cost+val_final_soc))

    problem.write("test_frontal.lp")
    problem.solve()
    print(problem.solution.get_objective_value())  
    #print(problem.solution.get_values())
    
Frontal(data, actuel_time, verbose=False)