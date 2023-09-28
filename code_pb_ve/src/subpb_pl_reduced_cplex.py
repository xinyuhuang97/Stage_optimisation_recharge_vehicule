import sys
sys.path.append('../data')
sys.path.append('/Applications/CPLEX_Studio221/cplex/python/3.7/x86-64_osx/cplex/_internal')
#import py37_cplex2210 as cplex
import cplex
import os
import sys
import numpy as np

def resolution_subpb(data, lambda_1_k, lambda_2_k, i, x_bar_k,verbose=False,):
    # Create a new LP problem
    problem = cplex.Cplex()

    if not verbose:
        problem.set_log_stream(None)
        problem.set_error_stream(None)
        problem.set_warning_stream(None)
        problem.set_results_stream(None)

    # Prepare data
    T=data["optim_horizon"]

    soc_init=data["evses"][i]["SOC_init"]*data["evses"][i]["capacity"]
    soc_max=data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"]
    soc_min=data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"]

    p_charge_max=data["evses"][i]["p_charge_max"]
    p_charge_min=data["evses"][i]["p_charge_min"]
    p_discharge_max=data["evses"][i]["p_discharge_max"]
    p_discharge_min=data["evses"][i]["p_discharge_min"]

    time_mesh_to_hour = data["time_mesh"]/60
    s_final=data["evses"][i]["SOC_final"]*data["evses"][i]["capacity"]
    s_t_min=[max(data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"],(s_final-p_charge_max*(T-t)*time_mesh_to_hour) ) for t in range(1,T+1)]
    #print(s_t_min)
    cost_electricity=data["cost_of_electricity"]
    penality_SOC_fin=data["penalties"]["SOC_fin"]

    beta_min=data["penalties"]["beta_min"]
    beta_max=data["penalties"]["beta_max"]
    alpha=data["penalties"]["fcr_up"]

    # Create variables
    s_bl = ["s_bl_" + str(t) for t in range(T+1)]
    c_bl = ["c_bl_" + str(t) for t in range(1,T+1)]
    d_bl = ["d_bl_" + str(t) for t in range(1,T+1)]
    u_bl = ["u_bl_" + str(t) for t in range(1,T+1)]
    v_bl = ["v_bl_" + str(t) for t in range(1,T+1)]
    u_up = ["u_up_" + str(t) for t in range(1,T+1)]
    v_up = ["v_up_" + str(t) for t in range(1,T+1)]

    y = ["y_" + str(t) for t in range(1,T+1)]

    z_1 = ["z_1_" + str(t) for t in range(1,T+1)]
    z_2 = ["z_2_" + str(t) for t in range(1,T+1)]

    problem.variables.add(names=s_bl,lb=[-cplex.infinity]*(T+1),ub=[cplex.infinity]*(T+1))
    problem.variables.add(names=c_bl,lb=[-cplex.infinity]*(T),ub=[cplex.infinity]*(T))
    problem.variables.add(names=d_bl,lb=[-cplex.infinity]*(T),ub=[cplex.infinity]*(T))
    problem.variables.add(types=['B']*T, names=u_bl)
    problem.variables.add(types=['B']*T, names=u_up)
    problem.variables.add(types=['B']*T, names=v_bl)
    problem.variables.add(types=['B']*T, names=v_up)
    problem.variables.add(names=y,lb=[-cplex.infinity]*(T),ub=[cplex.infinity]*(T))
    problem.variables.add(names=z_1,lb=[-cplex.infinity]*(T),ub=[cplex.infinity]*(T))
    problem.variables.add(names=z_2,lb=[-cplex.infinity]*(T),ub=[cplex.infinity]*(T))

    # Add constraints to the problem

    # constraints 1 and 2 :initial state of charge equal to the initial state of charge of the EV
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_0"], val=[1])], senses=["E"], rhs=[soc_init],names=["SOC_init bl"])

    # constraints 3 and 4 : the state of charge should be between 0 and the maximum capacity of the battery
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(t)], val=[1]) for t in range(1, T+1)],senses=["L"]*T,rhs=[soc_max]*T,names=["SOC_max bl"+str(t) for t in range(1, T+1)])

    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(t)], val=[1]) for t in range(1, T+1)],senses=["G"]*T,rhs=[soc_min]*T,names=["SOC_min bl"+str(t) for t in range(1, T+1)])


    # constraints 5 and 6 :only one of the charging or discharging variables can be equal to 1
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bl_"+str(t),"v_bl_"+str(t)], val=[1, 1]) for t in range(1,T+1)], senses=["L"]*T, rhs=[1]*T,names=["charge or discharge bl"+str(t) for t in range(1,T+1)])
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_up_"+str(t),"v_up_"+str(t)], val=[1, 1]) for t in range(1,T+1)], senses=["L"]*T, rhs=[1]*T,names=["charge or discharge up"+str(t) for t in range(1,T+1)])

    # constraints 7 and 9 : if the vehicle is charging, the charging power should be smaller than the maximum charging power
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_bl_"+str(t),"u_bl_"+str(t)],val=[1,-p_charge_max]) for t in range(1,T+1)], senses=["L"]*T, rhs=[0]*T,names=["charge upper bound bl"+str(t) for t in range(1,T+1)])
    #problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_up_"+str(t),"u_up_"+str(t)],val=[1,-p_charge_max]) for t in range(1,T+1)], senses=["L"]*T, rhs=[0]*T,names=["charge upper bound up"+str(t) for t in range(1,T+1)])

    # constraints 8 and 10 : if the vehicle is discharging, the discharging power should be smaller than the maximum discharging po wer
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_bl_"+str(t),"v_bl_"+str(t)],val=[1,-p_discharge_max]) for t in range(1,T+1)], senses=["L"]*T, rhs=[0]*T,names=["discharge upper bound bl"+str(t) for t in range(1,T+1)])
    #problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_up_"+str(t),"v_up_"+str(t)],val=[1,-p_discharge_max]) for t in range(1,T+1)], senses=["L"]*T, rhs=[0]*T,names=["discharge upper bound up"+str(t) for t in range(1,T+1)])

    # constraints 11 and 12 : the charge power should be greater than the minimum charge power
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bl_"+str(t),"c_bl_"+str(t)],val=[p_charge_min,-1]) for t in range(1,T+1)], senses=["L"]*T, rhs=[0]*T,names=["charge lower bound bl"+str(t) for t in range(1,T+1)])
    #problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_up_"+str(t),"c_up_"+str(t)],val=[p_charge_min,-1]) for t in range(1,T+1)], senses=["L"]*T, rhs=[0]*T,names=["charge lower bound up"+str(t) for t in range(1,T+1)])

    # constraints 13 and 14 : the discharge power should be greater than the minimum discharge po wer
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_bl_"+str(t),"d_bl_"+str(t)],val=[p_discharge_min, -1]) for t in range(1,T+1)], senses=["L"]*T, rhs=[0]*T,names=["discharge lower bound bl"+str(t) for t in range(1,T+1)])
    #problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_up_"+str(t),"d_up_"+str(t)],val=[p_discharge_min, -1]) for t in range(1,T+1)], senses=["L"]*T, rhs=[0]*T,names=["discharge lower bound up"+str(t) for t in range(1,T+1)])    
    
    # constraints 15 and 16 : the diffenrence in charge level between two successive time steps should be equal to the difference between the production and consumption

    
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(t+1),"s_bl_"+str(t),"c_bl_"+str(t+1),"d_bl_"+str(t+1)],\
                                                              val=[1, -1, -time_mesh_to_hour, time_mesh_to_hour]) for t in range(T)], senses=["E"]*(T), rhs=[0]*(T),names=["Production balance bl"+str(t) for t in range(T)])
    #problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(t+1),"s_bl_"+str(t),"c_up_"+str(t+1),"d_up_"+str(t+1)],\
    #                                                          val=[1, -1, -time_mesh_to_hour, time_mesh_to_hour]) for t in range(T)], senses=["E"]*(T), rhs=[0]*(T),names=["Production balance up"+str(t) for t in range(T)])
    
    # constraints 18-21 : extract the positif part/ negatif part of the differce in hi(xi) function

    #print("here",[-x_bar_k["c_bl"][t]+x_bar_k["d_bl"][t]+x_bar_k["c_up"][t]-x_bar_k["d_up"] for t in range(1,T+1)])
    #problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["y_"+str(t), "c_bl_"+str(t),"d_bl_"+str(t),"c_up_"+str(t),"d_up_"+str(t)],\
    #                                                          val=[1, -1, 1, 1, -1]) for t  in range(1,T+1)], senses=["E"]*(T), rhs=[0]*T)
                                                                #val=[1, -1, 1, 1, -1]) for t  in range(1,T+1)], senses=["E"]*(T), rhs=[-x_bar_k["c_bl"][i][t]+x_bar_k["d_bl"][i][t]+x_bar_k["c_up"][i][t]-x_bar_k["d_up"][i][t] for t in range(T)])
                                                                
    # constraints pour limiter le chargement minimal a chaque pas de temps pour que le vehicule puisse atteindre la charge finale
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(t)],val=[1]) for t in range(1,T+1)], senses=["G"]*(T), rhs=s_t_min)
    #problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(t)],val=[1]) for t in range(1,T+1)], senses=["G"]*(T), rhs=s_t_min)
    
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["z_1_"+str(t), "u_up_"+str(t), "v_up_"+str(t)],val=[1, p_charge_max, -p_discharge_min]) for t in range(1,T+1)], senses=["G"]*(T), rhs=[0]*T)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["z_2_"+str(t), "u_up_"+str(t), "v_up_"+str(t)],val=[1, -p_charge_min, p_discharge_max]) for t in range(1,T+1)], senses=["G"]*(T), rhs=[0]*T)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["z_1_"+str(t), "s_bl_"+str(t-1)], val=[time_mesh_to_hour, -1]) for t in range(1,T+1)], senses=["G"]*(T), rhs=[-soc_max]*T)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["z_2_"+str(t), "s_bl_"+str(t-1)], val=[time_mesh_to_hour, 1]) for t in range(1,T+1)], senses=["G"]*(T), rhs=s_t_min)

    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["y_"+str(t), "c_bl_"+str(t),"d_bl_"+str(t), "z_1_"+str(t), "z_2_"+str(t)], \
                                                              val=[-1, lambda_1_k[t-1]+lambda_2_k[t-1],-lambda_1_k[t-1]-lambda_2_k[t-1],lambda_1_k[t-1],-lambda_2_k[t-1]]) for t in range(1,T+1)], senses=["E"]*(T), rhs=[0]*T)

    
    exprs_scalar_prod=["y_" + str(t) for t in range(1,T+1)]
    val_scalar_prod=[1 for t in range(1,T+1)]

    exprs_electric_cost=[x for t in range(1,T+1) for x in ["c_bl_"+str(t),"d_bl_"+str(t)]]
    val_electric_cost=[x for t in range(1,T+1) for x in [cost_electricity[t-1]*time_mesh_to_hour,-cost_electricity[t-1]*time_mesh_to_hour] ]

    exprs_final_soc=["s_bl_"+str(T)]
    val_final_soc=[-penality_SOC_fin*cost_electricity[T]]
    list_value=[]
    list_value.extend(val_electric_cost)#=+b1#+val_scalaire_prod
    list_value.extend(val_final_soc)
    list_value.extend(val_scalar_prod)
 
    problem.objective.set_linear(zip(exprs_electric_cost+exprs_final_soc+exprs_scalar_prod,list_value))
    
    # Write the problem as an LP file
    #problem.write("sub_pbs/sub_pb"+str(i)+"_"+str(k)+".lp")
    problem.write("sub_pbs_test.lp")
    # Solve the problemx``
    
    problem.solve()
    #print("y", np.sum([problem.solution.get_values(y[t])**2*lambda_k[t] for t in range(T)]) )
    #print("soc_fin",np.sum( [problem.solution.get_values(s_bl[T])*(-penality_SOC_fin*cost_electricity[T])] ))
    return problem.solution.get_values(c_bl), problem.solution.get_values(d_bl),\
        problem.solution.get_values(u_bl), problem.solution.get_values(u_up), problem.solution.get_values(v_bl), problem.solution.get_values(v_up),\
        problem.solution.get_values(s_bl[1:]),problem.solution.get_values(y), problem.solution.get_values(z_1), problem.solution.get_values(z_2)