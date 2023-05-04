import sys
sys.path.append('../data')
sys.path.append('/Applications/CPLEX_Studio221/cplex/python/3.7/x86-64_osx/cplex/_internal')
#import py37_cplex2210 as cplex
import cplex
import os
import sys

def resolution_subpb(data, lambda_k,x_bar_k,i, verbose=False):
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

    soc_init=data["evses"][i]["SOC_init"]*data["evses"][i]["capacity"]
    soc_max=data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"]
    soc_min=data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"]
    p_charge_max=data["evses"][i]["p_charge_max"]
    p_charge_min=data["evses"][i]["p_charge_min"]
    p_discharge_max=data["evses"][i]["p_discharge_max"]
    p_discharge_min=data["evses"][i]["p_discharge_min"]

    time_mesh_to_hour = data["time_mesh"]/60
    s_final=data["evses"][i]["SOC_final"]*data["evses"][i]["capacity"]
    s_t_min=[max(data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"],s_final-p_discharge_max*(T-t) ) for t in range(1,T+1)]
    s_max=[-data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for t in range(1,T+1)]
    cost_electricity=data["cost_of_electricity"]
    penality_SOC_fin=data["penalties"]["SOC_fin"]

    # Create variables
    s = ["s_" + str(i) for i in range(T+1)]
    c = ["c_" + str(i) for i in range(1,T+1)]
    d = ["d_" + str(i) for i in range(1,T+1)]
    u = ["u_" + str(i) for i in range(1,T+1)]
    v = ["v_" + str(i) for i in range(1,T+1)]
    n = ["n_" + str(i) for i in range(1,T+1)]
    p = ["p_" + str(i) for i in range(1,T+1)]

    s_bar = ["s_bar_" + str(i) for i in range(T+1)] 
    c_bar = ["c_bar_" + str(i) for i in range(1,T+1)]
    d_bar = ["d_bar_" + str(i) for i in range(1,T+1)]
    u_bar = ["u_bar_" + str(i) for i in range(1,T+1)]
    v_bar = ["v_bar_" + str(i) for i in range(1,T+1)]
    n_bar = ["n_bar_" + str(i) for i in range(1,T+1)]
    p_bar = ["p_bar_" + str(i) for i in range(1,T+1)]

    problem.variables.add(lb=[soc_min]*(T+1), ub=[soc_max]*(T+1), names=s)
    problem.variables.add(lb=[soc_min]*(T+1), ub=[soc_max]*(T+1), names=s_bar)
    """delete lb ub """
    problem.variables.add(lb=[0]*T, ub=[p_charge_max]*T, names=c)
    problem.variables.add(lb=[0]*T, ub=[p_charge_max]*T, names=c_bar)
    problem.variables.add(lb=[0]*T, ub=[p_discharge_max]*T, names=d)
    problem.variables.add(lb=[0]*T, ub=[p_discharge_max]*T, names=d_bar)
    problem.variables.add(types=['B']*T, names=u)
    problem.variables.add(types=['B']*T, names=u_bar)
    problem.variables.add(types=['B']*T, names=v)
    problem.variables.add(types=['B']*T, names=v_bar)
    problem.variables.add(names=n)
    problem.variables.add(names=n_bar)
    problem.variables.add(names=p)
    problem.variables.add(names=p_bar)
    # Add constraints to the problem

    # constraints 1 and 2 :initial state of charge equal to the initial state of charge of the EV
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_0"], val=[1])], senses=["E"], rhs=[soc_init])
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bar_0"], val=[1])], senses=["E"], rhs=[soc_init])

    # already added in the definition of variables
    """
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_"+str(i)], val=[1]) for i in range(1, T+1)],senses=["L"]*T,rhs=[soc_max]*T)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bar_"+str(i)], val=[1]) for i in range(1, T+1)], senses=["L"]*T, rhs=[soc_max]*T)
    """

    # constraints 5 and 6 :only one of the charging or discharging variables can be equal to 1
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_"+str(i),"v_"+str(i)], val=[1, 1]) for i in range(1,T+1)], senses=["L"]*T, rhs=[1]*T)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bar_"+str(i),"v_bar_"+str(i)], val=[1, 1]) for i in range(1,T+1)], senses=["L"]*T, rhs=[1]*T)

    # constraints 7 and 9 : if the vehicle is charging, the charging power should be smaller than the maximum charging power
    
    #print(data["evses"][i]["p_charge_max"])
    
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_"+str(i), "u_"+str(i)],val=[1, -p_charge_max]) for i in range(1,T+1)], senses=["L"]*T, rhs=[0]*T)
    
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_bar_"+str(i),"u_bar_"+str(i)],val=[1,-p_charge_max]) for i in range(1,T+1)], senses=["L"]*T, rhs=[0]*T)

    # constraints 8 and 10 : if the vehicle is discharging, the discharging power should be smaller than the maximum discharging power
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_"+str(i),"v_"+str(i)],val=[1,-p_discharge_max]) for i in range(1,T+1)], senses=["L"]*T, rhs=[0]*T)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_bar_"+str(i), "v_bar_"+str(i)],val=[1,-p_discharge_max]) for i in range(1,T+1)], senses=["L"]*T, rhs=[0]*T)

    # constraints 11 and 12 : the charge power should be greater than the minimum charge power
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_"+str(i),"c_"+str(i)],val=[p_charge_min,-1]) for i in range(1,T+1)], senses=["L"]*T, rhs=[0]*T)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bar_"+str(i),"c_bar_"+str(i)],val=[p_charge_min, -1]) for i in range(1,T+1)], senses=["L"]*T, rhs=[0]*T)

    # constraints 13 and 14 : the discharge power should be greater than the minimum discharge power
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_"+str(i),"d_"+str(i)],val=[p_discharge_min, -1]) for i in range(1,T+1)], senses=["L"]*T, rhs=[0]*T)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_bar_"+str(i),"d_bar_"+str(i)],val=[p_discharge_min, -1]) for i in range(1,T+1)], senses=["L"]*T, rhs=[0]*T)    
    
    # constraints 15 and 16 : the diffenrence in charge level between two successive time steps should be equal to the difference between the production and consumption

    
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_"+str(i+1),"s_"+str(i),"c_"+str(i+1),"d_"+str(i+1)],\
                                                              val=[1, -1, -time_mesh_to_hour, time_mesh_to_hour]) for i in range(T)], senses=["E"]*(T), rhs=[0]*(T))
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bar_"+str(i+1),"s_bar_"+str(i),"c_bar_"+str(i+1),"d_bar_"+str(i+1)],\
                                                              val=[1, -1, -time_mesh_to_hour, time_mesh_to_hour]) for i in range(T)], senses=["E"]*(T), rhs=[0]*(T))
    
    # constraints 18-21 : extract the positif part/negatif part of the differce in hi(xi) function

    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["n_"+str(i),"s_"+str(i)],val=[1,1]) for i in range(1,T+1)], senses=["G"]*T, rhs=s_t_min)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["n_bar_"+str(i),"s_bar_"+str(i)],val=[1,1]) for i in range(1,T+1)], senses=["G"]*T, rhs=s_t_min)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["p_"+str(i),"s_"+str(i)],val=[1,-1]) for i in range(1,T+1)], senses=["G"]*T, rhs=s_max)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["p_bar_"+str(i),"s_bar_"+str(i)],val=[1,-1]) for i in range(1,T+1)], senses=["G"]*T, rhs=s_max)


    # Set objective function
    # expression of (cit -dit)-(c_bar_it -d_bar_it) where cit and dit are express otherwise to avoid the use of dupolicated variables
    gi_exprs=[x for t in range(1,T+1) for x in ["c_bar_"+str(t), "d_bar_"+str(t)]]
    #gi_exprs_T=[x for x in ["c_bar_"+str(T), "d_bar_"+str(T)]]
    #gj_exprs_T_val=[-lambda_k[T-1],lambda_k[T-1]]
    gi_val=[x for t in range(1,T+1) for x in [-lambda_k[t-1],lambda_k[t-1]]]

    for i in range(1,T+1):
        problem.objective.set_quadratic_coefficients("n_"+str(i),"n_"+str(i),1)
        problem.objective.set_quadratic_coefficients("p_"+str(i),"p_"+str(i),1)
        problem.objective.set_quadratic_coefficients("n_bar_"+str(i),"n_bar_"+str(i),1)
        problem.objective.set_quadratic_coefficients("p_bar_"+str(i),"p_bar_"+str(i),1)
    
    hi_linear_exprs=[x for t in range(1,T+1) for x in ["c_"+str(t),"d_"+str(t)]]
    hi_linear_val=[x for t in range(1,T+1) for x in [cost_electricity[t-1]*time_mesh_to_hour+lambda_k[t-1],-cost_electricity[t-1]*time_mesh_to_hour]-lambda_k[t-1] ]
    penality_exprs=["s_"+str(T)]
    penality_val=[-penality_SOC_fin*cost_electricity[T]]
    #print("test",-pernality_SOC_fin*cost_electricity[T])

    exprs=gi_exprs+hi_linear_exprs+penality_exprs
    #val=gi_val+gj_exprs_T_val+hi_linear_val+penality_val
    val=gi_val+hi_linear_val+penality_val
    problem.objective.set_linear(zip(exprs,val))
    
    # Write the problem as an LP file
    problem.write("test.lp")

    # Solve the problemx``
    problem.solve()

    # Print solution
    """
    print("c = ", problem.solution.get_values(c))
    print("c_bar = ", problem.solution.get_values(c_bar))
    print("d = ", problem.solution.get_values(d))
    print("d_bar = ", problem.solution.get_values(d_bar))"""
    # optimal value
    #print("Optimal value = ", problem.solution.get_objective_value())
    return problem.solution.get_values(c), problem.solution.get_values(d), problem.solution.get_values(c_bar), problem.solution.get_values(d_bar)#, problem.solution.get_values(s), problem.solution.get_values(s_bar)