import sys
sys.path.append('../data')
sys.path.append('/Applications/CPLEX_Studio221/cplex/python/3.7/x86-64_osx/cplex/_internal')
#import py37_cplex2210 as cplex
import cplex
import os
import sys
import numpy as np

def delta_optimize(x_k, x_bar_k, data ,actual_time,verbose=False):
    N=len(data["evses"])
    T=data["optim_horizon"]
    alpha=data["penalties"]["fcr_up"]
    soc_max = [data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)]
    y_t_up = [data["announced_capacity"]["up"][actual_time+t] for t in range(T)]
    penality_SOC_fin = data["penalties"]["SOC_fin"]


    G_bar_k = np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"],axis=0)/N
    G_k = np.sum(x_k["c_bl"] - x_k["d_bl"] - x_k["c_up"] + x_k["d_up"],axis=0)/N
    
    s_T_bar = x_bar_k["s_bl"][:,T-1]
    H_bar_k = np.sum([(x_bar_k['c_bl'][:,t]-x_bar_k['d_bl'][:,t]) * data["cost_of_electricity"][t]*data["time_mesh"]/60 for t in range(T)])/N \
                + np.sum([(soc_max[i]-s_T_bar[i]) * data["cost_of_electricity"][actual_time+T]*penality_SOC_fin for i in range(N)])/N
    s_T = x_k["s_bl"][:,T-1]
    H_k = np.sum([(x_k['c_bl'][:,t]-x_k['d_bl'][:,t]) * data["cost_of_electricity"][t]*data["time_mesh"]/60 for t in range(T)])/N \
                + np.sum([(soc_max[i]-s_T[i]) * data["cost_of_electricity"][actual_time+T]*penality_SOC_fin for i in range(N)])/N
    
    problem = cplex.Cplex()

    #problem.parameters.mip.tolerances.integrality.set(1e-9)
    if not verbose:
        problem.set_log_stream(None)
        problem.set_error_stream(None)
        problem.set_warning_stream(None)
        problem.set_results_stream(None)

    problem.objective.set_sense(problem.objective.sense.minimize)
    delta=["delta"]
    y=["y_" + str(t) for t in range(1, T+1)]
    y_diff=["y_diff_" + str(t) for t in range(1, T+1)]

    problem.variables.add(lb=[0], ub=[1], names=delta)
    problem.variables.add(lb=[-cplex.infinity]*(T),ub=[cplex.infinity]*(T),names=y)
    problem.variables.add(lb=[-cplex.infinity]*(T),ub=[cplex.infinity]*(T), names=y_diff)
    
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["y_"+str(t), "delta"], val=[1, -G_k[t-1]+G_bar_k[t-1]]) for t in range(1, T+1)], senses=["E"]*T, rhs=[G_bar_k[t] for t in range(T)])
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["y_diff_"+str(t), "y_"+str(t)], val=[1, -1]) for t in range(1, T+1)], senses=["E"]*T, rhs=[-y_t_up[t]/N for t in range(T)])

    for t in range(1,T+1):
        problem.objective.set_quadratic_coefficients("y_diff_"+str(t),"y_diff_"+str(t),alpha*2)
    expression_diff_H_k = ["delta"]
    val_diff_H_k = [H_k-H_bar_k]

    problem.objective.set_linear(zip(expression_diff_H_k,val_diff_H_k))

    problem.solve()
    return problem.solution.get_values(delta[0])

