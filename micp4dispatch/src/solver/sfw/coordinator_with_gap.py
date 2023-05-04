import json
import os
import copy
import numpy.random as rd
from .one_ev_solver import optimize_sub
from ...instance.dispatch_config import DispatchConfig, EVSE


rd.seed(1)


def dispatch(dispatch_config: DispatchConfig, nb_it = 100):
    T = range(dispatch_config.optim_horizon)
    N = range(len(dispatch_config.evses))
    alpha_up = dispatch_config.penalties_fcr_up
    alpha_down = dispatch_config.penalties_fcr_down
    nb_ev = len(dispatch_config.evses)

    stats = {
        "objective" : [],
        "rel_gap" : [],
        "primal_dual_rel_gap" : []
    }

    def lance_curve_penalty(evse: EVSE, solution_dict):
        penalty = 0
        for trajectory in "baseline", "fcr_up", "fcr_down":
            for t in T:
                if t == 0:
                    continue
                penalty += min(solution_dict[f"soc_{trajectory}"][t] - evse.soc_lance_curve[t], 0) ** 2
        penalty *= dispatch_config.penalties_soc_fin
        penalty += solution_dict[f"soc_baseline"][-1] * dispatch_config.penalties_soc_fin
        return penalty
    
    demand_up = [0] + dispatch_config.announced_capacity_up[1:dispatch_config.optim_horizon]
    demand_down = [0] + dispatch_config.announced_capacity_down[1:dispatch_config.optim_horizon]

    charge_baseline = [[0 for t in T] for n in N]       # for each ev, charging power at baseline
    discharge_baseline = [[0 for t in T] for n in N]    # for each ev, discharging power at baseline
    charge_fcr_up = [[0 for t in T] for n in N]          # for each ev, charging power at FCR up service
    discharge_fcr_up = [[0 for t in T] for n in N]        # for each ev, discharging power at FCR up service
    charge_fcr_down = [[0 for t in T] for n in N]        # for each ev, charging power at FCR down service
    discharge_fcr_down = [[0 for t in T] for n in N]     # for each ev, discharging power at FCR down service

    service_up = [[0 for t in T] for n in N]           # individual FCR
    service_down = [[0 for t in T] for n in N]         # individual FCR

    individual_penalties = [0 for n in N]
    slope_fcr_up = [0 for t in T]
    slope_fcr_down = [0 for t in T]

    ### ITERATIONS ###

    obj = [] # list of succesive objective value (without repetition)
    objective_history = [] # list of succesive objective value (with repetition)
    for k in range(nb_it):
        
        # compute slopes
        slope_fcr_up = [2 * alpha_up * (sum(service_up[n][t] for n in N) - demand_up[t]) / nb_ev for t in T]

        omega = 2 / (k + 2)

        print(f"\n iteration{k}")
        print(f"slopes up : {slope_fcr_up}")
        print(f"slopes down : {slope_fcr_down}")
        print(f'av service up : {[sum(service_up[n][t]/ nb_ev for n in N) for t in T]}')
        print(f'av service down : {[sum(service_down[n][t]/ nb_ev for n in N) for t in T]}')


        # combination of curent and previous solution

        curr_solution = {} # denoted \bar{x}_k

        if len(obj) < 2 or obj[-1] < obj[-2]:
            for n, ev in enumerate(dispatch_config.evses):
                curr_solution[n] = optimize_sub(ev, dispatch_config, slope_fcr_up, slope_fcr_down)
                curr_solution[n]["individual penalties"] = lance_curve_penalty(ev, curr_solution[n])
        curr_cout_charge = sum(sum((curr_solution[n]["p_charge_baseline"][t] - curr_solution[n]["p_discharge_baseline"][t]) * dispatch_config.cost_of_electricity[t] * (dispatch_config.time_mesh / 60)
                for t in T) for n in N)

        solution_dict = []      # denoted \hat{x}_k
        objectif_value = []     #value of solution_dict[k]

        for i in range(1):
            solution_dict.append({})
            for n, ev in enumerate(dispatch_config.evses):
                if rd.random() < omega :
                    solution_dict[i][n] = copy.deepcopy(curr_solution[n])
                else :
                    solution_dict[i][n] = copy.deepcopy(last_solution[n])

            new_cout_charge = sum(sum((solution_dict[i][n] ["p_charge_baseline"][t] - solution_dict[i][n] ["p_discharge_baseline"][t]) * dispatch_config.cost_of_electricity[t] * (dispatch_config.time_mesh / 60)
                for t in T) for n in N)
            
            objectif_value.append((sum((alpha_up * (sum(solution_dict[i][n] ["fcr_up"][t] for n in N) - demand_up[t]) ** 2
                    + alpha_down * (sum(solution_dict[i][n] ["fcr_down"][t] for n in N) - demand_down[t]) ** 2
                    ) for t in T))  / nb_ev ** 2 
                + (sum(solution_dict[i][n]["individual penalties"] for n in N)
                + new_cout_charge) / nb_ev)

        i_opt = objectif_value.index(min(objectif_value))

        if k > 0:        
            pdgap = (2*(sum(((alpha_up * (sum(last_solution[n]["fcr_up"][t] for n in N) - demand_up[t]) * sum(last_solution[n]["fcr_up"][t] - curr_solution[n]["fcr_up"][t] for n in N)
                        + alpha_down * (sum(last_solution[n]["fcr_down"][t] for n in N) - demand_down[t]) * sum(last_solution[n]["fcr_down"][t] - curr_solution[n]["fcr_down"][t] for n in N)
                        ) for t in T))  / nb_ev ** 2
                    + (sum(last_solution[n]["individual penalties"] for n in N) - sum(curr_solution[n]["individual penalties"] for n in N)
                    + last_cout_charge - curr_cout_charge) / nb_ev))
        
            if k == 1:
                pdgap_1 = pdgap
            print(f" rel primal-dual gap : {pdgap / pdgap_1}")

        if k == 0 or objectif_value[i_opt] < obj[-1]:
            obj.append(objectif_value[i_opt])
            print(f"F = {objectif_value[i_opt]}")

            for n in solution_dict[i_opt].keys():
                service_up[n] = solution_dict[i_opt][n] ["fcr_up"]
                service_down[n] = solution_dict[i_opt][n] ["fcr_down"]
                charge_baseline[n] = solution_dict[i_opt][n] ["p_charge_baseline"]
                discharge_baseline[n] = solution_dict[i_opt][n] ["p_discharge_baseline"]
                charge_fcr_up[n] = solution_dict[i_opt][n] ["p_charge_fcr_up"]
                discharge_fcr_up[n] = solution_dict[i_opt][n] ["p_discharge_fcr_up"]
                charge_fcr_down[n] = solution_dict[i_opt][n] ["p_charge_fcr_down"]
                discharge_fcr_down[n] = solution_dict[i_opt][n] ["p_discharge_fcr_down"]
                individual_penalties[n] = solution_dict[i_opt][n] ["individual penalties"]
            
            f0 = sum([alpha_up * (sum(service_up[n][t] for n in N) - demand_up[t])**2 for t in T]) / nb_ev**2 + \
            sum([alpha_down * (sum(service_down[n][t] for n in N) - demand_down[t])**2 for t in T]) / nb_ev**2
            print(f"f0 = {f0}")

            last_solution = solution_dict[i_opt] #denoted x_k
            last_cout_charge = sum(sum((solution_dict[i_opt][n] ["p_charge_baseline"][t] - solution_dict[i_opt][n] ["p_discharge_baseline"][t]) * dispatch_config.cost_of_electricity[t] * (dispatch_config.time_mesh / 60)
                for t in T) for n in N)

        objective_history.append(obj[-1])


        if k > 0:
            if dispatch_config.frontal_opt is not None :
                rel_gap = (obj[-1] - dispatch_config.frontal_opt) / (obj[0] - dispatch_config.frontal_opt)
                print(f" rel gap = {rel_gap}")
                stats["rel_gap"].append(rel_gap)
            stats["objective"].append(obj[-1])
            stats["primal_dual_rel_gap"].append(pdgap / pdgap_1)


    print(f'av charge baseline : {[sum(charge_baseline[n][t] for n in N)/len(N) for t in T] }')
    print(f'av charge fcr up : {[sum(charge_fcr_up[n][t] for n in N)/len(N) for t in T] }')
    print(f'av charge fcr down : {[sum(charge_fcr_down[n][t] for n in N)/len(N) for t in T] }')
    print(f'av discharge baseline : {[sum(discharge_baseline[n][t] for n in N)/len(N) for t in T] }')
    print(f'av discharge fcr up : {[sum(discharge_fcr_up[n][t] for n in N)/len(N) for t in T] }')
    print(f'av discharge fcr down : {[sum(discharge_fcr_down[n][t] for n in N)/len(N) for t in T] }')

    print(f'av service up : {[sum(service_up[n][t]/ nb_ev for n in N) for t in T]}')
    print(f'av service down : {[sum(service_down[n][t]/ nb_ev for n in N) for t in T]}')

    print(f'service up mismatch : {[(sum(service_up[n][t] for n in N) - demand_up[t])/ nb_ev for t in T]}')

    penalite_fcr_up = sum(alpha_up * (sum(service_up[n][t] for n in N) - demand_up[t]) ** 2
        for t in T)
    penalite_fcr_down = sum(alpha_down * (sum(service_down[n][t] for n in N) - demand_down[t]) ** 2
        for t in T)
    #cout_charge = sum(sum((charge_baseline[t][n] - discharge_baseline[t][n]) * dispatch_config.cost_of_electricity[t] * (dispatch_config.time_mesh / 60)
    #        for t in T) for n in N)

    print(f"best sfw objective value : \t {min(obj)} \t from it {objective_history.index(min(obj))} ")
    print(f"last sfw objective value : \t {obj[-1]}")
    print(f"  av penalite lance curve : \t {sum(individual_penalties)/ nb_ev}")
    print(f"  av penalite FCR up : \t\t {penalite_fcr_up/ (nb_ev **2)}")
    print(f"  av penalite FCR down : \t\t {penalite_fcr_down/ (nb_ev ** 2)}")
    print(f"  av cout charge : \t\t {last_cout_charge / nb_ev}")

    return stats
