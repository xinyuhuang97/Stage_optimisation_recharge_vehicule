import json
import os
import numpy.random as rd
from .one_ev_solver_fw import optimize_sub
from ...instance.dispatch_config import DispatchConfig, EVSE




rd.seed(1)


obj = []

def dispatch(dispatch_config: DispatchConfig, nb_it = 100):
    T = range(dispatch_config.optim_horizon)
    N = range(len(dispatch_config.evses))
    alpha_up = dispatch_config.penalties_fcr_up
    alpha_down = dispatch_config.penalties_fcr_down
    stats = {
        "objective" : []
    }

    def lance_curve_penalty(evse: EVSE, solution_dict):
        penalty = 0
        for trajectory in "baseline", "fcr_up", "fcr_down":
            for t in T:
                if t == 0:
                    continue
                penalty += min(solution_dict[f"soc_{trajectory}"][t] - evse.soc_lance_curve[t], 0) ** 2
        return penalty * dispatch_config.penalties_soc_fin

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


    for k in range(nb_it):
        omega = 2 / (k + 2)

        # compute slopes
        slope_fcr_up_k = [2 * alpha_up * (sum(service_up[n][t] for n in N) - demand_up[t]) for t in T]
        slope_fcr_down_k = [2 * alpha_down * (sum(service_down[n][t] for n in N) - demand_down[t]) for t in T]
        slope_fcr_up = [omega * slope_fcr_up_k[t] + (1 - omega) * slope_fcr_up[t] for t in T]
        slope_fcr_down = [omega * slope_fcr_down_k[t] + (1 - omega) * slope_fcr_down[t] for t in T]

        omega = 2 / (k + 2)

        print(f"\n iteration{k}")
        print(f"slopes up : {slope_fcr_up}")
        print(f"slopes down : {slope_fcr_down}")
        print(f'service up : {[sum(service_up[n][t] for n in N) for t in T]}')
        print(f'service down : {[sum(service_down[n][t] for n in N) for t in T]}')

        if True:
            f0 = sum([alpha_up * (sum(service_up[n][t] for n in N) - demand_up[t])**2 for t in T]) + \
            sum([alpha_down * (sum(service_down[n][t] for n in N) - demand_down[t])**2 for t in T])
            #print(f"iter {k} : f0 = {f0} \t \t {demand_up[2]} / {sum(service_up[n][2] for n in N)} \t {demand_down[2]} / {sum(service_down[n][2] for n in N) }")
            #print([sum(service_up[n][t] for n in N) for t in T])
            #print(demand_up)

        # combination of curent and previous solution
        for n, ev in enumerate(dispatch_config.evses):
            solution_dict = optimize_sub(ev, dispatch_config, slope_fcr_up, slope_fcr_down)
            for t in T:
                service_up[n][t] *= (1-omega)
                service_up[n][t] += omega * solution_dict["fcr_up"][t]

                service_down[n][t] *= (1-omega)
                service_down[n][t]  += omega *  solution_dict["fcr_down"][t]

                charge_baseline[n][t] *= (1-omega)
                charge_baseline[n][t]  += omega *  solution_dict["p_charge_baseline"][t]

                discharge_baseline[n][t] *= (1-omega)
                discharge_baseline[n][t]  += omega *  solution_dict["p_discharge_baseline"][t]

                charge_fcr_up[n][t] *= (1-omega)
                charge_fcr_up[n][t]  += omega *  solution_dict["p_charge_fcr_up"][t]

                discharge_fcr_up[n][t]  *= (1-omega)
                discharge_fcr_up[n][t]  += omega *  solution_dict["p_discharge_fcr_up"][t]

                charge_fcr_down[n][t] *= (1-omega)
                charge_fcr_down[n][t]  += omega *  solution_dict["p_charge_fcr_down"][t]

                discharge_fcr_down[n][t] *= (1-omega)
                discharge_fcr_down[n][t]  += omega *  solution_dict["p_discharge_fcr_down"][t]

            individual_penalties[n] = lance_curve_penalty(ev, solution_dict)

        cout_charge = sum(sum((charge_baseline[n][t] - discharge_baseline[n][t]) * dispatch_config.cost_of_electricity[t] * (dispatch_config.time_mesh / 60)
            for t in T) for n in N)
        obj_value = (sum((alpha_up * (sum(service_up[n][t] for n in N) - demand_up[t]) ** 2
                + alpha_down * (sum(service_down[n][t] for n in N) - demand_down[t]) ** 2
                ) for t in T)
            + sum(individual_penalties)
            + cout_charge)

        obj.append(obj_value)
        stats["objective"].append(obj[-1])

    #print(f'charge baseline : {charge_baseline[0]}')
    #print(f'charge fcr up : {charge_fcr_up[0]}')
    #print(f'charge fcr down : {charge_fcr_down[0]}')

    #print(f'discharge baseline : {discharge_baseline[0]}')
    #print(f'discharge fcr up : {discharge_fcr_up[0]}')
    #print(f'discharge fcr down : {discharge_fcr_down[0]}')

    print(f'service up : {[sum(service_up[n][t] for n in N) for t in T]}')
    print(f'service down : {[sum(service_down[n][t] for n in N) for t in T]}')



    print(f'service up mismatch : {[sum(service_up[n][t] for n in N) - dispatch_config.announced_capacity_up[t] for t in T]}')

    penalite_fcr_up = sum(alpha_up * (sum(service_up[n][t] for n in N) - dispatch_config.announced_capacity_up[t]) ** 2
        for t in T)
    penalite_fcr_down = sum(alpha_down * (sum(service_down[n][t] for n in N) - dispatch_config.announced_capacity_down[t]) ** 2
        for t in T)


    print(f"best sfw objective value : \t {min(obj)}")
    print(f"last sfw objective value : \t {obj[-1]}")
    print(f"  penalite lance curve : \t {sum(individual_penalties)}")
    print(f"  penalite FCR up : \t\t {penalite_fcr_up}")
    print(f"  penalite FCR down : \t\t {penalite_fcr_down}")
    print(f"  cout charge : \t\t {cout_charge}")
    return stats

    #with open(os.path.join(tmpfolder, f"final.json"), 'w') as outfile:
    #    json.dump(result_solver, outfile, indent = 3, sort_keys=True, default=str)
