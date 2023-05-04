from docplex.mp.model import Model
from ...instance.dispatch_config import DispatchConfig, EVSE

import os, json

from datetime import datetime

def optimize_sub(evse: EVSE, dispatch_config: DispatchConfig, fcr_up_price: list, fcr_down_price: list):
    T = dispatch_config.optim_horizon
    m = Model(name='subpb')

    activation = {}
    power = {}
    soc = {}
    soc_min_deviation = {}
    fcr_up = m.continuous_var_list(T, name = f"fcr_up")
    fcr_down = m.continuous_var_list(T, lb = 0, name = f"fcr_down")

    for trajectory in 'baseline', 'fcr_up', 'fcr_down':
        activation[trajectory] = {}
        power[trajectory] = {}
        soc[trajectory] = m.continuous_var_list(T, lb = 0,
            ub = evse.soc_max, name = f"soc_{trajectory}")
        soc_min_deviation[trajectory] = m.continuous_var_list(T, lb = 0,
            name = f"soc_min_deviation_{trajectory}")

        for sense in 'charge', 'discharge':
            activation[trajectory][sense] = m.binary_var_list(T, name = f"activation_{trajectory}_{sense}")
            power[trajectory][sense] = m.continuous_var_list(T, lb = 0, name = f"power_{trajectory}_{sense}")

        m.add_constraints([activation[trajectory]['charge'][t] +
            activation[trajectory]['discharge'][t] <= 1 for t in range(1, T)],
            [f"cstr_icds_{trajectory}_{t}" for t in range(1, T)])

        m.add_constraints([power[trajectory]['charge'][t] <=
            evse.p_charge_max * activation[trajectory]['charge'][t] for t in range(1, T)],
            [f"cstr_chrg_max_{trajectory}_{t}" for t in range(1, T)])

        m.add_constraints([power[trajectory]['discharge'][t] <=
            evse.p_charge_max * activation[trajectory]['discharge'][t] for t in range(1, T)],
            [f"cstr_dischrg_max_{trajectory}_{t}" for t in range(1, T)])

        m.add_constraint(soc[trajectory][0] == evse.soc_init,
            f"cstr_soc_init_{trajectory}")

        m.add_constraints([soc[trajectory][t] - soc['baseline'][t - 1]
            == (power[trajectory]['charge'][t] - power[trajectory]['discharge'][t])
            * (dispatch_config.time_mesh / 60) / evse.capacity for t in range(1, T)],
            names = [f"cstr_soc_evol_{trajectory}_{t}" for t in range(1, T)])

        m.add_constraints([soc_min_deviation[trajectory][t]
            >= evse.soc_lance_curve[t] - soc[trajectory][t] for t in range(1, T)],
            names = [f"cstr_soc_deviation_{trajectory}_{t}" for t in range(1, T)])

    m.add_constraints([power['baseline']['charge'][t]
        >= evse.p_charge_min * activation['baseline']['charge'][t] for t in range(1, T)],
        names = [f"cstr_chrg_min_baseline_{t}" for t in range(1, T)])

    m.add_constraints([power['baseline']['discharge'][t]
        >= evse.p_discharge_min * activation['baseline']['discharge'][t] for t in range(1, T)],
        names = [f"cstr_dischrg_min_baseline_{t}" for t in range(1, T)])

    m.add_constraints([power['baseline']['charge'][t] - power['baseline']['discharge'][t]
        - power['fcr_up']['charge'][t] + power['fcr_up']['discharge'][t]
        == fcr_up[t] for t in range(1, T)], names = [f"cstr_fcr_up_{t}" for t in range(1, T)])

    m.add_constraints([power['fcr_down']['charge'][t] - power['fcr_down']['discharge'][t]
        - power['baseline']['charge'][t] + power['baseline']['discharge'][t]
         == fcr_down[t] for t in range(1, T)], names = [f"cstr_fcr_down_{t}" for t in range(1, T)])


    m.set_objective("min", sum(fcr_up_price[t] * fcr_up[t]
            + fcr_down_price[t] * fcr_down[t]
            + sum(dispatch_config.penalties_soc_fin * soc_min_deviation[trajectory][t] ** 2
            + soc_min_deviation['baseline'][T-1] * dispatch_config.penalties_soc_fin
                for trajectory in {'baseline', 'fcr_up', 'fcr_down'})
            + (power['baseline']['charge'][t] - power['baseline']['discharge'][t]) * dispatch_config.cost_of_electricity[t] * (dispatch_config.time_mesh / 60)
            for t in range(1, T)
            )
        )


    solution = m.solve()

    #m.dump_as_lp(f"sub_pb_{datetime.now()}.lp")

    solution_dict = {
        "soc_baseline" : [solution[soc['baseline'][t]] for t in range(T)],
        "soc_fcr_up" : [solution[soc['fcr_up'][t]] for t in range(T)],
        "soc_fcr_down" : [solution[soc['fcr_down'][t]] for t in range(T)],
        "p_charge_baseline" : [0] + [solution[power['baseline']['charge'][t]] for t in range(1, T)],
        "p_discharge_baseline" : [0] + [solution[power['baseline']['discharge'][t]] for t in range(1, T)],
        "p_charge_fcr_up" : [0] + [solution[power['fcr_up']['charge'][t]] for t in range(1, T)],
        "p_discharge_fcr_up" : [0] + [solution[power['fcr_up']['discharge'][t]] for t in range(1, T)],
        "p_charge_fcr_down" : [0] + [solution[power['fcr_down']['charge'][t]] for t in range(1, T)],
        "p_discharge_fcr_down" : [0] + [solution[power['fcr_down']['discharge'][t]] for t in range(1, T)],
        "fcr_up" : [0] + [solution[fcr_up[t]] for t in range(1, T)],
        "fcr_down" : [0] + [solution[fcr_down[t]] for t in range(1, T)]        
    }
    #print(f"p_discharge_baseline_14 : {solution[power['baseline']['discharge'][14]]}")
    #print(f"soc baseline = {[solution[soc['baseline'][t]] for t in range(T)]}")
    #print(f"fcr_up = {[solution[fcr_up[t]] for t in range(T)]}")
    #print(f"fcr_down = {[solution[fcr_down[t]] for t in range(T)]}")
    with open(os.path.join(f"one_ev_solver_results.json"), 'w') as outfile:
        json.dump(solution_dict, outfile, indent = 3, sort_keys=True, default=str)
    #input()
    return solution_dict
