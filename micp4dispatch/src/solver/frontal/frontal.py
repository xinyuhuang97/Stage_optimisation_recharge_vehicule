from docplex.mp.model import Model
from ...instance.dispatch_config import DispatchConfig, EVSE

def optimize(dispatch_config: DispatchConfig):
    T = dispatch_config.optim_horizon
    N = len(dispatch_config.evses)
    alpha_up = dispatch_config.penalties_fcr_up
    alpha_down = dispatch_config.penalties_fcr_down
    nb_ev = N

    m = Model(name='frontal')

    activation = {}
    power = {}
    soc = {}
    soc_min_deviation = {}
    fcr_up = {}
    fcr_down = {}

    for n, evse in enumerate(dispatch_config.evses):
        activation[n] = {}
        power[n] = {}
        soc[n] = {}
        soc_min_deviation[n] = {}

        fcr_up[n] = m.continuous_var_list(T,lb = 0, ub = 1e9, name = f"fcr_up_{n}")
        fcr_down[n] = m.continuous_var_list(T, lb = 0, ub = 1e9, name = f"fcr_down_{n}")

        for trajectory in 'baseline', 'fcr_up', 'fcr_down':
            activation[n][trajectory] = {}
            power[n][trajectory] = {}
            soc[n][trajectory] = m.continuous_var_list(T, lb = 0,
                ub = evse.soc_max, name = f"soc_{n}_{trajectory}")
            soc_min_deviation[n][trajectory] = m.continuous_var_list(T, lb = 0, ub = 1e9,
                name = f"soc_min_deviation_{n}_{trajectory}")

            for sense in 'charge', 'discharge':
                activation[n][trajectory][sense] = m.binary_var_list(T, name = f"activation_{n}_{trajectory}_{sense}")
                power[n][trajectory][sense] = m.continuous_var_list(T, lb = 0, name = f"power_{n}_{trajectory}_{sense}")

            m.add_constraints([activation[n][trajectory]['charge'][t] +
                activation[n][trajectory]['discharge'][t] <= 1 for t in range(1, T)],
                [f"cstr_icds_{n}_{trajectory}_{t}" for t in range(1, T)])

            m.add_constraints([power[n][trajectory]['charge'][t] <=
                evse.p_charge_max * activation[n][trajectory]['charge'][t] for t in range(1, T)],
                [f"cstr_chrg_max_{n}_{trajectory}_{t}" for t in range(1, T)])

            m.add_constraints([power[n][trajectory]['discharge'][t] <=
                evse.p_charge_max * activation[n][trajectory]['discharge'][t] for t in range(1, T)],
                [f"cstr_dischrg_max_{n}_{trajectory}_{t}" for t in range(1, T)])

            m.add_constraint(soc[n][trajectory][0] == evse.soc_init,
                f"cstr_soc_init_{n}_{trajectory}")

            m.add_constraints([soc[n][trajectory][t] - soc[n]['baseline'][t - 1]
                == (power[n][trajectory]['charge'][t] - power[n][trajectory]['discharge'][t])
                * (dispatch_config.time_mesh / 60) / evse.capacity for t in range(1, T)],
                names = [f"cstr_soc_evol_{n}_{trajectory}_{t}" for t in range(1, T)])

            m.add_constraints([soc_min_deviation[n][trajectory][t]
                >= evse.soc_lance_curve[t] - soc[n][trajectory][t] for t in range(1, T)],
                names = [f"cstr_soc_deviation_{n}_{trajectory}_{t}" for t in range(1, T)])

        m.add_constraints([power[n]['baseline']['charge'][t]
            >= evse.p_charge_min * activation[n]['baseline']['charge'][t] for t in range(1, T)],
            names = [f"cstr_chrg_min_baseline_{n}_{t}" for t in range(1, T)])

        m.add_constraints([power[n]['baseline']['discharge'][t]
            >= evse.p_discharge_min * activation[n]['baseline']['discharge'][t] for t in range(1, T)],
            names = [f"cstr_dischrg_min_baseline_{n}_{t}" for t in range(1, T)])

        m.add_constraints([power[n]['baseline']['charge'][t] - power[n]['baseline']['discharge'][t]
            - power[n]['fcr_up']['charge'][t] + power[n]['fcr_up']['discharge'][t]
            == fcr_up[n][t] for t in range(1, T)], names = [f"cstr_fcr_up_{n}_{t}" for t in range(1, T)])

        m.add_constraints([power[n]['fcr_down']['charge'][t] - power[n]['fcr_down']['discharge'][t]
            - power[n]['baseline']['charge'][t] + power[n]['baseline']['discharge'][t]
            == fcr_down[n][t] for t in range(1, T)], names = [f"cstr_fcr_down_{n}_{t}" for t in range(1, T)])

    m.set_objective("min", sum(sum(sum(dispatch_config.penalties_soc_fin  * soc_min_deviation[n][trajectory][t] ** 2
                    for trajectory in {'baseline', 'fcr_up', 'fcr_down'}) for t in range(1, T))
                for n, evse in enumerate(dispatch_config.evses)) / nb_ev 
            + sum(soc_min_deviation[n]['baseline'][T-1]/N for n in range(N)) * dispatch_config.penalties_soc_fin
            + sum(alpha_up * (sum(fcr_up[n][t] for n in range(N)) - dispatch_config.announced_capacity_up[t]) ** 2
                + alpha_down * (sum(fcr_down[n][t] for n in range(N)) - dispatch_config.announced_capacity_down[t]) ** 2
                for t in range (1, T)) / nb_ev**2
            + sum(sum((power[n]['baseline']['charge'][t] - power[n]['baseline']['discharge'][t]) * dispatch_config.cost_of_electricity[t] * (dispatch_config.time_mesh / 60)
                for t in range (1, T)) for n in range(N))/ nb_ev 
            )

    solution = m.solve()

    cout_charge = sum(sum((solution[power[n]['baseline']['charge'][t]] - solution[power[n]['baseline']['discharge'][t]]) * dispatch_config.cost_of_electricity[t] * (dispatch_config.time_mesh / 60)
                for t in range (1, T)) for n in range(N))
    penalite_lance_curve = sum(sum(sum(dispatch_config.penalties_soc_fin  * solution[soc_min_deviation[n][trajectory][t]] ** 2
        for t in range(1, T))
        for trajectory in {'baseline', 'fcr_up', 'fcr_down'})
        + 1e6*dispatch_config.penalties_soc_fin  * solution[soc_min_deviation[n]['baseline'][-1]]
        for n, evse in enumerate(dispatch_config.evses))
    penalite_fcr_up = sum(alpha_up * (sum(solution[fcr_up[n][t]] for n in range(N)) - dispatch_config.announced_capacity_up[t]) ** 2
        for t in range (1, T))
    penalite_fcr_down = sum(alpha_down * (sum(solution[fcr_down[n][t]] for n in range(N)) - dispatch_config.announced_capacity_down[t]) ** 2
        for t in range (1, T))

    print(f"frontal objective value : \t {m.objective_value}")
    print(f"  av penalite lance curve : \t {penalite_lance_curve/ nb_ev}")
    print(f"  av penalite FCR up : \t\t {penalite_fcr_up/ nb_ev}")
    print(f"  av penalite FCR down : \t\t {penalite_fcr_down/ nb_ev}")
    print(f"  av cout charge : \t\t {cout_charge/ nb_ev}")


    print(f"av charge baseline : {[sum(solution[power[n]['baseline']['charge'][t]] for n in range(N))/(N) for t in range(T)]}")
    print(f"av charge fcr up : {[sum(solution[power[n]['fcr_up']['charge'][t]] for n in range(N))/(N) for t in range(T)]}")
    print(f"av charge fcr down : {[sum(solution[power[n]['fcr_down']['charge'][t]] for n in range(N))/(N) for t in range(T)]}")
    print(f"av discharge baseline : {[sum(solution[power[n]['baseline']['discharge'][t]] for n in range(N))/(N) for t in range(T)]}")
    print(f"av discharge fcr up : {[sum(solution[power[n]['fcr_up']['discharge'][t]] for n in range(N))/(N) for t in range(T)]}")
    print(f"av discharge fcr down : {[sum(solution[power[n]['fcr_down']['discharge'][t]] for n in range(N))/(N) for t in range(T)]}")
    print(f"av soc baseline : {[sum(solution[soc[n]['baseline'][t]] for n in range(N))/(N) for t in range(T)]}")
    print(f"av service up : {[sum(solution[fcr_up[n][t]] for n in range(N))/ nb_ev for t in range(T)]}")
    print(f"av service down : {[sum(solution[fcr_down[n][t]] for n in range(N))/ nb_ev for t in range(T)]}")

    return m.objective_value
