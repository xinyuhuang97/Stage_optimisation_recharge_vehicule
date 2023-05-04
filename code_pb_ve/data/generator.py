import json
import random
import copy
import sys

#random.seed(123)

default_data = {
    "optim_horizon" : 3,
    "fcr_commit" : 15,
    "time_mesh" : 15,
    "announced_capacity" : {
        "up" : [7000.0 for _ in range (96)],
        "down" : [0.0 for _ in range (96)]
    },
    "evses" : [],
    #"cost_of_electricity" : [0.1 - 0.000/(t+1) for t in range(96)],
    "cost_of_electricity" : [0 for t in range(96)],
    "penalties" : {
        "fcr_up" : 0.01,# alpha
        "fcr_down" : 0,
        "SOC_fin" : 10000
        #"SOC_fin" : 100000
    }
}

default_evse = {
    "charging_horizon" : 96,
    "p_charge_max" : 15000,
    #"p_charge_min" : 1000,
    "p_charge_min" : 1000,
    "p_discharge_max" : 15000,
    "p_discharge_min" : 1000,
    "conversion_rate" : 1, # not used
    "capacity" : 30000,
    "SOC_max" : 1.0,
    "SOC_init" : 0.3,
    "SOC_min" : 0.0,
    "SOC_final" : 0.5
}
"""
new_evse = {
    "charging_horizon" : 96,
    "p_charge_max" : 15000,
    #"p_charge_min" : 1000,
    "p_charge_min" : 1000,
    "p_discharge_max" : 15000,
    "p_discharge_min" : 1000,
    "conversion_rate" : 1, # not used
    "capacity" : 30000,
    "SOC_max" : 1.0,
    "SOC_init" : 0.3,
    "SOC_min" : 0.0,
    "SOC_final" : 0.5
}"""

def instance_json(nb_ev):
    data = copy.deepcopy(default_data)
    data["announced_capacity"]["up"] = [_ * nb_ev for _ in data["announced_capacity"]["up"]]
    data["announced_capacity"]["down"] = [_ * nb_ev for _ in data["announced_capacity"]["down"]]
    for ev_id in range(nb_ev):
        ev = copy.deepcopy(default_evse)
        ev["id"] = f"EV_{ev_id}"
        ev["charging_horizon"] = data["optim_horizon"]
        ev["SOC_init"] += (- 0.05 + random.random()/10)
        ev["SOC_final"] += (- 0.05 + random.random()/10)
        data["evses"].append(ev)

    with open(f"../data/instance_{nb_ev}.json", 'w') as f:
        json.dump(data, f)


def instance_dict(nb_ev):
    data = copy.deepcopy(default_data)
    data["announced_capacity"]["up"] = [_ * nb_ev for _ in data["announced_capacity"]["up"]]
    data["announced_capacity"]["down"] = [_ * nb_ev for _ in data["announced_capacity"]["down"]]
    for ev_id in range(nb_ev):
        ev = copy.deepcopy(default_evse)
        ev["id"] = f"EV_{ev_id}"
        ev["charging_horizon"] = data["optim_horizon"]
        ev["SOC_init"] += (- 0.05 + random.random()/10)
        ev["SOC_final"] += (- 0.05 + random.random()/10)
        data["evses"].append(ev)
    return data
