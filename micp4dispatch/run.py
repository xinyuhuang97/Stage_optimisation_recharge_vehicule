from src.instance.dispatch_config import DispatchConfig
import src.solver.sfw.coordinator as coordinator
import src.solver.fw.coordinator_fw as coordinator_fw
import src.solver.frontal.frontal as frontal
import src.solver.frontal.frontal_fw as frontal_fw
import data.generator as generator

import os
import json

import sys
import time

nb_ev = int(sys.argv[1])
nb_it = int(sys.argv[2])

tir_id = 0
if len(sys.argv) > 3:
    tir_id = int(sys.argv[3])

stats = {}

data = generator.instance_dict(nb_ev)
config = DispatchConfig(**data)

#print(f"\n--- Dispatch with {len(config.evses)} ev ---")

#print("\n> Relaxed Frontal Method")
#stats[tir_id]["lp obj"] = frontal_fw.optimize(config, nb_it * nb_ev)

#print("\n> FW Method")
#stats[tir_id] = coordinator_fw.dispatch(config)


if nb_ev <201 :
    config.frontal_opt= frontal.optimize(config)

print("\n> SFW Method")

t_init = time.time()
stats[tir_id] = coordinator.dispatch(config, nb_it)
stats[tir_id]["time"] = time.time() - t_init

print("\n> Frontal Method")
frontal.optimize(config)

with open(os.path.join("stats_sfw_no_gap", f"{nb_ev}ev_{nb_it}it_{tir_id}.json"), 'w') as outfile:
    json.dump(stats, outfile, indent = 3, sort_keys=True, default=str)