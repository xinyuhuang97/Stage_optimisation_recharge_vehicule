import json
import numpy as np
my_instance = "../data/instance_20.json"
data=json.load(open(my_instance))
N=len(data["evses"]) # Nombre d'evse
print("Nombre d'evse : ",N)
T=data["optim_horizon"] # Nombre de pas de temps
actual_time=0
penality_SOC_fin = data["penalties"]["SOC_fin"]
beta_min=data["penalties"]["beta_min"]
beta_max=data["penalties"]["beta_max"]
alpha=data["penalties"]["fcr_up"]
time_mesh_to_hour = data["time_mesh"]/60
soc_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)]
s_final=np.array([data["evses"][i]["SOC_final"]*data["evses"][i]["capacity"] for i in range(N)]) #liste contenant soc_final_i
s_i_t_min = np.array([[max(data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"], (s_final[i]-data["evses"][i]["p_charge_max"]*(T-t)*time_mesh_to_hour) ) for t in range(1,T+1)] for i in range(N)]) # Liste contenant s_i_t_min caleculer a partir de soc_min_i et p_d_bl_max_i
s_i_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)] # Liste contenant s_zxi_max