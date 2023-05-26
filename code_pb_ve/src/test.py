import sys
import numpy as np
import json
sys.path.append('../data')
sys.path.append('/Applications/CPLEX_Studio221/cplex/python/3.7/x86-64_osx/cplex/_internal')
#import py37_cplex2210 as cplex
import cplex


data=json.load(open('../data/instance_3.json'))

N=len(data["evses"]) # Nombre d'evse
T=data["optim_horizon"]

soc_init=[data["evses"][i]["SOC_init"]*data["evses"][i]["capacity"] for i in range(N)]
soc_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)]
soc_min=[data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"] for i in range(N)]

p_charge_max=[data["evses"][i]["p_charge_max"] for i in range(N)]
p_charge_min=[data["evses"][i]["p_charge_min"] for i in range(N)]
p_discharge_max=[data["evses"][i]["p_discharge_max"] for i in range(N)]
p_discharge_min=[data["evses"][i]["p_discharge_min"] for i in range(N)]

s_final = [data["evses"][i]["SOC_final"]*data["evses"][i]["capacity"] for i in range(N)]
s_t_min = [[max(data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"],s_final[i]-p_discharge_max[i]*(T-t) ) for t in range(1,T+1)] for i in range(N)]

s_bl = [["s_bl_" + str(i) +"_" +str(t) for t in range(T+1)]  for i in range(N) ]
c_bl = [["c_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
d_bl = [["d_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
u_bl = [["u_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
v_bl = [["v_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
n_bl = [["n_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
p_bl = [["p_bl_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]

s_up = [["s_up_" + str(i) +"_" +str(t) for t in range(T+1)]  for i in range(N) ]
c_up = [["c_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
d_up = [["d_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
u_up = [["u_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
v_up = [["v_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
n_up = [["n_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]
p_up = [["p_up_" + str(i) +"_" +str(t) for t in range(1,T+1)]for i in range(N) ]

y = ["y_" + str(t) for t in range(1,T+1)]


problem = cplex.Cplex()
"""
problem.variables.add(names=["x"])
problem.objective.set_quadratic_coefficients("x","x",2)
problem.objective.set_sense(problem.objective.sense.minimize)
"""

problem.read("Frontal_problem.lp")

problem.solve()
sample_c_bl=np.zeros((N,T))
sample_d_bl=np.zeros((N,T))
sample_c_up=np.zeros((N,T))
sample_d_up=np.zeros((N,T))
sample_s_bl=np.zeros((N,T))
sample_s_up=np.zeros((N,T))
sample_n_bl=np.zeros((N,T))
sample_n_up=np.zeros((N,T))
sample_p_bl=np.zeros((N,T))
sample_p_up=np.zeros((N,T))
for i in range(N):
    sample_c_bl[i]=(problem.solution.get_values(c_bl[i]))
    sample_d_bl[i]=(problem.solution.get_values(c_up[i]))
    sample_c_up[i]=(problem.solution.get_values(d_bl[i]))
    sample_d_up[i]=(problem.solution.get_values(d_up[i]))
    sample_s_bl[i]=(problem.solution.get_values(s_bl[i][1:]))
    sample_s_up[i]=(problem.solution.get_values(s_up[i][1:]))
    for t in range(1,T+1):
        sample_n_bl[i][t-1]=(problem.solution.get_values("n_bl_"+str(i)+"_"+str(t)))
        sample_n_up[i][t-1]=(problem.solution.get_values("n_up_"+str(i)+"_"+str(t)))
        sample_p_bl[i][t-1]=(problem.solution.get_values("p_bl_"+str(i)+"_"+str(t)))
        sample_p_up[i][t-1]=(problem.solution.get_values("p_up_"+str(i)+"_"+str(t)))


print("n_bl",sample_n_bl)
print("n_up",sample_n_up)
print("p_bl",sample_p_bl)
print("p_up",sample_p_up)
print("c_bl",sample_c_bl)
print("d_bl",sample_d_bl)
print("c_up",sample_c_up)
print("d_up",sample_d_up)
print("s_bl",sample_s_bl)
print("s_up",sample_s_up)
print("s_t_min",s_t_min)
print("s_t_max",soc_max)
print("y",problem.solution.get_values(y))
print("valeur optimal retourne par cplex",problem.solution.get_objective_value()) 
