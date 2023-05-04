
import sys
import time
import numpy as np
import pandas as pd
sys.path.append('../data')
from tqdm import tqdm
from generator import *
from subpb_pl_cplex import *
from copy import deepcopy

beta_min=1
beta_max=1

# Generer une nouvelle instance a=instance_json(20)
instance_json(3)
data=json.load(open('../data/instance_3.json'))
N=len(data["evses"]) # Nombre d'evse
T=data["optim_horizon"] # Nombre de pas de temps
M=15 # Nombre de pas de temps dans un intervalle de temps
# Actuel_time \in {0,1,...,H-1}???
actuel_time=0

def Frank_Wolfe_ev(data, actuel_time, verbose=False):
    """!!! gap primal-duale"""
    # Initialisation
    x_bar_0=initialisation_x(data) # Initialisation de x_bar_0
    K= max(2*N,100) # Nb d'itérations
    progress_bar = tqdm(total=K, unit='iteration')
    nk=[10]*int(K) # Nb de tirages 

    # Initialisation des np.array pour stocker les valeurs de x_bar_k
    charge=np.zeros((N,T))
    charge_bar=np.zeros((N,T))
    discharge=np.zeros((N,T))
    discharge_bar=np.zeros((N,T))

    x_bar_k=x_bar_0

    # Preparation des donnees s_i_t_min et s_i_max pour les mettre dans la fonction objective
    s_final=np.array([data["evses"][k]["SOC_final"]*data["evses"][k]["capacity"] for k in range(N)]) #liste contenant soc_final_i
    s_i_t_min = np.array([[max(data["evses"][k]["SOC_min"]*data["evses"][k]["capacity"], s_final[k]-data["evses"][k]["p_discharge_max"]*(T-t)) for t in range(1,T+1)] for k in range(N)]) # Liste contenant s_i_t_min caleculer a partir de soc_min_i et p_discharge_max_i
    s_i_max=[data["evses"][k]["p_charge_max"] for k in range(N)] # Liste contenant s_zxi_max
    
    # Boucle principale
    for k in range(K):

        # charge_service

        # Le gradient de (y - y^up)^2 -> 2(y - y^up) with y = x_bar_k["charge"]-x_bar_k["discharge"] - x_bar_k["charge_bar"] + x_bar_k["discharge_bar"] et y^up = 
        """ add alpha """
        lambda_k= 2/N*np.sum((x_bar_k["charge"]-x_bar_k["discharge"])-(x_bar_k["charge_bar"]-x_bar_k["discharge_bar"])\
                             -data["announced_capacity"]["up"][actuel_time:actuel_time+T],axis=0)#data["cost_of_electricity"][actuel_time:actuel_time+T])
        print("lambda_k",lambda_k)
        # Resolution des sous-problemes
        for i in range(N):
            #c_i, d_i, c_bar_i, d_bar_i, _, _ =resolution_subpb(data, lambda_k[i],x_bar_k,i, verbose=verbose)
            c_i, d_i, c_bar_i, d_bar_i =resolution_subpb(data, lambda_k,x_bar_k,i, verbose=verbose)
            #print(charge[i])
            charge[i]=(c_i)
            #print(charge[i])
            charge_bar[i]=(d_i)
            discharge[i]=(c_bar_i)
            discharge_bar[i]=(d_bar_i)
        # Obtention de x_k a partir de x_bar_k
        x_k = {"charge": charge, "discharge": discharge, "charge_bar": charge_bar, "discharge_bar": discharge_bar}
        #print("x_k",x_k)
        delta_k = 2/(k+2) #

        # Tirage de nk[k] echantillons
        list_sample = []
        for _ in range(nk[k]):
            sample_charge=np.zeros((N,T))
            sample_discharge=np.zeros((N,T))
            sample_charge_bar=np.zeros((N,T))
            sample_discharge_bar=np.zeros((N,T))
            
            for i in range(N):

                draw=np.random.rand() # Tirer au sort un valeur entre 0 et 1
                # Si draw <= 1-delta_k alors on prend x_bar_k sinon on prend x_k
                if draw<=1-delta_k:
                    sample_charge[i]=deepcopy(x_bar_k["charge"][i])
                    sample_discharge[i]=deepcopy(x_bar_k["discharge"][i])
                    sample_charge_bar[i]=deepcopy(x_bar_k["charge_bar"][i])
                    sample_discharge_bar[i]=deepcopy(x_bar_k["discharge_bar"][i])
                else:
                    sample_charge[i]=deepcopy(charge[i])
                    sample_discharge[i]=deepcopy(discharge[i])
                    sample_charge_bar[i]=deepcopy(charge_bar[i])
                    sample_discharge_bar[i]=deepcopy(discharge_bar[i])

            sample = {"charge": sample_charge, "discharge": sample_discharge, "charge_bar": sample_charge_bar, "discharge_bar": sample_discharge_bar}
            list_sample.append(sample)

        # Chercher la meilleur echantillon et calculer le score de x_bar_k, x_k et la meilleure echantillon
        best_sample = min(list_sample, key=lambda x: objective_function(data, x, s_i_t_min, s_i_max))
        best_sample_score= objective_function(data, best_sample, s_i_t_min, s_i_max)
        x_bar_k_score = objective_function(data, x_bar_k, s_i_t_min, s_i_max)
        x_k_score = objective_function(data, x_k, s_i_t_min, s_i_max)
        print("best_sample_score",best_sample_score)
        print("x_bar_k_score",x_bar_k_score)
        print("x_k_score",x_k_score)
        # Mettre a jour x_bar_k en fonction des scores de x_bar_k, x_k et du meilleur echantillon
        if best_sample_score > x_bar_k_score or best_sample_score > x_k_score:
            if x_bar_k_score > x_k_score:
                x_bar_k = deepcopy(x_k)
            else:
                x_bar_k = deepcopy(x_bar_k)
        else:
            x_bar_k =deepcopy(best_sample)
        print("after",objective_function(data, x_bar_k, s_i_t_min, s_i_max))
        print("====================")
        if (k+1)%10==0 :
            progress_bar.update(10)
        #print(objective_function(data, x_bar_k, s_i_t_min, s_i_max))
        #print(x_bar_k)
    progress_bar.close()
    #print(x_bar_k, objective_function(data, x_bar_k, s_i_t_min, s_i_max))
    return x_bar_k
        


def initialisation_x(data):

    # Initialisation de x_bar_0
    charge=np.zeros((N,T))
    charge_bar=np.zeros((N,T))
    discharge=np.zeros((N,T))
    discharge_bar=np.zeros((N,T))
    charge_level=np.zeros(N)
    charge_level_bar=np.zeros(N)
    
    for i in range(N):
        #soc_bl
        #soc_up
        charge_level[i]=data["evses"][i]["SOC_init"]
        charge_level_bar[i]=data["evses"][i]["SOC_init"]
        p_charge_max=data["evses"][i]["p_charge_max"]
        p_charge_min=data["evses"][i]["p_charge_min"]
        p_discharge_max=data["evses"][i]["p_discharge_max"]
        p_discharge_min=data["evses"][i]["p_discharge_min"]
        soc_max=data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"]
        soc_min=data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"]

        for j in range(actuel_time,actuel_time+data["optim_horizon"]):
            draw = np.random.randint(0,3)
            if draw == 0:
                if charge_level[i]+p_charge_min<=soc_max:
                    if charge_level[i]+p_charge_max>=soc_max:
                        charge[i,j]=np.random.randint(p_charge_min,soc_max-charge_level[i]+1)
                    else:
                        charge[i,j]=np.random.randint(p_charge_min,p_charge_max+1)
                    charge_level[i]+=charge[i,j]
                if charge_level_bar[i]+p_charge_min<=soc_max:
                    if charge_level_bar[i]+p_charge_max>=soc_max:
                        charge_bar[i,j]=np.random.randint(p_charge_min,soc_max-charge_level_bar[i]+1)
                    else:
                        charge_bar[i,j]=np.random.randint(p_charge_min,p_charge_max+1)
                    charge_level_bar[i]+=charge_bar[i,j]

            elif draw == 1:
                if charge_level[i]-p_discharge_min>=soc_min:
                    if charge_level[i]-p_discharge_max<=soc_min:
                        discharge[i,j]=np.random.randint(p_discharge_min,charge_level[i]-soc_min+1)
                    else:
                        discharge[i,j]=np.random.randint(p_discharge_min,p_discharge_max+1)
                    charge_level[i]-=discharge[i,j]
                if charge_level_bar[i]-p_discharge_min>=soc_min:
                    if charge_level_bar[i]-p_discharge_max<=soc_min:
                        discharge_bar[i,j]=np.random.randint(p_discharge_min,charge_level_bar[i]-soc_min+1)
                    else:
                        discharge_bar[i,j]=np.random.randint(p_discharge_min,p_discharge_max+1)
                    charge_level_bar[i]-=discharge_bar[i,j]
                    
    return {"charge": charge, "discharge": discharge, "charge_bar": charge_bar, "discharge_bar": discharge_bar, "charge_level": charge_level, "charge_level_bar": charge_level_bar}

def objective_function(data, x, s_i_t_min, s_i_max):

    # Recuperer les valeurs des variables
    charge=x["charge"]
    discharge=x["discharge"]
    charge_bar=x["charge_bar"]
    discharge_bar=x["discharge_bar"]
    charge_level=np.zeros((N,T))
    charge_level_bar=np.zeros((N,T))

    # Calculer les niveaux de charge pour chaque vehicule et chaque instant
    for i in range(N):
        charge_level[i,0]=data["evses"][i]["SOC_init"]
        charge_level_bar[i,0]=data["evses"][i]["SOC_init"]
        """check range(1,T) or (1,T+1)"""
        for t in range(1,T):
            charge_level[i,t]=charge_level[i,t-1]+charge[i,t]-discharge[i,t]
            charge_level_bar[i,t]=charge_level_bar[i,t-1]+charge_bar[i,t]-discharge_bar[i,t]
    """check T-1 or T"""
    s_T=charge_level[:,T-1]
    #s_T=np.sum([x["charge"][i][t-1]-x["discharge"][i][t-1] for t in range(1,T+1) for i in range(N)])

    # Calculer la fonction objectif
    cost=0 # Valeur de la fonction objectif
    sum_f=0 # Somme de la fonction f
    for i in range(N):
        #print(charge_level[i,T-1])
        sum_f+=charge[i]-discharge[i]-charge_bar[i]+discharge_bar[i]
        for j in range(actuel_time,actuel_time+data["optim_horizon"]):
            hi=  beta_min*(-min(charge_level[i,j]-s_i_t_min[i,j],0))**2/N + beta_max*max(charge_level[i,j]-s_i_max[i],0)**2/N + \
                beta_min*(-min(charge_level_bar[i,j]-s_i_t_min[i,j],0))**2/N + beta_max*max(charge_level_bar[i,j]-s_i_max[i],0)**2/N
            cost_electricity = (charge[i,j]-discharge[i,j])*data["cost_of_electricity"][j]*data["time_mesh"]/60/N
            """print("-----------------")
            print(hi)
            print(cost_electricity)
            print("-----------------")"""
            cost+=hi+cost_electricity
            """cost+=(charge[i,j]-discharge[i,j])*data["cost_of_electricity"][j]*data["time_mesh"]/60/N
            cost-= min(charge_level[i,j]-s_i_t_min[i,j],0)**2/N
            cost+= max(charge_level[i,j]-s_i_max[i],0)**2/N
            cost-= min(charge_level_bar[i,j]-s_i_t_min[i,j],0)**2/N
            cost+= max(charge_level_bar[i,j]-s_i_max[i],0)**2/N"""
        # We want the level of charge to be the high at the end of the optimization horizon
        cost-=s_T[i]*data["cost_of_electricity"][actuel_time+data["optim_horizon"]]/N 
    cost+=np.sum((sum_f/N- data["announced_capacity"]["up"][actuel_time:actuel_time+data["optim_horizon"]])**2,axis=0)
    #print(cost)
    return cost

#print(data['evses'][0]['p_charge_max'])
Frank_Wolfe_ev(data,actuel_time)