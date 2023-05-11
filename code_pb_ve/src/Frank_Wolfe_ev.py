
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
#instance_json(3)
data=json.load(open('../data/instance_100.json'))
N=len(data["evses"]) # Nombre d'evse
T=data["optim_horizon"] # Nombre de pas de temps
# Actuel_time \in {0,1,...,H-1}???
actuel_time=0
penality_SOC_fin = data["penalties"]["SOC_fin"]

def Frank_Wolfe_ev(data, actuel_time, verbose=False):
    """!!! gap primal-duale"""
    # Initialisation
    x_bar_0=initialisation_x(data) # Initialisation de x_bar_0
    K= max(2*N,100) # Nb d'itérations
    progress_bar = tqdm(total=K, unit='iteration')
    nk=[10]*int(K) # Nb de tirages 

    # Initialisation des np.array pour stocker les valeurs de x_bar_k
    c_bl=np.zeros((N,T))
    c_up=np.zeros((N,T))
    d_bl=np.zeros((N,T))
    d_up=np.zeros((N,T))
    soc=np.zeros((N,T))
    soc_bar=np.zeros((N,T))

    x_bar_k=x_bar_0

    # Preparation des donnees s_i_t_min et s_i_max pour les mettre dans la fonction objective
    s_final=np.array([data["evses"][i]["SOC_final"]*data["evses"][i]["capacity"] for i in range(N)]) #liste contenant soc_final_i
    s_i_t_min = np.array([[max(data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"], s_final[i]-data["evses"][i]["p_discharge_max"]*(T-t)) for t in range(1,T+1)] for i in range(N)]) # Liste contenant s_i_t_min caleculer a partir de soc_min_i et p_d_bl_max_i
    s_i_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)] # Liste contenant s_zxi_max
    
    

    # Boucle principale
    for k in range(K):
        # Le gradient de (y - y^up)^2 -> 2(y - y^up) with y = x_bar_k["c_bl"]-x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"] et y^up = 
        """ add alpha """
        #print("gu",x_bar_k["c_bl"])
        print(np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"]\
                            ,axis=0)/N-[x/N for x in data["announced_capacity"]["up"][actuel_time:actuel_time+T]])
        lambda_k = (np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"]\
                             ,axis=0)/N-[x/N for x in data["announced_capacity"]["up"][actuel_time:actuel_time+T]])*2
        
        #print(lambda_k,T)
        
        # Resolution des sous-problemes
        """print("before",soc)
        print(c_bl)
        print(c_up)
        print(d_bl)
        print(d_up)
        """
        for i in range(N):
            c_i, d_i, c_bar_i, d_bar_i ,soc_i, soc_bar_i, y=resolution_subpb(data, lambda_k,x_bar_k,i, verbose=verbose)
            c_bl[i]=(c_i)
            c_up[i]=(d_i)
            d_bl[i]=(c_bar_i)
            d_up[i]=(d_bar_i)
            soc[i]=soc_i
            soc_bar[i]=soc_bar_i

        # Obtention de x_k a partir de x_bar_k
        x_k = {"c_bl": c_bl, "d_bl": d_bl, "c_up": c_up, "d_up": d_up, "soc": soc, "soc_bar": soc_bar}
        #print("why",objective_function(data, x_k, s_i_t_min, s_i_max))
        delta_k = 2/(k+2) #

        # Tirage de nk[k] echantillons
        list_sample = []
        for _ in range(nk[k]):
            sample_c_bl=np.zeros((N,T))
            sample_d_bl=np.zeros((N,T))
            sample_c_up=np.zeros((N,T))
            sample_d_up=np.zeros((N,T))
            sample_s_bl=np.zeros((N,T))
            sample_s_up=np.zeros((N,T))
            
            for i in range(N):

                draw=np.random.rand() # Tirer au sort un valeur entre 0 et 1
                # Si draw <= 1-delta_k alors on prend x_bar_k sinon on prend x_k
                if draw<=1-delta_k:
                    sample_c_bl[i]=deepcopy(x_bar_k["c_bl"][i])
                    sample_d_bl[i]=deepcopy(x_bar_k["d_bl"][i])
                    sample_c_up[i]=deepcopy(x_bar_k["c_up"][i])
                    sample_d_up[i]=deepcopy(x_bar_k["d_up"][i])
                    sample_s_bl[i]=deepcopy(x_bar_k["soc"][i])
                    sample_s_up[i]=deepcopy(x_bar_k["soc_bar"][i])
                else:
                    sample_c_bl[i]=deepcopy(c_bl[i])
                    sample_d_bl[i]=deepcopy(d_bl[i])
                    sample_c_up[i]=deepcopy(c_up[i])
                    sample_d_up[i]=deepcopy(d_up[i])
                    sample_s_bl[i]=deepcopy(soc[i])
                    sample_s_up[i]=deepcopy(soc_bar[i])
            sample = {"c_bl": sample_c_bl, "d_bl": sample_d_bl, \
                      "c_up": sample_c_up, "d_up": sample_d_up,\
                      "soc": sample_s_bl, "soc_bar": sample_s_up}
            list_sample.append(sample)

        # Chercher la meilleur echantillon et calculer le score de x_bar_k, x_k et la meilleure echantillon
        best_sample = min(list_sample, key=lambda x: objective_function(data, x, s_i_t_min, s_i_max))
        best_sample_score= objective_function(data, best_sample, s_i_t_min, s_i_max)
        x_bar_k_score = objective_function(data, x_bar_k, s_i_t_min, s_i_max)
        #print("x_k!!!!")
        x_k_score = objective_function(data, x_k, s_i_t_min, s_i_max)
        print("best_sample_score",best_sample_score)
        #print(best_sample)
        print("x_bar_k_score",x_bar_k_score)
        print("x_k_score",x_k_score)
        # Mettre a jour x_bar_k en fonction des scores de x_bar_k, x_k et du meilleur echantillon
        if k!=0:
            if best_sample_score > x_bar_k_score or best_sample_score > x_k_score:
                if x_bar_k_score > x_k_score:
                    x_bar_k = deepcopy(x_k)
                else:
                    x_bar_k = deepcopy(x_bar_k)
            else:
                x_bar_k =deepcopy(best_sample)
        else:
            x_bar_k = deepcopy(best_sample)
        print("after",objective_function(data, x_bar_k, s_i_t_min, s_i_max))
        print("====================")
        if (k+1)%10==0 :
            progress_bar.update(10)
    progress_bar.close()
    return x_bar_k
        

def objective_function(data, x, s_i_t_min, s_i_max):

    # Recuperer les valeurs des variables
    c_bl=x["c_bl"]
    d_bl=x["d_bl"]
    c_up=x["c_up"]
    d_up=x["d_up"]
    s_bl=x["soc"]
    s_up=x["soc_bar"]
    y_val= np.sum([ (np.sum([c_bl[i][t]-d_bl[i][t]-c_up[i][t]+d_up[i][t] for i in range(N) ])/N-data["announced_capacity"]["up"][actuel_time+t]/N)**2 for t in range(T)  ])
    s_T=s_bl[:,T-1]
    # Calculer la fonction objectif
    cost=0 # Valeur de la fonction objectif
    
    for i in range(N):
        for j in range(actuel_time,actuel_time+data["optim_horizon"]):
            hi=  (beta_min*(-min(s_bl[i,j]-s_i_t_min[i,j],0))**2 + \
                  beta_max*max(s_bl[i,j]-s_i_max[i],0)**2 + \
                  beta_min*(-min(s_up[i,j]-s_i_t_min[i,j],0))**2+ \
                  beta_max*max(s_up[i,j]-s_i_max[i],0)**2)/N
            cost_electricity = (c_bl[i,j]-d_bl[i,j])*data["cost_of_electricity"][j]*data["time_mesh"]/60/N
            cost+=hi+cost_electricity
        # We want the level of c_bl to be the high at the end of the optimization horizon
        cost-=s_T[i]*data["cost_of_electricity"][actuel_time+data["optim_horizon"]]*penality_SOC_fin/N 
    #print(y_val,"y_val")
    cost+=y_val
    #print(cost1, cost2)
    return cost


def initialisation_x(data):

    # Initialisation de x_bar_0
    c_bl=np.zeros((N,T))
    c_up=np.zeros((N,T))
    d_bl=np.zeros((N,T))
    d_up=np.zeros((N,T))
    soc=np.zeros((N,T))
    soc_bar=np.zeros((N,T))

    return {"c_bl": c_bl, "d_bl": d_bl, "c_up": c_up, \
            "d_up": d_up, "soc": soc, \
            "soc_bar": soc_bar, "soc": soc}

Frank_Wolfe_ev(data,actuel_time)