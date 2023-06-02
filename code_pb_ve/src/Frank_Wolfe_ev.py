

import sys
import time
import numpy as np
import pandas as pd
sys.path.append('../data')
from tqdm import tqdm
from generator import *
from subpb_pl_cplex import *
from tools import *
#from copy import 


# Generer une nouvelle instance a=instance_json(20)
#instance_json(3)
if __name__ == "__main__":
    my_instance = "../data/instance_10.json"
actual_time=0
#file_name="Frontal_problem.lp"

data=json.load(open(my_instance))


N=len(data["evses"]) # Nombre d'evse
print("Nombre d'evse : ",N)
T=data["optim_horizon"] # Nombre de pas de temps
# Actual_time \in {0,1,...,H-1}???
actual_time=0
penality_SOC_fin = data["penalties"]["SOC_fin"]
beta_min=data["penalties"]["beta_min"]
beta_max=data["penalties"]["beta_max"]
alpha=data["penalties"]["fcr_up"]
soc_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)]
time_mesh_to_hour = data["time_mesh"]/60

# Afficher la fonction objective, gap primal-dual,evolution au cours du temps/iterations
# Afficher l'evolution de le moyen/std de la puissance de chargement a chaque pas de temps/ pour une vehicule choisi aleatoirement (Service/baseline)
# L'evolution de soc(service/baseline) pour une vehicule choisi aleatoirement 
# Comparer avec les differentes valeurs des parametres \beta \alpha \gamma \cost_elec

def FW_solve(data, actual_time, verbose=False, analyse=False, K=max(2*N,100), nk=50):
    """!!! gap primal-duale"""
    # Initialisation
    df=0
    x_bar_0=initialisation_x(data) # Initialisation de x_bar_0
    K= K#max(2*N,100) # Nb d'itérations
    nk=[nk]*int(K) # Nb de tirages 
    
    progress_bar = tqdm(total=K, unit='iteration')

    # Initialisation des np.array pour stocker les valeurs de x_bar_k
    c_bl=np.zeros((N,T))
    c_up=np.zeros((N,T))
    d_bl=np.zeros((N,T))
    d_up=np.zeros((N,T))
    s_bl=np.zeros((N,T))
    s_up=np.zeros((N,T))

    x_bar_k=x_bar_0

    # Preparation des donnees s_i_t_min et s_i_max pour les mettre dans la fonction objective
    s_final=np.array([data["evses"][i]["SOC_final"]*data["evses"][i]["capacity"] for i in range(N)]) #liste contenant soc_final_i
    s_i_t_min = np.array([[max(data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"], (s_final[i]-data["evses"][i]["p_charge_max"]*(T-t)) ) for t in range(1,T+1)] for i in range(N)]) # Liste contenant s_i_t_min caleculer a partir de soc_min_i et p_d_bl_max_i
    s_i_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)] # Liste contenant s_zxi_max
    
    

    # Boucle principale
    #print("iteration 0")
    x_bar_0_score = objective_function(data, x_bar_k, s_i_t_min, s_i_max)
    #print(x_bar_0_score)
    for k in range(K):
        # Le gradient de (y - y^up)^2 -> 2(y - y^up) with y = x_bar_k["c_bl"]-x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"] et y^up = 
        #print(x_bar_k)
        lambda_k=np.array([0.]*T)
        #print(derivee)
        for t in range(T):
            ss=0
            for i in range(N):
                ss+=x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]-x_bar_k["c_up"][i][t]+x_bar_k["d_up"][i][t]
            lambda_k[t]=(ss/N-data["announced_capacity"]["up"][actual_time+t]/N)*2*alpha
        #print(np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"]\
        #                     ,axis=0)/N)
        #print((x_bar_k["c_bl"][0][0] - x_bar_k["d_bl"][0][0] - x_bar_k["c_up"][0][0] + x_bar_k["d_up"][0][0])/N)
        #lambda_k = (np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"]\
        #                     ,axis=0)/N-[x/N for x in data["announced_capacity"]["up"][actual_time:actual_time+T]])*2*alpha
        #print(derivee, lambda_k)
        # Resolution des sous-problemes
        for i in range(N):
            c_bl_i, d_bl_i, c_up_i, d_up_i ,s_bl_i, s_up_i, y = resolution_subpb(data, lambda_k,k,i, verbose=verbose)
            c_bl[i]=(c_bl_i)
            c_up[i]=(c_up_i)
            d_bl[i]=(d_bl_i)
            d_up[i]=(d_up_i)
            s_bl[i]=s_bl_i
            s_up[i]=s_up_i
        
        # Obtention de x_k a partir de x_bar_k
        x_k = {"c_bl": c_bl, "d_bl": d_bl, "c_up": c_up, "d_up": d_up, "s_bl": s_bl, "s_up": s_up}
        
        delta_k = 2/(k+2) 

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
                    sample_c_bl[i]=(x_bar_k["c_bl"][i])
                    sample_d_bl[i]=(x_bar_k["d_bl"][i])
                    sample_c_up[i]=(x_bar_k["c_up"][i])
                    sample_d_up[i]=(x_bar_k["d_up"][i])
                    sample_s_bl[i]=(x_bar_k["s_bl"][i])
                    sample_s_up[i]=(x_bar_k["s_up"][i])
                else:
                    sample_c_bl[i]=(c_bl[i])
                    sample_d_bl[i]=(d_bl[i])
                    sample_c_up[i]=(c_up[i])
                    sample_d_up[i]=(d_up[i])
                    sample_s_bl[i]=(s_bl[i])
                    sample_s_up[i]=(s_up[i])
            sample = {"c_bl": sample_c_bl, "d_bl": sample_d_bl, \
                      "c_up": sample_c_up, "d_up": sample_d_up,\
                      "s_bl": sample_s_bl, "s_up": sample_s_up}
            list_sample.append(sample)

        # Chercher la meilleur echantillon et calculer le score de x_bar_k, x_k et la meilleure echantillon
        best_sample = min(list_sample, key=lambda x: objective_function(data, x, s_i_t_min, s_i_max))
        best_sample_score= objective_function(data, best_sample, s_i_t_min, s_i_max)
        x_bar_k_score = objective_function(data, x_bar_k, s_i_t_min, s_i_max)
        x_k_score = objective_function(data, x_k, s_i_t_min, s_i_max)

        if verbose:
            print("best_sample_score",best_sample_score)
            print("x_bar_k_score",x_bar_k_score)
            print("x_k_score",x_k_score)
        """print("best_sample_score",best_sample_score)
        print("x_bar_k_score",x_bar_k_score)
        print("x_k_score",x_k_score)
        """
        print("============")
        print("lambda_k",lambda_k)
        print(np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"],axis=0)/N)
        print(np.sum(x_k["c_bl"] - x_k["d_bl"] - x_k["c_up"] + x_k["d_up"],axis=0)/N)
        print("============")
        gap_primal_dual = (np.dot(lambda_k,np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"],axis=0)/N\
                                -np.sum(x_k["c_bl"] - x_k["d_bl"] - x_k["c_up"] + x_k["d_up"],axis=0)/N))\
                                +function_h(x_bar_k, s_i_t_min, s_i_max)-function_h(x_k, s_i_t_min, s_i_max)
        if verbose:
            print("Gap primal dual : ",gap_primal_dual)
        # Mettre a jour x_bar_k en fonction des scores de x_bar_k, x_k et du meilleur echantillon
        if k!=0:
            if best_sample_score > x_bar_k_score or best_sample_score > x_k_score:
                if x_bar_k_score > x_k_score:
                    x_bar_k = (x_k)
                else:
                    x_bar_k = (x_bar_k)
            else:
                x_bar_k =(best_sample)
        else:
            x_bar_k = (best_sample)
        
        if verbose:
            print("after",objective_function(data, x_bar_k, s_i_t_min, s_i_max))
            print("====================")
        if (k+1)%10==0 :
            progress_bar.update(10)
        if analyse:
            if k==0:
                df = pd.DataFrame({"k": [k], "best_score":[min(best_sample_score,x_bar_k_score,x_k_score)], "charge":[x_bar_k["c_bl"]],"decharge":[x_bar_k["d_bl"]],"charge_up":[x_bar_k["c_up"]],"decharge_up":[x_bar_k["d_up"]],"soc":[x_bar_k["s_bl"]],"soc_up":[x_bar_k["s_up"]]})
            else:
                df = df.append({"k": k, "best_score":min(best_sample_score,x_bar_k_score,x_k_score), "charge":x_bar_k["c_bl"],"decharge":x_bar_k["d_bl"],"charge_up":x_bar_k["c_up"],"decharge_up":x_bar_k["d_up"],"soc":x_bar_k["s_bl"],"soc_up":x_bar_k["s_up"]}, ignore_index=True)
    progress_bar.close()
    print("Gap primal dual : ",gap_primal_dual)
    print("Objective value is : ",objective_function(data, x_bar_k, s_i_t_min, s_i_max))
    return x_bar_k,df
        

def objective_function(data, x, s_i_t_min, s_i_max):

    penality_SOC_fin = data["penalties"]["SOC_fin"]

    # Recuperer les valeurs des variables
    c_bl=x["c_bl"]
    d_bl=x["d_bl"]
    c_up=x["c_up"]
    d_up=x["d_up"]
    s_bl=x["s_bl"]
    s_up=x["s_up"]
    f_val= alpha*np.sum([ (np.sum([c_bl[i][t]-d_bl[i][t]-c_up[i][t]+d_up[i][t] for i in range(N) ])/N-data["announced_capacity"]["up"][actual_time+t]/N)**2 for t in range(T)  ])
    s_T=s_bl[:,T-1]
    #print("y_t : ",[np.sum([c_bl[i][t]-d_bl[i][t]-c_up[i][t]+d_up[i][t] for i in range(N) ])/N for t in range(T)])
    #print("y_t^up : ",[data["announced_capacity"]["up"][actual_time+t]/N for t in range(T)])
    soc_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(N)]
    # Calculer la fonction objectif
    cost = 0 # Valeur de la fonction objectif
    
    sum_cost=0
    sum_n_p=0
    for i in range(N):
        for t in range(actual_time,actual_time+T):
            neg_pos_part =  (beta_min*(-min(s_bl[i][t]-s_i_t_min[i][t],0))**2 + \
                  beta_max*max(s_bl[i][t]-s_i_max[i],0)**2 + \
                  beta_min*(-min(s_up[i][t]-s_i_t_min[i][t],0))**2+ \
                  beta_max*max(s_up[i][t]-s_i_max[i],0)**2)/N
            cost_electricity = (c_bl[i][t]-d_bl[i][t])*data["cost_of_electricity"][t]*data["time_mesh"]/60/N
            sum_cost+=cost_electricity
            sum_n_p+=neg_pos_part
            cost+=neg_pos_part+cost_electricity
        # We want the level of c_bl to be the high at the end of the optimization horizon
        exprs_final_soc=(soc_max[i]-s_T[i])*data["cost_of_electricity"][actual_time+T]*penality_SOC_fin/N 
        cost+=exprs_final_soc
    cost+=f_val
    """
    print("cost f",f_val)
    print("cost elec",sum_cost)
    print("cost neg_pos",sum_n_p)
    """
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
            "d_up": d_up, "s_bl": soc, \
            "s_up": soc_bar}

def function_h(x, s_i_t_min, s_i_max):
    neg_pos_part = np.sum([beta_min*(-min(x["s_bl"][i][t]-s_i_t_min[i][t],0))**2 + \
                    beta_max*max(x["s_bl"][i][t]-s_i_max[i],0)**2 + \
                    beta_min*(-min(x["s_up"][i][t]-s_i_t_min[i][t],0))**2+ \
                    beta_max*max(x["s_up"][i][t]-s_i_max[i],0)**2 for i in range(N) for t in range(T)])/N
    cost_eletricity = np.sum([(x["c_bl"][i][t]-x["d_bl"][i][t])*data["cost_of_electricity"][t]*data["time_mesh"]/60/N for i in range(N) for t in range(T)])
    exprs_final_soc=np.sum([(soc_max[i]-x["s_bl"][i][T-1])*data["cost_of_electricity"][T-1]*penality_SOC_fin/N for i in range(N)])
    return neg_pos_part+cost_eletricity+exprs_final_soc

def function_g(x):
    return np.sum([np.sum(x["c_bl"][i][t]-x["d_bl"][i][t]-x["c_up"][i][t]+x["d_up"][i][t] for i in range(N) ) for t in range(T)  ])/N
    
#FW_solve(data,actual_time)



