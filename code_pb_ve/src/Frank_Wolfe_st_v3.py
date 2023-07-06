import sys
import time
import numpy as np
import pandas as pd
sys.path.append('../data')
from tqdm import tqdm
from generator import *
from subpb_pl_cplex import *
from subpb_pl_sd_cplex_v2 import *
#from subpb_pl_sd_opt_cplex import *
from tools import *
from copy import deepcopy

#instance_json(3)


if __name__ == "__main__":
    my_instance = "../data/instance_10.json"
actual_time=0


# Afficher la fonction objective, gap primal-dual,evolution au cours du temps/iterations
# Afficher l'evolution de le moyen/std de la puissance de chargement a chaque pas de temps/ pour une vehicule choisi aleatoirement (Service/baseline)
# L'evolution de soc(service/baseline) pour une vehicule choisi aleatoirement 
# Comparer avec les differentes valeurs des parametres \beta \alpha \gamma \cost_elec

class Frank_Wolfe_Standard_v2:
    def __init__(self,my_instance):
        self.data=json.load(open(my_instance))
        self.N=len(self.data["evses"])
        self.T=self.data["optim_horizon"]
        self.beta_min=self.data["penalties"]["beta_min"]
        self.beta_max=self.data["penalties"]["beta_max"]
        self.alpha=self.data["penalties"]["fcr_up"]
        self.soc_max=[self.data["evses"][i]["SOC_max"]*self.data["evses"][i]["capacity"] for i in range(self.N)]
        self.time_mesh_to_hour = self.data["time_mesh"]/60
        self.s_final=np.array([self.data["evses"][i]["SOC_final"]*self.data["evses"][i]["capacity"] for i in range(self.N)]) #liste contenant soc_final_i
        self.s_i_t_min = np.array([[max(self.data["evses"][i]["SOC_min"]*self.data["evses"][i]["capacity"], (self.s_final[i]-self.data["evses"][i]["p_charge_max"]*(self.T-t)*self.time_mesh_to_hour) ) for t in range(1,self.T+1)] for i in range(self.N)]) # Liste contenant s_i_t_min caleculer a partir de soc_min_i et p_d_bl_max_i
        self.s_i_min=np.array([self.data["evses"][i]["SOC_min"]*self.data["evses"][i]["capacity"] for i in range(self.N)]) # Liste contenant s_i_min
        self.s_i_max=[self.data["evses"][i]["SOC_max"]*self.data["evses"][i]["capacity"] for i in range(self.N)] # Liste contenant s_zxi_max
        self.penality_SOC_fin = self.data["penalties"]["SOC_fin"]

    def initialisation_x(self):
        # Initialisation de x_bar_0
        c_bl=np.zeros((self.N,self.T))
        c_up=np.zeros((self.N,self.T))
        d_bl=np.zeros((self.N,self.T))
        d_up=np.zeros((self.N,self.T))
        u_bl=np.zeros((self.N,self.T))
        u_up=np.zeros((self.N,self.T))
        v_bl=np.zeros((self.N,self.T))
        v_up=np.zeros((self.N,self.T))
        soc=np.zeros((self.N,self.T))
        soc_bar=np.zeros((self.N,self.T))

        return {"c_bl": c_bl, "d_bl": d_bl, "c_up": c_up, \
                "d_up": d_up, "u_bl": u_bl, "u_up": u_up, \
                "v_bl": v_bl, "v_up": v_up, "s_bl": soc, "s_up": soc_bar}
    
    def function_h(self, x):

        neg_pos_part=0
        """neg_pos_part = np.sum([self.beta_min * (-min(x["s_bl"][i][t] - self.s_i_t_min[i][t],0))**2 + \
                        self.beta_max * max(x["s_bl"][i][t] - self.s_i_max[i],0)**2 + \
                        self.beta_min * (-min(x["s_up"][i][t] - self.s_i_t_min[i][t],0))**2 + \
                        self.beta_max * max(x["s_up"][i][t] - self.s_i_max[i],0)**2 for i in range(self.N) for t in range(self.T)])/self.N"""
        cost_eletricity = np.sum([(x["c_bl"][i][t]-x["d_bl"][i][t]) * self.data["cost_of_electricity"][t] * self.data["time_mesh"]/60 for i in range(self.N) for t in range(self.T)])/self.N
        exprs_final_soc=np.sum([(self.soc_max[i]-x["s_bl"][i][self.T-1])*self.data["cost_of_electricity"][self.T-1]*self.penality_SOC_fin for i in range(self.N)])/self.N
        return neg_pos_part+cost_eletricity+exprs_final_soc
    
    def objective_function(self, x):

        penality_SOC_fin = self.data["penalties"]["SOC_fin"]

        # Recuperer les valeurs des variables
        c_bl=x["c_bl"]
        d_bl=x["d_bl"]
        c_up=x["c_up"]
        d_up=x["d_up"]
        s_bl=x["s_bl"]
        s_up=x["s_up"]
        f_val= self.alpha*np.sum([ (np.sum([c_bl[i][t]-d_bl[i][t]-c_up[i][t]+d_up[i][t] for i in range(self.N) ])/self.N\
                                     - self.data["announced_capacity"]["up"][actual_time+t]/self.N)**2 for t in range(self.T)  ])
        s_T=s_bl[:,self.T-1]
        #print("y_t : ",[np.sum([c_bl[i][t]-d_bl[i][t]-c_up[i][t]+d_up[i][t] for i in range(N) ])/N for t in range(T)])
        #print("y_t^up : ",[data["announced_capacity"]["up"][actual_time+t]/N for t in range(T)])
        soc_max=[self.data["evses"][i]["SOC_max"]*self.data["evses"][i]["capacity"] for i in range(self.N)]
        # Calculer la fonction objectif
        cost = 0 # Valeur de la fonction objectif
        
        sum_cost=0
        sum_n_p=0
        for i in range(self.N):
            for t in range(actual_time, actual_time + self.T):
                neg_pos_part=0
                """neg_pos_part =  (self.beta_min * (-min(s_bl[i][t] - self.s_i_t_min[i][t],0))**2 + \
                    self.beta_max*max(s_bl[i][t] - self.s_i_max[i],0)**2 + \
                    self.beta_min*(-min(s_up[i][t] - self.s_i_t_min[i][t],0))**2+ \
                    self.beta_max*max(s_up[i][t] - self.s_i_max[i],0)**2)/self.N"""
                cost_electricity = (c_bl[i][t]-d_bl[i][t]) * self.data["cost_of_electricity"][t]*self.data["time_mesh"]/60/self.N
                sum_cost+=cost_electricity
                sum_n_p+=neg_pos_part
                cost+=neg_pos_part+cost_electricity
            # We want the level of c_bl to be the high at the end of the optimization horizon
            exprs_final_soc=(soc_max[i]-s_T[i]) * self.data["cost_of_electricity"][actual_time+self.T]*penality_SOC_fin/self.N 
            cost+=exprs_final_soc
        cost+=f_val
        return cost
    
    def solution_optimize(self, solution):
        borne_inf_i_t=np.zeros((self.N,self.T+1))
        borne_sup_i_t=np.zeros((self.N,self.T+1))
        y_t_inf=np.zeros((self.T+1))
        y_t_sup=np.zeros((self.T+1))
        new_solution=deepcopy(solution)
        #c_new_up=np.zeros((self.N,self.T))
        #d_new_up=np.zeros((self.N,self.T))
        diff_i_t=np.zeros((self.N,self.T+1))
        for i in range(self.N):
            borne_inf_i_t[i][1]=max((self.s_i_min[i]-self.data["evses"][i]["SOC_init"]*self.data["evses"][i]["capacity"])/self.time_mesh_to_hour,\
                                    solution["u_up"][i][0]*self.data["evses"][i]["p_charge_min"]-solution["v_up"][i][0]*self.data["evses"][i]["p_discharge_max"])
            borne_sup_i_t[i][1]=min((self.s_i_max[i]-self.data["evses"][i]["SOC_init"]*self.data["evses"][i]["capacity"])/self.time_mesh_to_hour,\
                                    solution["u_up"][i][0]*self.data["evses"][i]["p_charge_max"]-solution["v_up"][i][0]*self.data["evses"][i]["p_discharge_min"])
            for t in range(2,self.T+1):
                # s_i_t_min range (1, T+1) , solution["s_bl"] range (1,T+1)
                borne_inf_i_t[i][t]=max((self.s_i_min[i]-solution["s_bl"][i][t-2])/self.time_mesh_to_hour,\
                                        solution["u_up"][i][t-1]*self.data["evses"][i]["p_charge_min"]-solution["v_up"][i][t-1]*self.data["evses"][i]["p_discharge_max"])
                borne_sup_i_t[i][t]=min((self.s_i_max[i]-solution["s_bl"][i][t-2])/self.time_mesh_to_hour,\
                                        solution["u_up"][i][t-1]*self.data["evses"][i]["p_charge_max"]-solution["v_up"][i][t-1]*self.data["evses"][i]["p_discharge_min"])
        for t in range(1,self.T+1):
            #y_t_inf[t]=np.sum(solution["c_bl"][:,t-1]-solution["d_bl"][:,t-1])-np.sum(borne_sup_i_t[:,t])
            #y_t_sup[t]=np.sum(solution["c_bl"][:,t-1]-solution["d_bl"][:,t-1])-np.sum(borne_inf_i_t[:,t])
            for i in range(self.N):
                y_t_inf[t]+=solution["c_bl"][i,t-1]-solution["d_bl"][i,t-1]-borne_sup_i_t[i,t]
                y_t_sup[t]+=solution["c_bl"][i,t-1]-solution["d_bl"][i,t-1]-borne_inf_i_t[i,t]
            y_t_up=self.data["announced_capacity"]["up"][actual_time+t-1]
            """print("###########################")
            print("y_t_up",y_t_up)
            print("y_t_inf",y_t_inf[t])
            print("y_t_sup",y_t_sup[t])
            print("###########################")"""
            if y_t_inf[t]<y_t_up and y_t_sup[t]>y_t_up:
                beta=(y_t_up-y_t_inf[t])/(y_t_sup[t]-y_t_inf[t])
                #diff_i_t[:,t]=(1-beta)*borne_inf_i_t[:,t]+beta*borne_sup_i_t[:,t]
                diff_i_t[:,t]=(1-beta)*borne_sup_i_t[:,t]+beta*borne_inf_i_t[:,t]
                #d_new_up[:,t]=beta*x_bar_k["d_bl"][:,t]+(1-beta)*borne_sup_i_t[:,t]
            elif y_t_inf[t]>y_t_up :
                diff_i_t[:,t]=borne_sup_i_t[:,t]
            else:
                diff_i_t[:,t]=borne_inf_i_t[:,t]
        for i in range(self.N):
            for t in range(self.T):    
                if solution["u_up"][i][t]==1:
                    new_solution["c_up"][i][t]=diff_i_t[i][t+1]
                    new_solution["d_up"][i][t]=0
                else:
                    new_solution["c_up"][i][t]=0
                    new_solution["d_up"][i][t]=-diff_i_t[i][t+1]
        return new_solution

    def FW_st_solve(self, actual_time, verbose=False, analyse=False, K=100, nk=50, n_pre=20, gap_calculate=False, optimize=True):
        """!!! gap primal-duale"""
        data=self.data
        # Initialisation
        df=0
        x_bar_0=self.initialisation_x() # Initialisation de x_bar_0
        K=K#max(2*N,100) # Nb d'itérations
        n_pre=n_pre
        nk=[nk]*int(K) # Nb de tirages 
        
        progress_bar = tqdm(total=K, unit='iteration')

        # Initialisation des np.array pour stocker les valeurs de x_bar_k
        c_bl=np.zeros((self.N,self.T))
        c_up=np.zeros((self.N,self.T))
        d_bl=np.zeros((self.N,self.T))
        d_up=np.zeros((self.N,self.T))
        u_bl=np.zeros((self.N,self.T))
        u_up=np.zeros((self.N,self.T))
        v_bl=np.zeros((self.N,self.T))
        v_up=np.zeros((self.N,self.T))
        s_bl=np.zeros((self.N,self.T))
        s_up=np.zeros((self.N,self.T))

        x_bar_k=x_bar_0

        # Boucle principale
        x_k = {"c_bl": c_bl, "d_bl": d_bl, "c_up": c_up, "d_up": d_up, "u_bl": u_bl, "v_bl": v_bl, "u_up": u_up, "v_up": v_up, "s_bl": s_bl, "s_up": s_up}
        for k in range(n_pre):
            print("Iteration : ",k)
            # Le gradient de (y - y^up)^2 -> 2(y - y^up) with y = x_bar_k["c_bl"]-x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"] et y^up = 
            lambda_k=np.array([0.]*self.T)
            for t in range(self.T):
                ss=0
                for i in range(self.N):
                    ss+=x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]-x_bar_k["c_up"][i][t]+x_bar_k["d_up"][i][t]
                lambda_k[t]=2*self. alpha*(ss/self.N-self.data["announced_capacity"]["up"][actual_time+t]/self.N)
            
            #lambda_k = (np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"]\
            #                     ,axis=0)/N-[x/N for x in data["announced_capacity"]["up"][actual_time:actual_time+T]])*2*alpha
            # Resolution des sous-problemes
            if gap_calculate==True:
                for i in range(self.N):
                    c_bl_i, d_bl_i, c_up_i, d_up_i ,u_bl_i, u_up_i, v_bl_i, v_up_i, s_bl_i, s_up_i, y = resolution_subpb(data, lambda_k,i,x_bar_k, verbose=verbose)
                    c_bl[i]=c_bl_i
                    c_up[i]=c_up_i
                    d_bl[i]=d_bl_i
                    d_up[i]=d_up_i
                    s_bl[i]=s_bl_i
                    s_up[i]=s_up_i
                    u_bl[i]=(u_bl_i)
                    u_up[i]=(u_up_i)
                    v_bl[i]=(v_bl_i)
                    v_up[i]=(v_up_i)
            # Obtention de x_k a partir de x_bar_k
            
            
            delta_k = 2/(k+2) 

            # Tirage de nk[k] echantillons
            list_sample = []
            done=[0]*self.N
            #print(k,len(nk))
            for _ in range(nk[k]):
                sample_c_bl=np.zeros((self.N,self.T))
                sample_d_bl=np.zeros((self.N,self.T))
                sample_c_up=np.zeros((self.N,self.T))
                sample_d_up=np.zeros((self.N,self.T))
                sample_u_bl=np.zeros((self.N,self.T))
                sample_v_bl=np.zeros((self.N,self.T))
                sample_u_up=np.zeros((self.N,self.T))
                sample_v_up=np.zeros((self.N,self.T))
                sample_s_bl=np.zeros((self.N,self.T))
                sample_s_up=np.zeros((self.N,self.T))
               
                for i in range(self.N):
                    
                    draw=np.random.rand() # Tirer au sort un valeur entre 0 et 1
                    # Si draw <= 1-delta_k alors on prend x_bar_k sinon on prend x_k
                    if draw<=1-delta_k:
                        sample_c_bl[i]=(x_bar_k["c_bl"][i])
                        sample_d_bl[i]=(x_bar_k["d_bl"][i])
                        sample_c_up[i]=(x_bar_k["c_up"][i])
                        sample_d_up[i]=(x_bar_k["d_up"][i])
                        sample_u_bl[i]=(x_bar_k["u_bl"][i])
                        sample_v_bl[i]=(x_bar_k["v_bl"][i])
                        sample_u_up[i]=(x_bar_k["u_up"][i])
                        sample_v_up[i]=(x_bar_k["v_up"][i])
                        sample_s_bl[i]=(x_bar_k["s_bl"][i])
                        sample_s_up[i]=(x_bar_k["s_up"][i])
                    else:
                        if gap_calculate==False:
                            if done[i]==0:
                                    done[i]=1
                                    c_bl_i, d_bl_i, c_up_i, d_up_i ,u_bl_i, u_up_i, v_bl_i, v_up_i, s_bl_i, s_up_i, y = resolution_subpb(data, lambda_k,i,x_bar_k, verbose=verbose)
                                    c_bl[i]=c_bl_i
                                    c_up[i]=c_up_i
                                    d_bl[i]=d_bl_i
                                    d_up[i]=d_up_i
                                    s_bl[i]=s_bl_i
                                    s_up[i]=s_up_i
                                    u_bl[i]=(u_bl_i)
                                    u_up[i]=(u_up_i)
                                    v_bl[i]=(v_bl_i)
                                    v_up[i]=(v_up_i)                     
                        sample_c_bl[i]=(c_bl[i])
                        sample_d_bl[i]=(d_bl[i])
                        sample_c_up[i]=(c_up[i])
                        sample_d_up[i]=(d_up[i])
                        sample_s_bl[i]=(s_bl[i])
                        sample_s_up[i]=(s_up[i])
                        sample_u_bl[i]=(u_bl[i])
                        sample_v_bl[i]=(v_bl[i])
                        sample_u_up[i]=(u_up[i])
                        sample_v_up[i]=(v_up[i])

                sample = {"c_bl": sample_c_bl, "d_bl": sample_d_bl, \
                        "c_up": sample_c_up, "d_up": sample_d_up,\
                        "u_bl": sample_u_bl, "v_bl": sample_v_bl, \
                            "u_up": sample_u_up, "v_up": sample_v_up,\
                            "s_bl": sample_s_bl, "s_up": sample_s_up}
                if optimize==True:
                    """print("==================iteration",k,"==================")
                    self.affichage(sample)
                    print(self.objective_function(sample))
                    print("---------------------------------------")"""
                    sample=self.solution_optimize(sample)
                    """self.affichage(sample)
                    print(self.objective_function(sample))"""
                list_sample.append(sample)

            # Chercher la meilleur echantillon et calculer le score de x_bar_k, x_k et la meilleure echantillon
            best_sample = min(list_sample, key=lambda x: self.objective_function(x))
            best_sample_score= self.objective_function(best_sample)
            x_bar_k_score = self.objective_function(x_bar_k)
            x_k_score = self.objective_function(x_k)

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
            gap_primal_dual = (np.dot(lambda_k,np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] \
                                                        + x_bar_k["d_up"],axis=0)/self.N\
                            -np.sum(x_k["c_bl"] - x_k["d_bl"] - x_k["c_up"] + x_k["d_up"],axis=0)/self.N))\
                            +self.function_h(x_bar_k) - self.function_h(x_k)
            #print(gap_primal_dual)
            if verbose:
                print("best_sample_score",best_sample_score)
                print("x_bar_k_score",x_bar_k_score)
                print("x_k_score",x_k_score)
                print("after",self.objective_function(x_bar_k))
                print("====================")
            if (k+1)%10==0 :
                progress_bar.update(10)

            if analyse:
                if k==0:
                    df = pd.DataFrame({"k": [k], "best_score":[min(best_sample_score,x_bar_k_score,x_k_score)], "charge":[x_bar_k["c_bl"]],"decharge":[x_bar_k["d_bl"]],"charge_up":[x_bar_k["c_up"]],"decharge_up":[x_bar_k["d_up"]],"soc":[x_bar_k["s_bl"]],"soc_up":[x_bar_k["s_up"]],"gap_primal_dual":[gap_primal_dual]})
                else:
                    df = df.append({"k": k, "best_score":min(best_sample_score,x_bar_k_score,x_k_score), "charge":x_bar_k["c_bl"],"decharge":x_bar_k["d_bl"],"charge_up":x_bar_k["c_up"],"decharge_up":x_bar_k["d_up"],"soc":x_bar_k["s_bl"],"soc_up":x_bar_k["s_up"], "gap_primal_dual":gap_primal_dual}, ignore_index=True)
        update_solution=True
        for k in range(n_pre, K):
            lambda_k=np.array([0.]*self.T)
            #print(derivee)
            for t in range(self.T):
                ss=0
                for i in range(self.N):
                    ss+=x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]-x_bar_k["c_up"][i][t]+x_bar_k["d_up"][i][t]
                lambda_k[t]=2*self.alpha*(ss/self.N-data["announced_capacity"]["up"][actual_time+t]/self.N)
            
            #lambda_k = (np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"]\
            #                     ,axis=0)/N-[x/N for x in data["announced_capacity"]["up"][actual_time:actual_time+T]])*2*alpha
            # Resolution des sous-problemes
            if update_solution==True:
                for i in range(self.N):
                    c_bl_i, d_bl_i, c_up_i, d_up_i, u_bl_i, v_bl_i, u_up_i, v_up_i ,s_bl_i, s_up_i, y = resolution_subpb_st(data, lambda_k,i,x_bar_k, verbose=verbose)
                    c_bl[i]=(c_bl_i)
                    c_up[i]=(c_up_i)
                    d_bl[i]=(d_bl_i)
                    d_up[i]=(d_up_i)
                    u_bl[i]=(u_bl_i)
                    u_up[i]=(u_up_i)
                    v_bl[i]=(v_bl_i)
                    v_up[i]=(v_up_i)
                    s_bl[i]=(s_bl_i)
                    s_up[i]=(s_up_i)
                    
            # Obtention de x_k a partir de x_bar_k
            x_k = {"c_bl": c_bl, "d_bl": d_bl, "c_up": c_up, "d_up": d_up, "u_bl": u_bl, "v_bl": v_bl, "u_up": u_up, "v_up": v_up, "s_bl": s_bl, "s_up": s_up}
            delta_k = 2/(100+k+2)
            gap_primal_dual = (np.dot(lambda_k,np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] \
                                                      + x_bar_k["d_up"],axis=0)/self.N\
                        -np.sum(x_k["c_bl"] - x_k["d_bl"] - x_k["c_up"] + x_k["d_up"],axis=0)/self.N))\
                        +self.function_h(x_bar_k) - self.function_h(x_k)
           
            x_bar_k_old=deepcopy(x_bar_k)
            for i in range(self.N):
                x_bar_k["c_bl"][i]=(1-delta_k)*x_bar_k["c_bl"][i]+delta_k*x_k["c_bl"][i]
                x_bar_k["d_bl"][i]=(1-delta_k)*x_bar_k["d_bl"][i]+delta_k*x_k["d_bl"][i]
                x_bar_k["c_up"][i]=(1-delta_k)*x_bar_k["c_up"][i]+delta_k*x_k["c_up"][i]
                x_bar_k["d_up"][i]=(1-delta_k)*x_bar_k["d_up"][i]+delta_k*x_k["d_up"][i]
                x_bar_k["s_bl"][i]=(1-delta_k)*x_bar_k["s_bl"][i]+delta_k*x_k["s_bl"][i]
                x_bar_k["s_up"][i]=(1-delta_k)*x_bar_k["s_up"][i]+delta_k*x_k["s_up"][i]
                # u, v are fixed

            if update_solution==True and optimize==True:
                x_bar_k=self.solution_optimize(x_bar_k)
            score_old = self.objective_function(x_bar_k_old)
            score_new = self.objective_function(x_bar_k)

            if score_old<score_new:
                x_bar_k=deepcopy(x_bar_k_old)
                update_solution=False
            else:
                print("Gap primal dual : "+str(k),(np.dot(lambda_k,np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] \
                                                      + x_bar_k["d_up"],axis=0)/self.N\
                        -np.sum(x_k["c_bl"] - x_k["d_bl"] - x_k["c_up"] + x_k["d_up"],axis=0)/self.N)),self.function_h(x_bar_k) - self.function_h(x_k))
                print("Objective value is : ",score_new)
                update_solution=True

            if verbose:
                print("Gap primal dual : ",gap_primal_dual)

            score= self.objective_function(x_bar_k)
            if (k+1)%10==0 :
                progress_bar.update(10)
            if analyse:
                df = df.append({"k": k, "best_score":score, "charge":x_bar_k["c_bl"],"decharge":x_bar_k["d_bl"],"charge_up":x_bar_k["c_up"],"decharge_up":x_bar_k["d_up"],"soc":x_bar_k["s_bl"],"soc_up":x_bar_k["s_up"], "gap_primal_dual":gap_primal_dual}, ignore_index=True)
            
        progress_bar.close()
        print("Gap primal dual : ",gap_primal_dual)
        print("Objective value is : ",self.objective_function(x_bar_k))
        return x_bar_k,df
    
    def affichage(self, x_bar_k):
        print("alpha:", self.alpha, "Beta_min:", self.beta_min, "Beta_max:", self.beta_max, "Gamma:", self.penality_SOC_fin)
        print("------------------------------------------------------")
        print("t\t","charge net bl\t","charge net up\t", "y_t^up/N \t", "Ecart de Service")
        for t in range(self.T):
            print(t,"\t",round(np.sum([x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t] for i in range(self.N)])/self.N,0),\
                "\t",round(np.sum([x_bar_k["c_up"][i][t]-x_bar_k["d_up"][i][t] for i in range(self.N)])/self.N,0),"\t",\
                    round(self.data["announced_capacity"]["up"][actual_time+t]/self.N,0),"\t",round(np.sum([x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]-x_bar_k["c_up"][i][t]+x_bar_k["d_up"][i][t] for i in range(self.N)])/self.N - self.data["announced_capacity"]["up"][actual_time+t]/self.N,0))
        #for t in range(T):
        x=x_bar_k
        cost=0
        c_bl=x["c_bl"]
        d_bl=x["d_bl"]
        c_up=x["c_up"]
        d_up=x["d_up"]
        s_bl=x["s_bl"]
        s_up=x["s_up"]
        f_val= self.alpha*np.sum([ (np.sum([c_bl[i][t]-d_bl[i][t]-c_up[i][t]+d_up[i][t] for i in range(self.N) ])/self.N - self.data["announced_capacity"]["up"][actual_time+t]/self.N)**2 for t in range(self.T)  ])
        sum_neg_pos_part=0
        sum_cost_electricity=0
        sum_soc=0
        s_T=s_bl[:,self.T-1]
        soc_max=[self.data["evses"][i]["SOC_max"] * self.data["evses"][i]["capacity"] for i in range(self.N)]
        for i in range(self.N):
            for t in range(actual_time,actual_time + self.T):
                sum_neg_pos_part=0
                """sum_neg_pos_part+=  (self.beta_min * (-min(s_bl[i][t] - self.s_i_t_min[i][t],0))**2 + \
                    self.beta_max * max(s_bl[i][t] - self.s_i_max[i],0)**2 + \
                    self.beta_min * (-min(s_up[i][t] - self.s_i_t_min[i][t],0))**2+ \
                    self.beta_max * max(s_up[i][t] - self.s_i_max[i],0)**2)/self.N"""
                sum_cost_electricity+= (c_bl[i][t] - d_bl[i][t]) * self.data["cost_of_electricity"][t] * self.data["time_mesh"] / 60 /self.N
                
            # We want the level of c_bl to be the high at the end of the optimization horizon
            #print("s_T[i]",s_T[i])
            #print("s_i_max[i]",soc_max[i])
            sum_soc+=(soc_max[i] - s_T[i]) * self.data["cost_of_electricity"][actual_time + self.T] * self.penality_SOC_fin / self.N 
        print("------------------------------------------------------")
        print("f_val:",f_val, "neg_pos_part:",sum_neg_pos_part, "cost_electricity:",sum_cost_electricity, "soc:",sum_soc)
        print("value of the objective function:",f_val+sum_neg_pos_part+sum_cost_electricity+sum_soc)
        print("------------------------------------------------------")
        print("t\t","s_i_min\t","s_i_T\t", "s_i_max")
        for i in range(self.N):
            print("------------------------------------------------------")
            print("Vehicle ",i,"\t","S_i_T:",s_T[i],"\t")
            print("S_i_min:", self.s_i_t_min[i])
            print("S_i_max:", self.s_i_max[i])