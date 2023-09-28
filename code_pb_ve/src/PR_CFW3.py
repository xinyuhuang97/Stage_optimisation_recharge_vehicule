import sys
import time
import numpy as np
import pandas as pd
sys.path.append('../data')
from tqdm import tqdm
from generator import *
from subpb_pl_reduced_cplex import *
from subpb_pl_reduced_sd_cplex import *
from delta_optimize_pb_rd_v2 import *
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

class Frank_Wolfe_Reduced_standard_v3:
    def __init__(self,my_instance):
        self.instance=my_instance
        self.algo_name="PR_CFW3"
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
        d_bl=np.zeros((self.N,self.T))
        u_bl=np.zeros((self.N,self.T))
        u_up=np.zeros((self.N,self.T))
        v_bl=np.zeros((self.N,self.T))
        v_up=np.zeros((self.N,self.T))
        s_bl=np.zeros((self.N,self.T))
        z_1=np.zeros((self.N,self.T))
        z_2=np.zeros((self.N,self.T))
        #soc_bar=np.zeros((self.N,self.T))

        return {"c_bl": c_bl, "d_bl": d_bl, "u_bl": u_bl, "u_up": u_up, \
                "v_bl": v_bl, "v_up": v_up, "s_bl": s_bl, "z_1": z_1, "z_2": z_2}
    
    def function_h(self, x):

        cost_eletricity = np.sum([(x["c_bl"][i][t]-x["d_bl"][i][t]) * self.data["cost_of_electricity"][t] * self.data["time_mesh"]/60 for i in range(self.N) for t in range(self.T)])/self.N
        exprs_final_soc=np.sum([(self.soc_max[i]-x["s_bl"][i][self.T-1])*self.data["cost_of_electricity"][self.T-1]*self.penality_SOC_fin for i in range(self.N)])/self.N
        return cost_eletricity+exprs_final_soc
    
    def objective_function(self, x):

        penality_SOC_fin = self.data["penalties"]["SOC_fin"]

        # Recuperer les valeurs des variables
        c_bl=x["c_bl"]
        d_bl=x["d_bl"]
        s_bl=x["s_bl"]
        #s_up=x["s_up"]
        z_1=x["z_1"]
        z_2=x["z_2"]
        y_1=[np.sum([c_bl[i][t]-d_bl[i][t]+z_1[i][t] for i in range(self.N)  ])/self.N for t in range(self.T)]
        y_2=[np.sum([c_bl[i][t]-d_bl[i][t]-z_2[i][t] for i in range(self.N)  ])/self.N for t in range(self.T)]
        f_val=0
        
        for t in range(1,self.T+1):
            if self.data["announced_capacity"]["up"][actual_time+t]/self.N<y_1[t-1]:
                f_val+=(y_1[t-1]-self.data["announced_capacity"]["up"][actual_time+t]/self.N)**2*self.alpha
            elif self.data["announced_capacity"]["up"][actual_time+t]/self.N>y_2[t-1]:
                f_val+=(self.data["announced_capacity"]["up"][actual_time+t]/self.N-y_2[t-1])**2*self.alpha
        s_T=s_bl[:,self.T-1]
        
        soc_max=[self.data["evses"][i]["SOC_max"]*self.data["evses"][i]["capacity"] for i in range(self.N)]
        # Calculer la fonction objectif
        cost = 0 # Valeur de la fonction objectif
        
        sum_cost=0
        sum_n_p=0
        for i in range(self.N):
            for t in range(actual_time, actual_time + self.T):
                neg_pos_part=0
                cost_electricity = (c_bl[i][t]-d_bl[i][t]) * self.data["cost_of_electricity"][t]*self.data["time_mesh"]/60/self.N
                sum_cost+=cost_electricity
                sum_n_p+=neg_pos_part
                cost+=neg_pos_part+cost_electricity
            # We want the level of c_bl to be the high at the end of the optimization horizon
            exprs_final_soc=(soc_max[i]-s_T[i]) * self.data["cost_of_electricity"][actual_time+self.T]*penality_SOC_fin/self.N 
            cost+=exprs_final_soc
        cost+=f_val
        return cost

    def solve(self, actual_time, verbose=False, analyse=False, K=100, nk=50, n_pre=20, gap_calculate=False, optimize=True, active_criterion="iterations", max_time=3600):
        """!!! gap primal-duale"""
        start_time = time.time()
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
        #c_up=np.zeros((self.N,self.T))
        d_bl=np.zeros((self.N,self.T))
        #d_up=np.zeros((self.N,self.T))
        u_bl=np.zeros((self.N,self.T))
        u_up=np.zeros((self.N,self.T))
        v_bl=np.zeros((self.N,self.T))
        v_up=np.zeros((self.N,self.T))
        s_bl=np.zeros((self.N,self.T))
        z_1=np.zeros((self.N,self.T))
        z_2=np.zeros((self.N,self.T))


        x_bar_k=x_bar_0

        # Boucle principale
        x_k = {"c_bl": c_bl, "d_bl": d_bl, "u_bl": u_bl, "v_bl": v_bl, "u_up": u_up, "v_up": v_up, "s_bl": s_bl, "z_1": z_1, "z_2": z_2}
        update_solution=True
        for k in range(n_pre):
            print("=========iteration : ",k,"===========")
            # Le gradient de (y - y^up)^2 -> 2(y - y^up) with y = x_bar_k["c_bl"]-x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"] et y^up = 
            lambda_1_k=np.array([0.]*self.T)
            lambda_2_k=np.array([0.]*self.T)
            for t in range(self.T):
                y_1_t=0
                y_2_t=0
                for i in range(self.N):
                    y_1_t+=(x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]+x_bar_k["z_1"][i][t])/self.N
                    y_2_t+=(x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]-x_bar_k["z_2"][i][t])/self.N
                if y_1_t>data["announced_capacity"]["up"][actual_time+t]/self.N:
                    lambda_1_k[t]=2*self.alpha*(y_1_t-data["announced_capacity"]["up"][actual_time+t]/self.N)
                elif y_2_t<data["announced_capacity"]["up"][actual_time+t]/self.N:
                    lambda_2_k[t]=2*self.alpha*(y_2_t-data["announced_capacity"]["up"][actual_time+t]/self.N)

            if update_solution==True:
                done=[0]*self.N

            #lambda_k = (np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] + x_bar_k["d_up"]\
            #                     ,axis=0)/N-[x/N for x in data["announced_capacity"]["up"][actual_time:actual_time+T]])*2*alpha
            # Resolution des sous-problemes
            if gap_calculate==True and update_solution==True:
                for i in range(self.N):
                    c_bl_i, d_bl_i, u_bl_i, u_up_i, v_bl_i, v_up_i, s_bl_i,  y, z_1_i, z_2_i = resolution_subpb(data, lambda_1_k, lambda_2_k,i,x_bar_k, verbose=verbose)
                    c_bl[i]=c_bl_i
                    d_bl[i]=d_bl_i
                    s_bl[i]=s_bl_i
                    u_bl[i]=u_bl_i
                    u_up[i]=u_up_i
                    v_bl[i]=v_bl_i
                    v_up[i]=v_up_i
                    z_1[i]=z_1_i
                    z_2[i]=z_2_i
            # Obtention de x_k a partir de x_bar_k
            
            delta_k = 2/(k+2) 
            #delta_k = 2/(k+2+100) 

            # Tirage de nk[k] echantillons
            list_sample = []
            
            #print(k,len(nk))
            for _ in range(nk[k]):
                sample_c_bl=np.zeros((self.N,self.T))
                sample_d_bl=np.zeros((self.N,self.T))
                sample_u_bl=np.zeros((self.N,self.T))
                sample_v_bl=np.zeros((self.N,self.T))
                sample_u_up=np.zeros((self.N,self.T))
                sample_v_up=np.zeros((self.N,self.T))
                sample_s_bl=np.zeros((self.N,self.T))
                sample_z_1=np.zeros((self.N,self.T))
                sample_z_2=np.zeros((self.N,self.T))
               
                for i in range(self.N):
                    
                    draw=np.random.rand() # Tirer au sort un valeur entre 0 et 1
                    # Si draw <= 1-delta_k alors on prend x_bar_k sinon on prend x_k
                    if draw<=1-delta_k:
                        sample_c_bl[i]=(x_bar_k["c_bl"][i])
                        sample_d_bl[i]=(x_bar_k["d_bl"][i])
                        sample_u_bl[i]=(x_bar_k["u_bl"][i])
                        sample_v_bl[i]=(x_bar_k["v_bl"][i])
                        sample_u_up[i]=(x_bar_k["u_up"][i])
                        sample_v_up[i]=(x_bar_k["v_up"][i])
                        sample_s_bl[i]=(x_bar_k["s_bl"][i])
                        sample_z_1[i]=(x_bar_k["z_1"][i])
                        sample_z_2[i]=(x_bar_k["z_2"][i])
                    else:
                        if gap_calculate==False:
                            if done[i]==0:
                                    done[i]=1
                                    c_bl_i, d_bl_i, u_bl_i, u_up_i, v_bl_i, v_up_i, s_bl_i,  y, z_1_i, z_2_i = resolution_subpb(data, lambda_1_k, lambda_2_k,i,x_bar_k, verbose=verbose)
                                    c_bl[i]=c_bl_i
                                    d_bl[i]=d_bl_i
                                    s_bl[i]=s_bl_i
                                    u_bl[i]=u_bl_i
                                    u_up[i]=u_up_i
                                    v_bl[i]=v_bl_i
                                    v_up[i]=v_up_i
                                    z_1[i]=z_1_i
                                    z_2[i]=z_2_i 
                        sample_c_bl[i]=c_bl[i]
                        sample_d_bl[i]=d_bl[i]
                        sample_u_bl[i]=u_bl[i]
                        sample_v_bl[i]=v_bl[i]
                        sample_u_up[i]=u_up[i]
                        sample_v_up[i]=v_up[i]
                        sample_s_bl[i]=s_bl[i]
                        sample_z_1[i]=z_1[i]
                        sample_z_2[i]=z_2[i]
                sample = {"c_bl": sample_c_bl, "d_bl": sample_d_bl, \
                        "u_bl": sample_u_bl, "v_bl": sample_v_bl, \
                            "u_up": sample_u_up, "v_up": sample_v_up,\
                            "s_bl": sample_s_bl, "z_1": sample_z_1, "z_2": sample_z_2}
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
                        update_solution=True
                    else:
                        x_bar_k = (x_bar_k)
                        update_solution=False
                else:
                    x_bar_k =(best_sample)
                    update_solution=True
            else:
                x_bar_k = (best_sample)

            sum_x_bar_k_1=[]
            sum_x_k_1=[]
            sum_x_bar_k_2=[]
            sum_x_k_2=[]
            for t in range(self.T):
                sum_x_bar_k_i_1=0
                sum_x_k_i_1=0
                sum_x_bar_k_i_2=0
                sum_x_k_i_2=0
                for i in range(self.N):
                    sum_x_bar_k_i_1+=x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]+x_bar_k["z_1"][i][t]
                    sum_x_k_i_1+=x_k["c_bl"][i][t]-x_k["d_bl"][i][t]+x_k["z_1"][i][t]
                    sum_x_bar_k_i_2+=x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]-x_bar_k["z_2"][i][t]
                    sum_x_k_i_2+=x_k["c_bl"][i][t]-x_k["d_bl"][i][t]-x_k["z_2"][i][t]
                sum_x_bar_k_1.append(sum_x_bar_k_i_1/self.N)
                sum_x_k_1.append(sum_x_k_i_1/self.N)
                sum_x_bar_k_2.append(sum_x_bar_k_i_2/self.N)
                sum_x_k_2.append(sum_x_k_i_2/self.N)
            print("lambda",lambda_1_k,lambda_2_k)
            print(np.dot(lambda_1_k,np.array(sum_x_bar_k_1)-np.array(sum_x_k_1) ), np.dot(lambda_2_k,np.array(sum_x_bar_k_2)-np.array(sum_x_k_2) ),self.function_h(x_bar_k)-self.function_h(x_k))
            gap_primal_dual = np.dot(lambda_1_k,np.array(sum_x_bar_k_1)-np.array(sum_x_k_1) )+np.dot(lambda_2_k,np.array(sum_x_bar_k_2)-np.array(sum_x_k_2) )\
                +self.function_h(x_bar_k)-self.function_h(x_k)
            print("gap_primal_dual",gap_primal_dual)

            print("after",self.objective_function(x_bar_k))
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
                    df = pd.DataFrame({"k": [k], "best_score":[min(best_sample_score,x_bar_k_score,x_k_score)], "charge":[x_bar_k["c_bl"]],"decharge":[x_bar_k["d_bl"]],"soc":[x_bar_k["s_bl"]],"gap_primal_dual":[gap_primal_dual],"time":[time.time()-start_time]})
                else:
                    df = df.append({"k": k, "best_score":min(best_sample_score,x_bar_k_score,x_k_score), "charge":x_bar_k["c_bl"],"decharge":x_bar_k["d_bl"],"soc":x_bar_k["s_bl"],"gap_primal_dual":gap_primal_dual,"time":time.time()-start_time}, ignore_index=True)#,"gap_primal_dual":gap_primal_dual}, ignore_index=True)
        update_solution=True

        #start_time = time.time()
        k=n_pre-1
        while True:
            k += 1
            if active_criterion == "iterations":
                if k == K:
                    break
            elif active_criterion == "time":
                if time.time() - start_time > max_time:
                    break
        #for k in range(n_pre, K):
            lambda_1_k=np.array([0.]*self.T)
            lambda_2_k=np.array([0.]*self.T)
            #print(derivee)
            for t in range(self.T):
                y_1_t=0
                y_2_t=0
                for i in range(self.N):
                    y_1_t+=(x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]+x_bar_k["z_1"][i][t])/self.N
                    y_2_t+=(x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]-x_bar_k["z_2"][i][t])/self.N
                if y_1_t>data["announced_capacity"]["up"][actual_time+t]/self.N:
                    lambda_1_k[t]=2*self.alpha*(y_1_t-data["announced_capacity"]["up"][actual_time+t]/self.N)
                elif y_2_t<data["announced_capacity"]["up"][actual_time+t]/self.N:
                    lambda_2_k[t]=2*self.alpha*(y_2_t-data["announced_capacity"]["up"][actual_time+t]/self.N)
            
    

            #start = time.time()
            if update_solution==True:
                for i in range(self.N):
                    c_bl_i, d_bl_i, u_bl_i, u_up_i, v_bl_i, v_up_i, s_bl_i,  y, z_1_i, z_2_i = resolution_subpb_st(data, lambda_1_k,lambda_2_k,i,x_bar_k, verbose=verbose)
                    c_bl[i]=c_bl_i
                    d_bl[i]=d_bl_i
                    s_bl[i]=s_bl_i
                    u_bl[i]=u_bl_i
                    u_up[i]=u_up_i
                    v_bl[i]=v_bl_i
                    v_up[i]=v_up_i
                    z_1[i]=z_1_i
                    z_2[i]=z_2_i
            #end = time.time()
            #print("Sous-problems time : ",end-start)
                    
            # Obtention de x_k a partir de x_bar_k
            x_k = {"c_bl": c_bl, "d_bl": d_bl, "u_bl": u_bl, "v_bl": v_bl, "u_up": u_up, "v_up": v_up, "s_bl": s_bl, "z_1": z_1, "z_2": z_2}
            #start = time.time()
            
            delta_k = delta_optimize_pb_rd_v2(x_k, x_bar_k, self.data, actual_time)
            #delta_k = 2/(100+k+2)
            #print(delta_k)
            x_bar_k_old=deepcopy(x_bar_k)
            print("=========iteration : ",k,"===========")
            #print("before combinaison",self.objective_function(x_bar_k))
            for i in range(self.N):
                x_bar_k["c_bl"][i]=(1-delta_k[i])*x_bar_k["c_bl"][i]+delta_k[i]*x_k["c_bl"][i]
                x_bar_k["d_bl"][i]=(1-delta_k[i])*x_bar_k["d_bl"][i]+delta_k[i]*x_k["d_bl"][i]
                x_bar_k["s_bl"][i]=(1-delta_k[i])*x_bar_k["s_bl"][i]+delta_k[i]*x_k["s_bl"][i]
                x_bar_k["z_1"][i]=(1-delta_k[i])*x_bar_k["z_1"][i]+delta_k[i]*x_k["z_1"][i]
                x_bar_k["z_2"][i]=(1-delta_k[i])*x_bar_k["z_2"][i]+delta_k[i]*x_k["z_2"][i]
                # u, v are fixed

            """if update_solution==True and optimize==True:
                print("-------------booster-------------")
                print("before",self.objective_function(x_bar_k))
                x_bar_k=self.solution_optimize(x_bar_k)
                print("after",self.objective_function(x_bar_k))
            """
            #end = time.time()
            #print("Booster&optimisation delta time : ",end-start)
            score_old = self.objective_function(x_bar_k_old)
            score_new = self.objective_function(x_bar_k)

            if score_old<score_new:
                x_bar_k=deepcopy(x_bar_k_old)
                update_solution=False
                #print("no improvement, iteration : ",k)
                #print("Gap primal dual : "+str(k),(np.dot(lambda_k,np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] \
                #                                      + x_bar_k["d_up"],axis=0)/self.N\
                #        -np.sum(x_k["c_bl"] - x_k["d_bl"] - x_k["c_up"] + x_k["d_up"],axis=0)/self.N)),self.function_h(x_bar_k) - self.function_h(x_k))
                #print("Old Objective value is : ",score_old)
                #print("New Objective value is : ",score_new)

                #print("====================================")
            else:
                """print("Gap primal dual : "+str(k),(np.dot(lambda_k,np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] \
                                                      + x_bar_k["d_up"],axis=0)/self.N\
                        -np.sum(x_k["c_bl"] - x_k["d_bl"] - x_k["c_up"] + x_k["d_up"],axis=0)/self.N)),self.function_h(x_bar_k) - self.function_h(x_k))
                print("Objective value is : ",score_new)
                """
                update_solution=True
            #print()
            gap_primal_dual=0

            sum_x_bar_k_1=[]
            sum_x_k_1=[]
            sum_x_bar_k_2=[]
            sum_x_k_2=[]
            for t in range(self.T):
                sum_x_bar_k_i_1=0
                sum_x_k_i_1=0
                sum_x_bar_k_i_2=0
                sum_x_k_i_2=0
                for i in range(self.N):
                    sum_x_bar_k_i_1+=x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]+x_bar_k["z_1"][i][t]
                    sum_x_k_i_1+=x_k["c_bl"][i][t]-x_k["d_bl"][i][t]+x_k["z_1"][i][t]
                    sum_x_bar_k_i_2+=x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]-x_bar_k["z_2"][i][t]
                    sum_x_k_i_2+=x_k["c_bl"][i][t]-x_k["d_bl"][i][t]-x_k["z_2"][i][t]
                sum_x_bar_k_1.append(sum_x_bar_k_i_1/self.N)
                sum_x_k_1.append(sum_x_k_i_1/self.N)
                sum_x_bar_k_2.append(sum_x_bar_k_i_2/self.N)
                sum_x_k_2.append(sum_x_k_i_2/self.N)
            #print("lambda",lambda_1_k,lambda_2_k)
            #print(np.dot(lambda_1_k,np.array(sum_x_bar_k_1)-np.array(sum_x_k_1) ), np.dot(lambda_2_k,np.array(sum_x_bar_k_2)-np.array(sum_x_k_2) ),self.function_h(x_bar_k)-self.function_h(x_k))
            gap_primal_dual = np.dot(lambda_1_k,np.array(sum_x_bar_k_1)-np.array(sum_x_k_1) )+np.dot(lambda_2_k,np.array(sum_x_bar_k_2)-np.array(sum_x_k_2) )\
                +self.function_h(x_bar_k)-self.function_h(x_k)
            print("score",max(score_new,score_old))
            #print("gap_primal_dual",gap_primal_dual)
            """gap_primal_dual = (np.dot(lambda_k,np.sum(x_bar_k["c_bl"] - x_bar_k["d_bl"] - x_bar_k["c_up"] \
                                                      + x_bar_k["d_up"],axis=0)/self.N\
                        -np.sum(x_k["c_bl"] - x_k["d_bl"] - x_k["c_up"] + x_k["d_up"],axis=0)/self.N))\
                        +self.function_h(x_bar_k) - self.function_h(x_k)
            """
            #print("delta",delta_k)
            if verbose:
                print("Gap primal dual : ",gap_primal_dual)

            score= self.objective_function(x_bar_k)
            if (k+1)%10==0 :
                progress_bar.update(10)
            time_elapsed = time.time() - start_time
            if analyse:
                df = df.append({"time_elapsed": time_elapsed,"k": k, "best_score":score, "charge":x_bar_k["c_bl"],"decharge":x_bar_k["d_bl"],"soc":x_bar_k["s_bl"],"gap_primal_dual":gap_primal_dual,"time":time.time()-start_time}, ignore_index=True)
        progress_bar.close()
        self.save_data(df, n_pre, K)
        #print("Gap primal dual : ",gap_primal_dual)
        print("Objective value is : ",self.objective_function(x_bar_k))
        return x_bar_k,df
    
    def affichage(self, x_bar_k):
        print("alpha:", self.alpha, "Beta_min:", self.beta_min, "Beta_max:", self.beta_max, "Gamma:", self.penality_SOC_fin)
        print("------------------------------------------------------")
        print("t\t","y_1\t","y_2\t", "y_t^up/N \t")#, "Ecart de Service")
        for t in range(self.T):
            print(t,"\t",round(np.sum([x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t] +x_bar_k["z_1"][i][t] for i in range(self.N)])/self.N,0),\
                "\t",round(np.sum([x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t] -x_bar_k["z_2"][i][t] for i in range(self.N)])/self.N,0),"\t",\
                    round(self.data["announced_capacity"]["up"][actual_time+t]/self.N,0))
        #for t in range(T):
        x=x_bar_k
        cost=0
        c_bl=x["c_bl"]
        d_bl=x["d_bl"]
        s_bl=x["s_bl"]
        f_val=0
        y_1=[np.sum([x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t] +x_bar_k["z_1"][i][t] for i in range(self.N)])/self.N for t in range(self.T)]
        y_2=[np.sum([x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t] -x_bar_k["z_2"][i][t] for i in range(self.N)])/self.N for t in range(self.T)]
        for t in range(1,self.T+1):
            if self.data["announced_capacity"]["up"][actual_time+t]/self.N<y_1[t-1]:
                f_val+=(y_1[t-1]-self.data["announced_capacity"]["up"][actual_time+t]/self.N)**2*self.alpha
            elif self.data["announced_capacity"]["up"][actual_time+t]/self.N>y_2[t-1]:
                f_val+=(self.data["announced_capacity"]["up"][actual_time+t]/self.N-y_2[t-1])**2*self.alpha
        sum_neg_pos_part=0
        sum_cost_electricity=0
        sum_soc=0
        s_T=s_bl[:,self.T-1]
        soc_max=[self.data["evses"][i]["SOC_max"] * self.data["evses"][i]["capacity"] for i in range(self.N)]
        for i in range(self.N):
            for t in range(actual_time,actual_time + self.T):
                sum_neg_pos_part=0
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
    def save_data(self, df, n_pre, n_iteration):
        instance = self.instance.split("/")[-1]
        instance_name = instance.replace(".json", "")
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        folder_path = os.path.join(parent_directory+"/result/pb_reduced/", instance_name)
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{instance_name}' created successfully at '{folder_path}'")
        filename="../result/pb_reduced/" + "/"+instance_name+"/" + "PR_CFW3_"+ str(n_pre) + "_" + str(n_iteration) + "_" + time.strftime("%Y%m%d-%H%M%S") + ".json"
        df.to_json(filename, orient="records")
