import sys
import time
import numpy as np
import pandas as pd
import json
sys.path.append('../data')
from tqdm import tqdm
from generator import *
from tools import *
sys.path.append('/Applications/CPLEX_Studio221/cplex/python/3.7/x86-64_osx/cplex/_internal')
sys.path.insert(0, "/Applications/CPLEX_Studio221/cplex/python/3.7/x86-64_osx")
import cplex


from tools import *
if __name__ == "__main__":
    my_instance = "../data/instance_10.json"

actual_time=0



class Frontal:
    def __init__(self, my_instance, actual_time, verbose=False):
        self.data = json.load(open(my_instance))
        self.instance=my_instance
        self.N = len(self.data["evses"])
        self.T = self.data["optim_horizon"]
        self.beta_min = self.data["penalties"]["beta_min"]
        self.beta_max = self.data["penalties"]["beta_max"]
        self.alpha = self.data["penalties"]["fcr_up"]

        self.soc_init = [self.data["evses"][i]["SOC_init"]*self.data["evses"][i]["capacity"] for i in range(self.N)]
        self.soc_max = [self.data["evses"][i]["SOC_max"]*self.data["evses"][i]["capacity"] for i in range(self.N)]
        self.soc_min = [self.data["evses"][i]["SOC_min"]*self.data["evses"][i]["capacity"] for i in range(self.N)]

        self.p_charge_max = [self.data["evses"][i]["p_charge_max"] for i in range(self.N)]
        self.p_charge_min = [self.data["evses"][i]["p_charge_min"] for i in range(self.N)]
        self.p_discharge_max = [self.data["evses"][i]["p_discharge_max"] for i in range(self.N)]
        self.p_discharge_min = [self.data["evses"][i]["p_discharge_min"] for i in range(self.N)]

        self.time_mesh_to_hour = self.data["time_mesh"]/60
        self.s_final = [self.data["evses"][i]["SOC_final"]*self.data["evses"][i]["capacity"] for i in range(self.N)]
        self.s_i_t_min = [[max(self.data["evses"][i]["SOC_min"]*self.data["evses"][i]["capacity"],self.s_final[i]-self.p_discharge_max[i]*(self.T-t)*self.time_mesh_to_hour ) for t in range(1,self.T+1)] for i in range(self.N)]
        self.s_i_max = [self.data["evses"][i]["SOC_max"]*self.data["evses"][i]["capacity"] for i in range(self.N)]
        self.cost_electricity = self.data["cost_of_electricity"]
        self.penality_SOC_fin = self.data["penalties"]["SOC_fin"]

    def Frontal_solve(self, actual_time, verbose=False):

        # Load data
        data = self.data
        file_name="../log/"+(self.instance.split("/")[-1]).replace(".json","")+"/model"+".lp"
        
        # Create a new LP problem
        problem = cplex.Cplex()

        """
        if not verbose:
            problem.set_log_stream(None)
            problem.set_error_stream(None)
            problem.set_warning_stream(None)
            problem.set_results_stream(None)
        """
        # Prepare data
        """T=data["optim_horizon"]
        N=len(data["evses"])
        beta_min=data["penalties"]["beta_min"]
        beta_max=data["penalties"]["beta_max"]
        alpha=data["penalties"]["fcr_up"]

        soc_init=[data["evses"][i]["SOC_init"]*data["evses"][i]["capacity"] for i in range(self.N)]
        soc_max=[data["evses"][i]["SOC_max"]*data["evses"][i]["capacity"] for i in range(self.N)]
        soc_min=[data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"] for i in range(self.N)]

        p_charge_max=[data["evses"][i]["p_charge_max"] for i in range(self.N)]
        p_charge_min=[data["evses"][i]["p_charge_min"] for i in range(self.N)]
        p_discharge_max=[data["evses"][i]["p_discharge_max"] for i in range(self.N)]
        p_discharge_min=[data["evses"][i]["p_discharge_min"] for i in range(self.N)]
        
        time_mesh_to_hour = data["time_mesh"]/60
        s_final = [data["evses"][i]["SOC_final"]*data["evses"][i]["capacity"] for i in range(self.N)]
        s_t_min = [[max(data["evses"][i]["SOC_min"]*data["evses"][i]["capacity"],s_final[i]-p_discharge_max[i]*(T-t)*time_mesh_to_hour ) for t in range(1,self.T+1)] for i in range(self.N)]

        cost_electricity = data["cost_of_electricity"]
        penality_SOC_fin = data["penalties"]["SOC_fin"]"""

        # Create variables
        s_bl = [["s_bl_" + str(i) +"_" +str(t) for t in range(self.T+1)]  for i in range(self.N) ]
        c_bl = [["c_bl_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        d_bl = [["d_bl_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        u_bl = [["u_bl_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        v_bl = [["v_bl_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        #n_bl = [["n_bl_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        #p_bl = [["p_bl_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]

        s_up = [["s_up_" + str(i) +"_" +str(t) for t in range(self.T+1)]  for i in range(self.N) ]
        c_up = [["c_up_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        d_up = [["d_up_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        u_up = [["u_up_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        v_up = [["v_up_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        #n_up = [["n_up_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]
        #p_up = [["p_up_" + str(i) +"_" +str(t) for t in range(1,self.T+1)]for i in range(self.N) ]

        y = ["y_" + str(t) for t in range(1,self.T+1)]
        problem.variables.add(names=y,lb=[-cplex.infinity]*(self.T),ub=[cplex.infinity]*(self.T))
        for i in range(self.N):
            problem.variables.add(names=s_bl[i])
            problem.variables.add(names=s_up[i])
            problem.variables.add(names=c_bl[i])
            problem.variables.add(names=c_up[i])
            problem.variables.add(names=d_bl[i])
            problem.variables.add(names=d_up[i])
            problem.variables.add(types=['B']*self.T,names=u_bl[i])
            problem.variables.add(types=['B']*self.T,names=u_up[i])
            problem.variables.add(types=['B']*self.T,names=v_bl[i])
            problem.variables.add(types=['B']*self.T,names=v_up[i])
            #problem.variables.add(names=n_bl[i])
            #problem.variables.add(names=n_up[i])
            #problem.variables.add(names=p_bl[i])
            #problem.variables.add(names=p_up[i])
        
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(0)],val=[1])],senses=["E"],rhs=[self.soc_init[i]],names=["Initial SOC bl "+str(i)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(0)],val=[1])],senses=["E"],rhs=[self.soc_init[i]],names=["Initial SOC up "+str(i)])

            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[self.soc_max[i] for _ in range(1,self.T+1)],names=["SOC max bl "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[self.soc_max[i] for _ in range(1,self.T+1)],names=["SOC max up "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,self.T+1)],senses=["G"]*self.T,rhs=[self.soc_min[i] for _ in range(1,self.T+1)],names=["SOC min bl "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,self.T+1)],senses=["G"]*self.T,rhs=[self.soc_min[i] for _ in range(1,self.T+1)],names=["SOC min up "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bl_"+str(i)+"_"+str(t),"v_bl_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[1]*self.T,names=["charge or discharge bl "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_up_"+str(i)+"_"+str(t),"v_up_"+str(i)+"_"+str(t)],val=[1,1]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[1]*self.T,names=["charge or discharge up "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_bl_"+str(i)+"_"+str(t),"u_bl_"+str(i)+"_"+str(t)],val=[1, -self.p_charge_max[i]]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[0]*self.T,names=["charge upper bound bl "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["c_up_"+str(i)+"_"+str(t),"u_up_"+str(i)+"_"+str(t)],val=[1, -self.p_charge_max[i]]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[0]*self.T,names=["charge upper bound up "+str(i)+" "+str(t) for t in range(1,self.T+1)])

            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_bl_"+str(i)+"_"+str(t),"v_bl_"+str(i)+"_"+str(t)],val=[1, -self.p_discharge_max[i]]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[0]*self.T,names=["discharge upper bound bl "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d_up_"+str(i)+"_"+str(t),"v_up_"+str(i)+"_"+str(t)],val=[1, -self.p_discharge_max[i]]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[0]*self.T,names=["discharge upper bound up "+str(i)+" "+str(t) for t in range(1,self.T+1)])

            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_bl_"+str(i)+"_"+str(t),"c_bl_"+str(i)+"_"+str(t)],val=[self.p_charge_min[i],-1]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[0]*self.T,names=["charge lower bound bl "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["u_up_"+str(i)+"_"+str(t),"c_up_"+str(i)+"_"+str(t)],val=[self.p_charge_min[i],-1]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[0]*self.T,names=["charge lower bound up "+str(i)+" "+str(t) for t in range(1,self.T+1)])

            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_bl_"+str(i)+"_"+str(t),"d_bl_"+str(i)+"_"+str(t)],val=[self.p_discharge_min[i],-1]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[0]*self.T,names=["discharge lower bound bl "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["v_up_"+str(i)+"_"+str(t),"d_up_"+str(i)+"_"+str(t)],val=[self.p_discharge_min[i],-1]) for t in range(1,self.T+1)],senses=["L"]*self.T,rhs=[0]*self.T,names=["discharge lower bound up "+str(i)+" "+str(t) for t in range(1,self.T+1)])
            
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t+1),"s_bl_"+str(i)+"_"+str(t),"c_bl_"+str(i)+"_"+str(t+1),"d_bl_"+str(i)+"_"+str(t+1)],\
                                                                    val=[1,-1,-self.time_mesh_to_hour,self.time_mesh_to_hour]) for t in range(self.T)],senses=["E"]*(self.T),rhs=[0]*(self.T),names=["Production balance bl"+str(i)+" "+str(t) for t in range(self.T)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t+1),"s_bl_"+str(i)+"_"+str(t),"c_up_"+str(i)+"_"+str(t+1),"d_up_"+str(i)+"_"+str(t+1)],\
                                                                    val=[1,-1,-self.time_mesh_to_hour,self.time_mesh_to_hour]) for t in range(self.T)],senses=["E"]*(self.T),rhs=[0]*(self.T),names=["Production balance up"+str(i)+" "+str(t) for t in range(self.T)])

            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_bl_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,self.T+1)], senses=["G"]*(self.T), rhs=[self.s_i_t_min[i][t] for t in range(self.T)])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_up_"+str(i)+"_"+str(t)],val=[1]) for t in range(1,self.T+1)], senses=["G"]*(self.T), rhs=[self.s_i_t_min[i][t] for t in range(self.T)])

        
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["y_"+str(t)]+[x for i in range(self.N) for x in ["c_bl_"+str(i)+"_"+str(t),"c_up_"+str(i)+"_"+str(t),"d_bl_"+str(i)+"_"+str(t),"d_up_"+str(i)+"_"+str(t)]],\
                                                                val=[1]+[x for _ in range(self.N) for x in [-1/self.N,1/self.N,1/self.N,-1/self.N]]) for t in range(1,self.T+1)],senses=["E"]*self.T,\
                                                                rhs=[-x/self.N for x in data["announced_capacity"]["up"][actual_time:actual_time+self.T]],names=["y_t facilitate calculations"+str(t) for t in range(1,self.T+1)])

        

        # Set the objective function
        problem.objective.set_sense(problem.objective.sense.minimize)

        # Add function f 
        for t in range(1,self.T+1):
            problem.objective.set_quadratic_coefficients("y_"+str(t),"y_"+str(t),2*self.alpha)

        # Add terms concerning negative/positive parts
        """
        for i in range(self.N):
            for t in range(1,self.T+1):
                problem.objective.set_quadratic_coefficients("n_bl_"+str(i)+"_"+str(t),"n_bl_"+str(i)+"_"+str(t),2*self.beta_min/self.N)
                problem.objective.set_quadratic_coefficients("n_up_"+str(i)+"_"+str(t),"n_up_"+str(i)+"_"+str(t),2*self.beta_min/self.N)
                problem.objective.set_quadratic_coefficients("p_bl_"+str(i)+"_"+str(t),"p_bl_"+str(i)+"_"+str(t),2*self.beta_max/self.N)
                problem.objective.set_quadratic_coefficients("p_up_"+str(i)+"_"+str(t),"p_up_"+str(i)+"_"+str(t),2*self.beta_max/self.N)"""
        
        # Add electricity cost 
        exprs_electricity_cost = [x for i in range(self.N) for t in range(1,self.T+1) for x in ["c_bl_"+str(i)+"_"+str(t),"d_bl_"+str(i)+"_"+str(t)]]
        val_electricity_cost = [x for _ in range(self.N) for t in range(1,self.T+1) for x in [self.time_mesh_to_hour*self.cost_electricity[t]/self.N,-self.time_mesh_to_hour*self.cost_electricity[t]/self.N]]
        
        # Add reward on final soc
        exprs_final_soc = ["s_bl_"+str(i)+"_"+str(self.T) for i in range(self.N)]
        val_final_soc = [-self.penality_SOC_fin*self.cost_electricity[self.T]/self.N]*self.N

        
        problem.objective.set_linear(zip(exprs_electricity_cost+exprs_final_soc,val_electricity_cost+val_final_soc))

        problem.write(file_name)
        problem.solve()
        
        #print("Valeur optimal retourne par cplex",problem.solution.get_objective_value()) 

        sample_c_bl=np.zeros((self.N,self.T))
        sample_d_bl=np.zeros((self.N,self.T))
        sample_c_up=np.zeros((self.N,self.T))
        sample_d_up=np.zeros((self.N,self.T))
        sample_s_bl=np.zeros((self.N,self.T))
        sample_s_up=np.zeros((self.N,self.T))

        for i in range(self.N):
            sample_c_bl[i]=(problem.solution.get_values(c_bl[i]))
            sample_d_bl[i]=(problem.solution.get_values(d_bl[i]))
            sample_c_up[i]=(problem.solution.get_values(c_up[i]))
            sample_d_up[i]=(problem.solution.get_values(d_up[i]))
            sample_s_bl[i]=(problem.solution.get_values(s_bl[i][1:]))
            sample_s_up[i]=(problem.solution.get_values(s_up[i][1:]))

        #print_objective_function(file_name)
        result ={"c_bl": sample_c_bl, "d_bl": sample_d_bl, \
                "c_up": sample_c_up, "d_up": sample_d_up,\
                "s_bl": sample_s_bl, "s_up": sample_s_up}
        save ={"c_bl": sample_c_bl.tolist(), "d_bl": sample_d_bl.tolist(), \
                "c_up": sample_c_up.tolist(), "d_up": sample_d_up.tolist(),\
                "s_bl": sample_s_bl.tolist(), "s_up": sample_s_up.tolist()}
        #print("Valeur optimal retourne par la fonction",objective_function(data,result,s_t_min,soc_max))
        #self.save_cplex_data(save)
        print("valeur optimal par function objective:", self.objective_function(result))
        return result,self.s_i_t_min,self.soc_max,self.objective_function(result)


    def objective_function(self, x):

        penality_SOC_fin = self.penality_SOC_fin

        # Recuperer les valeurs des variables
        #print(x.keys())
        c_bl=x["c_bl"]
        d_bl=x["d_bl"]
        c_up=x["c_up"]
        d_up=x["d_up"]
        s_bl=x["s_bl"]
        s_up=x["s_up"]
        f_val= self.alpha*np.sum([ (np.sum([c_bl[i][t]-d_bl[i][t]-c_up[i][t]+d_up[i][t] for i in range(self.N) ])/self.N - self.data["announced_capacity"]["up"][actual_time+t]/self.N)**2 for t in range(self.T)  ])
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
    
    def affichage(self, x_bar_k):
        print("alpha:", self.alpha, "Beta_min:", self.beta_min, "Beta_max:", self.beta_max, "Gamma:", self.penality_SOC_fin)
        print("------------------------------------------------------")
        print("t\t","charge net bl\t","charge net up\t", "y_t^up/N \t", "Ecart de Service")
        for t in range(self.T):
            """print(t,"\t",round(np.sum([x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t] for i in range(self.N)])/self.N,0),\
                "\t",round(np.sum([x_bar_k["c_up"][i][t]-x_bar_k["d_up"][i][t] for i in range(self.N)])/self.N,0),"\t",\
                    round(self.data["announced_capacity"]["up"][actual_time+t]/self.N,0),"\t",round(np.sum([x_bar_k["c_bl"][i][t]-x_bar_k["d_bl"][i][t]-x_bar_k["c_up"][i][t]+x_bar_k["d_up"][i][t] for i in range(self.N)])/self.N - self.data["announced_capacity"]["up"][actual_time+t]/self.N,0))"""
            c_bl_sum = np.sum([np.array(x_bar_k["c_bl"][i][t]) - np.array(x_bar_k["d_bl"][i][t]) for i in range(self.N)])
            c_up_sum = np.sum([np.array(x_bar_k["c_up"][i][t]) - np.array(x_bar_k["d_up"][i][t]) for i in range(self.N)])
            
            ecart_service = np.sum([
                np.array(x_bar_k["c_bl"][i][t]) - np.array(x_bar_k["d_bl"][i][t]) -
                np.array(x_bar_k["c_up"][i][t]) + np.array(x_bar_k["d_up"][i][t]) for i in range(self.N)])/self.N
            
            y_t_up = self.data["announced_capacity"]["up"][actual_time + t] / self.N
            
            print(t, "\t", round(c_bl_sum / self.N, 0), "\t", round(c_up_sum / self.N, 0),
                "\t", round(y_t_up, 0), "\t", round(ecart_service - y_t_up))
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
        s_T=[s_bl[i][self.T-1] for i in range(self.N)]
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
            #print(soc_max[i],s_T[i])
            sum_soc+=(soc_max[i] - s_T[i]) * self.data["cost_of_electricity"][actual_time + self.T] * self.penality_SOC_fin / self.N 
        print(s_T)
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


#x,s_i_t_min,s_i_max =Frontal(data, actual_time, verbose=False)
#print("valeur optimal retourne par la fonction",objective_function(data,x,s_i_t_min,s_i_max))
