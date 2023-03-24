from typing import Tuple
from typing import List
from Problem import *
from pl_algo1 import *
import numpy as np
class Frank_wolfe:

    # "algo 1"
    def FW_algo(self, x_bar_0: List[float], pb: MIQP) -> List[float]:
        # Input : x_bar_0 -> la liste des etats intials des agents
        #        pb -> le problem etuidie
        # Output : la liste des etats des agents qui optimise le problem

        K=1e3 # nb d'iteration
        p=MIQP(3,4,0,3) # problem du Kang

        x_bar_k = x_bar_0
        for k in range(int(K)):
            x_k = self.resolution_pl_algo1(x_bar_k,pb)
            delta_k = 2/(k+2)
            x_bar_kp1 = (1-delta_k)*x_bar_k + delta_k*x_k
        return x_bar_kp1

    def resolution_pl_algo1(self, x_bar_k: List[float], pb:MIQP)->List[float]:

        return pl_algo1(x_bar_k,pb)

    """
    def stochastic_FW_algo(self, mu_bar_0: List[List[float]], list_nk: List[int]) -> List[List[float]]:

        #Input : mu_bar_0 -> le produit des distributions de proba initial
        #        list_n_k -> une sequence des valeurs entieres(nb de tirage pour la phase selection)

        K = 1e3 # nb d'iterations
        mu_bar_k = mu_bar_0 # intialisation de mu_bar_k
        for k in range(K):
            mu_k = self.resolution_pl(mu_bar_k) # resoudre le programme lineaire
            delta_k = 2/(k+2)
            mu_bar_kp1 = (1-delta_k)*mu_bar_k + delta_k*mu_k
            x_chap_kp1 = self.selection(mu_bar_kp1, list_nk[k])
            mu_bar_kp1 = dirac(x_chap_kp1)



    def resolution_pl(self, mu_bar_k: List[List[float]])->Tuple[List[List[float]], float]:
        #Input : mu_bar_0 -> le produit des distributions de probabilité initiales
        #        list_n_k -> une sequence des valeurs entieres(nb de tirage pour la phase selection)

        pass

    def selection(self, mu_bar_kp1: List[List[float]], n_kp1: int)->List[float]: #List[float]?
        pass

    def dirac():
        pass
    """

"""pb= MIQP(3,4,0,3)
F=Frank_wolfe()
print(F.FW_algo(np.array([1.,4.,3.,2.]),pb))"""
