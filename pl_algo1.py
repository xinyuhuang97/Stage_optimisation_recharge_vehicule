from gurobipy import *
from typing import Tuple
from typing import List
from Problem import *
import numpy as np


# Programme pour resoudre le problem de Kang
def pl_algo1(x_bar_0: List[float], pb: MIQP) ->List[float]:
    # Input : x_bar_0 -> la liste des etats intials des agents
    #        pb -> le problem etuidie
    # Output : la liste des etats des agents generee

    A = pb.A # La matrice genere
    y_bar = x_bar_0 # La liste des etats

    dim_M, dim_N = A.shape # la dimension de la matrice

    lignes = dim_M
    colonnes = dim_N

    m = Model("example_kang")

    # declaration variables de decision
    x=[]
    for i in range(colonnes):
        x.append(m.addVar(vtype=GRB.BINARY, lb=0, name="x%d" % (i+1)))

    m.update()


    #obj = LinExpr()
    obj = QuadExpr()

    # declaration de la fonction objective
    for j in range(lignes):
        sum_Ax = 0
        for i in range(colonnes):
            sum_Ax += A[j][i]*x[i]
        obj += ((sum_Ax - y_bar[j])/colonnes)**2

    m.setObjective(obj,GRB.MINIMIZE)
    m.optimize()

    print('Solution optimale:')
    x_opt = [var.x for var in x]
    print(x_opt)
    print('Valeur de la fonction objectif :', m.objVal)

    return np.array(x_opt)


"""
p = MIQP(3,4,0,3)
pl_algo1([1,4,3,2],p)
"""
