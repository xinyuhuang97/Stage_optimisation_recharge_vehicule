


import cplex
T=10
problem = cplex.Cplex()
problem.variables.add(lb=[0 for _ in range(T)], ub=[10 for _ in range(T)], names=["s_"+str(i) for i in range(T)])
#problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_"+str(i)], val=[1]) for i in range(1,T)],senses=["LE" for _ in range(1,T)],rhs=[5 for _ in range(1,T)])
problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["s_"+str(i)], val=[1]) for i in range(1,T)],senses=["L" for _ in range(1,T)],rhs=[5 for _ in range(1,T)])

problem.objective.set_sense(problem.objective.sense.minimize)
problem.objective.set_linear("s_0", 1)
problem.solve()