from algo_FW import *



pb= MIQP(3,4,0,3)
F=Frank_wolfe()
print(F.FW_algo(np.array([1.,4.,3.,2.]),pb))
