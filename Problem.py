from Tools import *

# Mixed-interger linear quadratic problem
class MIQP:
    def __init__(self, m: int = 5, n: int = 5, low: float = 0, high: float = 10):
        self.A = Tools.generate_matrix(m, n, low, high)
