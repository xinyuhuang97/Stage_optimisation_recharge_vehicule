import numpy as np

class Tools:
    @staticmethod
    def generate_matrix(m: int, n: int, low: float, high: float)->np.ndarray:
        a=np.random.uniform(low=low, high=high, size=(m,n))
        print(a, a.size)
        return np.random.uniform(low=low, high=high, size=(m,n))
