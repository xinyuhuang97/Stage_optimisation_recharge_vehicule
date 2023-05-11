\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: - 1000 s_bl_4 + 0.025 c_bl_1 + 0.025 c_bl_2 + 0.025 c_bl_3
       + 0.025 c_bl_4 - 0.025 d_bl_1 - 0.025 d_bl_2 - 0.025 d_bl_3
       - 0.025 d_bl_4 + [ 2 n_bl_1 ^2 + 2 n_bl_2 ^2 + 2 n_bl_3 ^2 + 2 n_bl_4 ^2
       + 2 n_up_1 ^2 + 2 n_up_2 ^2 + 2 n_up_3 ^2 + 2 n_up_4 ^2 + 2 p_bl_1 ^2
       + 2 p_bl_2 ^2 + 2 p_bl_3 ^2 + 2 p_bl_4 ^2 + 2 p_up_1 ^2 + 2 p_up_2 ^2
       + 2 p_up_3 ^2 + 2 p_up_4 ^2 ] / 2
Subject To
 c1:  s_bl_0  = 9292.79839346397
 c2:  s_up_0  = 9292.79839346397
 c3:  s_bl_1 <= 30000
 c4:  s_bl_2 <= 30000
 c5:  s_bl_3 <= 30000
 c6:  s_bl_4 <= 30000
 c7:  s_up_1 <= 30000
 c8:  s_up_2 <= 30000
 c9:  s_up_3 <= 30000
 c10: s_up_4 <= 30000
 c11: s_bl_1 >= 0
 c12: s_bl_2 >= 0
 c13: s_bl_3 >= 0
 c14: s_bl_4 >= 0
 c15: s_up_1 >= 0
 c16: s_up_2 >= 0
 c17: s_up_3 >= 0
 c18: s_up_4 >= 0
 c19: u_bl_1 + v_bl_1 <= 1
 c20: u_bl_2 + v_bl_2 <= 1
 c21: u_bl_3 + v_bl_3 <= 1
 c22: u_bl_4 + v_bl_4 <= 1
 c23: u_up_1 + v_up_1 <= 1
 c24: u_up_2 + v_up_2 <= 1
 c25: u_up_3 + v_up_3 <= 1
 c26: u_up_4 + v_up_4 <= 1
 c27: c_bl_1 - 15000 u_bl_1 <= 0
 c28: c_bl_2 - 15000 u_bl_2 <= 0
 c29: c_bl_3 - 15000 u_bl_3 <= 0
 c30: c_bl_4 - 15000 u_bl_4 <= 0
 c31: c_up_1 - 15000 u_up_1 <= 0
 c32: c_up_2 - 15000 u_up_2 <= 0
 c33: c_up_3 - 15000 u_up_3 <= 0
 c34: c_up_4 - 15000 u_up_4 <= 0
 c35: d_bl_1 - 15000 v_bl_1 <= 0
 c36: d_bl_2 - 15000 v_bl_2 <= 0
 c37: d_bl_3 - 15000 v_bl_3 <= 0
 c38: d_bl_4 - 15000 v_bl_4 <= 0
 c39: d_up_1 - 15000 v_up_1 <= 0
 c40: d_up_2 - 15000 v_up_2 <= 0
 c41: d_up_3 - 15000 v_up_3 <= 0
 c42: d_up_4 - 15000 v_up_4 <= 0
 c43: - c_bl_1 + 1000 u_bl_1 <= 0
 c44: - c_bl_2 + 1000 u_bl_2 <= 0
 c45: - c_bl_3 + 1000 u_bl_3 <= 0
 c46: - c_bl_4 + 1000 u_bl_4 <= 0
 c47: - c_up_1 + 1000 u_up_1 <= 0
 c48: - c_up_2 + 1000 u_up_2 <= 0
 c49: - c_up_3 + 1000 u_up_3 <= 0
 c50: - c_up_4 + 1000 u_up_4 <= 0
 c51: - d_bl_1 + 1000 v_bl_1 <= 0
 c52: - d_bl_2 + 1000 v_bl_2 <= 0
 c53: - d_bl_3 + 1000 v_bl_3 <= 0
 c54: - d_bl_4 + 1000 v_bl_4 <= 0
 c55: - d_up_1 + 1000 v_up_1 <= 0
 c56: - d_up_2 + 1000 v_up_2 <= 0
 c57: - d_up_3 + 1000 v_up_3 <= 0
 c58: - d_up_4 + 1000 v_up_4 <= 0
 c59: - s_bl_0 + s_bl_1 - 0.25 c_bl_1 + 0.25 d_bl_1  = 0
 c60: - s_bl_1 + s_bl_2 - 0.25 c_bl_2 + 0.25 d_bl_2  = 0
 c61: - s_bl_2 + s_bl_3 - 0.25 c_bl_3 + 0.25 d_bl_3  = 0
 c62: - s_bl_3 + s_bl_4 - 0.25 c_bl_4 + 0.25 d_bl_4  = 0
 c63: - s_bl_0 + s_up_1 - 0.25 c_up_1 + 0.25 d_up_1  = 0
 c64: - s_bl_1 + s_up_2 - 0.25 c_up_2 + 0.25 d_up_2  = 0
 c65: - s_bl_2 + s_up_3 - 0.25 c_up_3 + 0.25 d_up_3  = 0
 c66: - s_bl_3 + s_up_4 - 0.25 c_up_4 + 0.25 d_up_4  = 0
 c67: s_bl_1 + n_bl_1 >= 0
 c68: s_bl_2 + n_bl_2 >= 0
 c69: s_bl_3 + n_bl_3 >= 0
 c70: s_bl_4 + n_bl_4 >= 13709.6308843775
 c71: s_up_1 + n_up_1 >= 0
 c72: s_up_2 + n_up_2 >= 0
 c73: s_up_3 + n_up_3 >= 0
 c74: s_up_4 + n_up_4 >= 13709.6308843775
 c75: - s_bl_1 + p_bl_1 >= -30000
 c76: - s_bl_2 + p_bl_2 >= -30000
 c77: - s_bl_3 + p_bl_3 >= -30000
 c78: - s_bl_4 + p_bl_4 >= -30000
 c79: - s_up_1 + p_up_1 >= -30000
 c80: - s_up_2 + p_up_2 >= -30000
 c81: - s_up_3 + p_up_3 >= -30000
 c82: - s_up_4 + p_up_4 >= -30000
 c83: - c_bl_1 + c_up_1 + d_bl_1 - d_up_1 + y_1  = -7000
 c84: - c_bl_2 + c_up_2 + d_bl_2 - d_up_2 + y_2  = -7000
 c85: - c_bl_3 + c_up_3 + d_bl_3 - d_up_3 + y_3  = -7000
 c86: - c_bl_4 + c_up_4 + d_bl_4 - d_up_4 + y_4  = -7000
Bounds
 0 <= u_bl_1 <= 1
 0 <= u_bl_2 <= 1
 0 <= u_bl_3 <= 1
 0 <= u_bl_4 <= 1
 0 <= u_up_1 <= 1
 0 <= u_up_2 <= 1
 0 <= u_up_3 <= 1
 0 <= u_up_4 <= 1
 0 <= v_bl_1 <= 1
 0 <= v_bl_2 <= 1
 0 <= v_bl_3 <= 1
 0 <= v_bl_4 <= 1
 0 <= v_up_1 <= 1
 0 <= v_up_2 <= 1
 0 <= v_up_3 <= 1
 0 <= v_up_4 <= 1
Binaries
 u_bl_1  u_bl_2  u_bl_3  u_bl_4  u_up_1  u_up_2  u_up_3  u_up_4  v_bl_1 
 v_bl_2  v_bl_3  v_bl_4  v_up_1  v_up_2  v_up_3  v_up_4 
End
