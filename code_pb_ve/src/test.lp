\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: - 10000 s_bl_3 + 0.025 c_bl_1 + 0.025 c_bl_2 + 0.025 c_bl_3
       - 0.025 d_bl_1 - 0.025 d_bl_2 - 0.025 d_bl_3 + [ 2 n_bl_1 ^2
       + 2 n_bl_2 ^2 + 2 n_bl_3 ^2 + 2 n_up_1 ^2 + 2 n_up_2 ^2 + 2 n_up_3 ^2
       + 2 p_bl_1 ^2 + 2 p_bl_2 ^2 + 2 p_bl_3 ^2 + 2 p_up_1 ^2 + 2 p_up_2 ^2
       + 2 p_up_3 ^2 ] / 2
Subject To
 c1:  s_bl_0  = 7994.65366241099
 c2:  s_up_0  = 7994.65366241099
 c3:  s_bl_1 <= 30000
 c4:  s_bl_2 <= 30000
 c5:  s_bl_3 <= 30000
 c6:  s_up_1 <= 30000
 c7:  s_up_2 <= 30000
 c8:  s_up_3 <= 30000
 c9:  s_bl_1 >= 0
 c10: s_bl_2 >= 0
 c11: s_bl_3 >= 0
 c12: s_up_1 >= 0
 c13: s_up_2 >= 0
 c14: s_up_3 >= 0
 c15: u_bl_1 + v_bl_1 <= 1
 c16: u_bl_2 + v_bl_2 <= 1
 c17: u_bl_3 + v_bl_3 <= 1
 c18: u_up_1 + v_up_1 <= 1
 c19: u_up_2 + v_up_2 <= 1
 c20: u_up_3 + v_up_3 <= 1
 c21: c_bl_1 - 15000 u_bl_1 <= 0
 c22: c_bl_2 - 15000 u_bl_2 <= 0
 c23: c_bl_3 - 15000 u_bl_3 <= 0
 c24: c_up_1 - 15000 u_up_1 <= 0
 c25: c_up_2 - 15000 u_up_2 <= 0
 c26: c_up_3 - 15000 u_up_3 <= 0
 c27: d_bl_1 - 15000 v_bl_1 <= 0
 c28: d_bl_2 - 15000 v_bl_2 <= 0
 c29: d_bl_3 - 15000 v_bl_3 <= 0
 c30: d_up_1 - 15000 v_up_1 <= 0
 c31: d_up_2 - 15000 v_up_2 <= 0
 c32: d_up_3 - 15000 v_up_3 <= 0
 c33: - c_bl_1 + 1000 u_bl_1 <= 0
 c34: - c_bl_2 + 1000 u_bl_2 <= 0
 c35: - c_bl_3 + 1000 u_bl_3 <= 0
 c36: - c_up_1 + 1000 u_up_1 <= 0
 c37: - c_up_2 + 1000 u_up_2 <= 0
 c38: - c_up_3 + 1000 u_up_3 <= 0
 c39: - d_bl_1 + 1000 v_bl_1 <= 0
 c40: - d_bl_2 + 1000 v_bl_2 <= 0
 c41: - d_bl_3 + 1000 v_bl_3 <= 0
 c42: - d_up_1 + 1000 v_up_1 <= 0
 c43: - d_up_2 + 1000 v_up_2 <= 0
 c44: - d_up_3 + 1000 v_up_3 <= 0
 c45: - s_bl_0 + s_bl_1 - 0.25 c_bl_1 + 0.25 d_bl_1  = 0
 c46: - s_bl_1 + s_bl_2 - 0.25 c_bl_2 + 0.25 d_bl_2  = 0
 c47: - s_bl_2 + s_bl_3 - 0.25 c_bl_3 + 0.25 d_bl_3  = 0
 c48: - s_bl_0 + s_up_1 - 0.25 c_up_1 + 0.25 d_up_1  = 0
 c49: - s_bl_1 + s_up_2 - 0.25 c_up_2 + 0.25 d_up_2  = 0
 c50: - s_bl_2 + s_up_3 - 0.25 c_up_3 + 0.25 d_up_3  = 0
 c51: s_bl_1 + n_bl_1 >= 0
 c52: s_bl_2 + n_bl_2 >= 0
 c53: s_bl_3 + n_bl_3 >= 14022.1766855634
 c54: s_up_1 + n_up_1 >= 0
 c55: s_up_2 + n_up_2 >= 0
 c56: s_up_3 + n_up_3 >= 14022.1766855634
 c57: - s_bl_1 + p_bl_1 >= -30000
 c58: - s_bl_2 + p_bl_2 >= -30000
 c59: - s_bl_3 + p_bl_3 >= -30000
 c60: - s_up_1 + p_up_1 >= -30000
 c61: - s_up_2 + p_up_2 >= -30000
 c62: - s_up_3 + p_up_3 >= -30000
 c63: - c_bl_1 + c_up_1 + d_bl_1 - d_up_1 + y_1  = -7000
 c64: - c_bl_2 + c_up_2 + d_bl_2 - d_up_2 + y_2  = -7000
 c65: - c_bl_3 + c_up_3 + d_bl_3 - d_up_3 + y_3  = -7000
Bounds
 0 <= u_bl_1 <= 1
 0 <= u_bl_2 <= 1
 0 <= u_bl_3 <= 1
 0 <= u_up_1 <= 1
 0 <= u_up_2 <= 1
 0 <= u_up_3 <= 1
 0 <= v_bl_1 <= 1
 0 <= v_bl_2 <= 1
 0 <= v_bl_3 <= 1
 0 <= v_up_1 <= 1
 0 <= v_up_2 <= 1
 0 <= v_up_3 <= 1
Binaries
 u_bl_1  u_bl_2  u_bl_3  u_up_1  u_up_2  u_up_3  v_bl_1  v_bl_2  v_bl_3 
 v_up_1  v_up_2  v_up_3 
End
