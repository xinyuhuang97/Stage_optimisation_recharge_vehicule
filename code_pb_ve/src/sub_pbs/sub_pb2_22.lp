\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: - 1000 s_bl_3 + 0.025 c_bl_1 + 0.025 c_bl_2 + 0.025 c_bl_3
       - 0.025 d_bl_1 - 0.025 d_bl_2 - 0.025 d_bl_3 - 138.50758713685 y_3
       + [ 20000 n_bl_1 ^2 + 20000 n_bl_2 ^2 + 20000 n_bl_3 ^2
       + 20000 n_up_1 ^2 + 20000 n_up_2 ^2 + 20000 n_up_3 ^2 + 20000 p_bl_1 ^2
       + 20000 p_bl_2 ^2 + 20000 p_bl_3 ^2 + 20000 p_up_1 ^2 + 20000 p_up_2 ^2
       + 20000 p_up_3 ^2 ] / 2
Subject To
 SOC_init_bl#0:                  s_bl_0  = 8653.13450584807
 SOC_init_up#1:                  s_up_0  = 8653.13450584807
 SOC_max_bl1#2:                  s_bl_1 <= 30000
 SOC_max_bl2#3:                  s_bl_2 <= 30000
 SOC_max_bl3#4:                  s_bl_3 <= 30000
 SOC_max_up1#5:                  s_up_1 <= 30000
 SOC_max_up2#6:                  s_up_2 <= 30000
 SOC_max_up3#7:                  s_up_3 <= 30000
 SOC_min_bl1#8:                  s_bl_1 >= 0
 SOC_min_bl2#9:                  s_bl_2 >= 0
 SOC_min_bl3#10:                 s_bl_3 >= 0
 SOC_min_up1#11:                 s_up_1 >= 0
 SOC_min_up2#12:                 s_up_2 >= 0
 SOC_min_up3#13:                 s_up_3 >= 0
 charge_or_discharge_bl1#14:     u_bl_1 + v_bl_1 <= 1
 charge_or_discharge_bl2#15:     u_bl_2 + v_bl_2 <= 1
 charge_or_discharge_bl3#16:     u_bl_3 + v_bl_3 <= 1
 charge_or_discharge_up1#17:     u_up_1 + v_up_1 <= 1
 charge_or_discharge_up2#18:     u_up_2 + v_up_2 <= 1
 charge_or_discharge_up3#19:     u_up_3 + v_up_3 <= 1
 charge_upper_bound_bl1#20:      c_bl_1 - 15000 u_bl_1 <= 0
 charge_upper_bound_bl2#21:      c_bl_2 - 15000 u_bl_2 <= 0
 charge_upper_bound_bl3#22:      c_bl_3 - 15000 u_bl_3 <= 0
 charge_upper_bound_up1#23:      c_up_1 - 15000 u_up_1 <= 0
 charge_upper_bound_up2#24:      c_up_2 - 15000 u_up_2 <= 0
 charge_upper_bound_up3#25:      c_up_3 - 15000 u_up_3 <= 0
 discharge_upper_bound_bl1#26:   d_bl_1 - 15000 v_bl_1 <= 0
 discharge_upper_bound_bl2#27:   d_bl_2 - 15000 v_bl_2 <= 0
 discharge_upper_bound_bl3#28:   d_bl_3 - 15000 v_bl_3 <= 0
 discharge_upper_bound_up1#29:   d_up_1 - 15000 v_up_1 <= 0
 discharge_upper_bound_up2#30:   d_up_2 - 15000 v_up_2 <= 0
 discharge_upper_bound_up3#31:   d_up_3 - 15000 v_up_3 <= 0
 charge_lower_bound_bl1#32:      - c_bl_1 + 1000 u_bl_1 <= 0
 charge_lower_bound_bl2#33:      - c_bl_2 + 1000 u_bl_2 <= 0
 charge_lower_bound_bl3#34:      - c_bl_3 + 1000 u_bl_3 <= 0
 charge_lower_bound_up1#35:      - c_up_1 + 1000 u_up_1 <= 0
 charge_lower_bound_up2#36:      - c_up_2 + 1000 u_up_2 <= 0
 charge_lower_bound_up3#37:      - c_up_3 + 1000 u_up_3 <= 0
 discharge_lower_bound_bl1#38:   - d_bl_1 + 1000 v_bl_1 <= 0
 discharge_lower_bound_bl2#39:   - d_bl_2 + 1000 v_bl_2 <= 0
 discharge_lower_bound_bl3#40:   - d_bl_3 + 1000 v_bl_3 <= 0
 discharge_lower_bound_up1#41:   - d_up_1 + 1000 v_up_1 <= 0
 discharge_lower_bound_up2#42:   - d_up_2 + 1000 v_up_2 <= 0
 discharge_lower_bound_up3#43:   - d_up_3 + 1000 v_up_3 <= 0
 Production_balance_bl0#44:      - s_bl_0 + s_bl_1 - 0.25 c_bl_1 + 0.25 d_bl_1
                                  = 0
 Production_balance_bl1#45:      - s_bl_1 + s_bl_2 - 0.25 c_bl_2 + 0.25 d_bl_2
                                  = 0
 Production_balance_bl2#46:      - s_bl_2 + s_bl_3 - 0.25 c_bl_3 + 0.25 d_bl_3
                                  = 0
 Production_balance_up0#47:      - s_bl_0 + s_up_1 - 0.25 c_up_1 + 0.25 d_up_1
                                  = 0
 Production_balance_up1#48:      - s_bl_1 + s_up_2 - 0.25 c_up_2 + 0.25 d_up_2
                                  = 0
 Production_balance_up2#49:      - s_bl_2 + s_up_3 - 0.25 c_up_3 + 0.25 d_up_3
                                  = 0
 Negative_part_bl1#50:           s_bl_1 + n_bl_1 >= 0
 Negative_part_bl2#51:           s_bl_2 + n_bl_2 >= 450.944643032251
 Negative_part_bl3#52:           s_bl_3 + n_bl_3 >= 15450.9446430323
 Negative_part_up1#53:           s_up_1 + n_up_1 >= 0
 Negative_part_up2#54:           s_up_2 + n_up_2 >= 450.944643032251
 Negative_part_up3#55:           s_up_3 + n_up_3 >= 15450.9446430323
 Positive_part_bl1#56:           - s_bl_1 + p_bl_1 >= -30000
 Positive_part_bl2#57:           - s_bl_2 + p_bl_2 >= -30000
 Positive_part_bl3#58:           - s_bl_3 + p_bl_3 >= -30000
 Positive_part_up1#59:           - s_up_1 + p_up_1 >= -30000
 Positive_part_up2#60:           - s_up_2 + p_up_2 >= -30000
 Positive_part_up3#61:           - s_up_3 + p_up_3 >= -30000
 y_t_facilitate_calculation1#62: - c_bl_1 + c_up_1 + d_bl_1 - d_up_1 + y_1  = 0
 y_t_facilitate_calculation2#63: - c_bl_2 + c_up_2 + d_bl_2 - d_up_2 + y_2  = 0
 y_t_facilitate_calculation3#64: - c_bl_3 + c_up_3 + d_bl_3 - d_up_3 + y_3  = 0
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
      y_1 Free
      y_2 Free
      y_3 Free
Binaries
 u_bl_1  u_bl_2  u_bl_3  u_up_1  u_up_2  u_up_3  v_bl_1  v_bl_2  v_bl_3 
 v_up_1  v_up_2  v_up_3 
End
