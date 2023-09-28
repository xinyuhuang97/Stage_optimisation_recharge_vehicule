\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: - 1000 s_bl_10 + 0.025 c_bl_1 + 0.025 c_bl_2 + 0.025 c_bl_3
       + 0.025 c_bl_4 + 0.025 c_bl_5 + 0.025 c_bl_6 + 0.025 c_bl_7
       + 0.025 c_bl_8 + 0.025 c_bl_9 + 0.025 c_bl_10 - 0.025 d_bl_1
       - 0.025 d_bl_2 - 0.025 d_bl_3 - 0.025 d_bl_4 - 0.025 d_bl_5
       - 0.025 d_bl_6 - 0.025 d_bl_7 - 0.025 d_bl_8 - 0.025 d_bl_9
       - 0.025 d_bl_10 + y_1 + y_2 + y_3 + y_4 + y_5 + y_6 + y_7 + y_8 + y_9
       + y_10
Subject To
 SOC_init_bl#0:                 s_bl_0  = 9936.59754056154
 SOC_max_bl1#1:                 s_bl_1 <= 30000
 SOC_max_bl2#2:                 s_bl_2 <= 30000
 SOC_max_bl3#3:                 s_bl_3 <= 30000
 SOC_max_bl4#4:                 s_bl_4 <= 30000
 SOC_max_bl5#5:                 s_bl_5 <= 30000
 SOC_max_bl6#6:                 s_bl_6 <= 30000
 SOC_max_bl7#7:                 s_bl_7 <= 30000
 SOC_max_bl8#8:                 s_bl_8 <= 30000
 SOC_max_bl9#9:                 s_bl_9 <= 30000
 SOC_max_bl10#10:               s_bl_10 <= 30000
 SOC_min_bl1#11:                s_bl_1 >= 0
 SOC_min_bl2#12:                s_bl_2 >= 0
 SOC_min_bl3#13:                s_bl_3 >= 0
 SOC_min_bl4#14:                s_bl_4 >= 0
 SOC_min_bl5#15:                s_bl_5 >= 0
 SOC_min_bl6#16:                s_bl_6 >= 0
 SOC_min_bl7#17:                s_bl_7 >= 0
 SOC_min_bl8#18:                s_bl_8 >= 0
 SOC_min_bl9#19:                s_bl_9 >= 0
 SOC_min_bl10#20:               s_bl_10 >= 0
 charge_or_discharge_bl1#21:    u_bl_1 + v_bl_1 <= 1
 charge_or_discharge_bl2#22:    u_bl_2 + v_bl_2 <= 1
 charge_or_discharge_bl3#23:    u_bl_3 + v_bl_3 <= 1
 charge_or_discharge_bl4#24:    u_bl_4 + v_bl_4 <= 1
 charge_or_discharge_bl5#25:    u_bl_5 + v_bl_5 <= 1
 charge_or_discharge_bl6#26:    u_bl_6 + v_bl_6 <= 1
 charge_or_discharge_bl7#27:    u_bl_7 + v_bl_7 <= 1
 charge_or_discharge_bl8#28:    u_bl_8 + v_bl_8 <= 1
 charge_or_discharge_bl9#29:    u_bl_9 + v_bl_9 <= 1
 charge_or_discharge_bl10#30:   u_bl_10 + v_bl_10 <= 1
 charge_or_discharge_up1#31:    u_up_1 + v_up_1 <= 1
 charge_or_discharge_up2#32:    u_up_2 + v_up_2 <= 1
 charge_or_discharge_up3#33:    u_up_3 + v_up_3 <= 1
 charge_or_discharge_up4#34:    u_up_4 + v_up_4 <= 1
 charge_or_discharge_up5#35:    u_up_5 + v_up_5 <= 1
 charge_or_discharge_up6#36:    u_up_6 + v_up_6 <= 1
 charge_or_discharge_up7#37:    u_up_7 + v_up_7 <= 1
 charge_or_discharge_up8#38:    u_up_8 + v_up_8 <= 1
 charge_or_discharge_up9#39:    u_up_9 + v_up_9 <= 1
 charge_or_discharge_up10#40:   u_up_10 + v_up_10 <= 1
 charge_upper_bound_bl1#41:     c_bl_1 - 15000 u_bl_1 <= 0
 charge_upper_bound_bl2#42:     c_bl_2 - 15000 u_bl_2 <= 0
 charge_upper_bound_bl3#43:     c_bl_3 - 15000 u_bl_3 <= 0
 charge_upper_bound_bl4#44:     c_bl_4 - 15000 u_bl_4 <= 0
 charge_upper_bound_bl5#45:     c_bl_5 - 15000 u_bl_5 <= 0
 charge_upper_bound_bl6#46:     c_bl_6 - 15000 u_bl_6 <= 0
 charge_upper_bound_bl7#47:     c_bl_7 - 15000 u_bl_7 <= 0
 charge_upper_bound_bl8#48:     c_bl_8 - 15000 u_bl_8 <= 0
 charge_upper_bound_bl9#49:     c_bl_9 - 15000 u_bl_9 <= 0
 charge_upper_bound_bl10#50:    c_bl_10 - 15000 u_bl_10 <= 0
 discharge_upper_bound_bl1#51:  d_bl_1 - 15000 v_bl_1 <= 0
 discharge_upper_bound_bl2#52:  d_bl_2 - 15000 v_bl_2 <= 0
 discharge_upper_bound_bl3#53:  d_bl_3 - 15000 v_bl_3 <= 0
 discharge_upper_bound_bl4#54:  d_bl_4 - 15000 v_bl_4 <= 0
 discharge_upper_bound_bl5#55:  d_bl_5 - 15000 v_bl_5 <= 0
 discharge_upper_bound_bl6#56:  d_bl_6 - 15000 v_bl_6 <= 0
 discharge_upper_bound_bl7#57:  d_bl_7 - 15000 v_bl_7 <= 0
 discharge_upper_bound_bl8#58:  d_bl_8 - 15000 v_bl_8 <= 0
 discharge_upper_bound_bl9#59:  d_bl_9 - 15000 v_bl_9 <= 0
 discharge_upper_bound_bl10#60: d_bl_10 - 15000 v_bl_10 <= 0
 charge_lower_bound_bl1#61:     - c_bl_1 <= 0
 charge_lower_bound_bl2#62:     - c_bl_2 <= 0
 charge_lower_bound_bl3#63:     - c_bl_3 <= 0
 charge_lower_bound_bl4#64:     - c_bl_4 <= 0
 charge_lower_bound_bl5#65:     - c_bl_5 <= 0
 charge_lower_bound_bl6#66:     - c_bl_6 <= 0
 charge_lower_bound_bl7#67:     - c_bl_7 <= 0
 charge_lower_bound_bl8#68:     - c_bl_8 <= 0
 charge_lower_bound_bl9#69:     - c_bl_9 <= 0
 charge_lower_bound_bl10#70:    - c_bl_10 <= 0
 discharge_lower_bound_bl1#71:  - d_bl_1 <= 0
 discharge_lower_bound_bl2#72:  - d_bl_2 <= 0
 discharge_lower_bound_bl3#73:  - d_bl_3 <= 0
 discharge_lower_bound_bl4#74:  - d_bl_4 <= 0
 discharge_lower_bound_bl5#75:  - d_bl_5 <= 0
 discharge_lower_bound_bl6#76:  - d_bl_6 <= 0
 discharge_lower_bound_bl7#77:  - d_bl_7 <= 0
 discharge_lower_bound_bl8#78:  - d_bl_8 <= 0
 discharge_lower_bound_bl9#79:  - d_bl_9 <= 0
 discharge_lower_bound_bl10#80: - d_bl_10 <= 0
 Production_balance_bl0#81:     - s_bl_0 + s_bl_1 - 0.25 c_bl_1 + 0.25 d_bl_1
                                 = 0
 Production_balance_bl1#82:     - s_bl_1 + s_bl_2 - 0.25 c_bl_2 + 0.25 d_bl_2
                                 = 0
 Production_balance_bl2#83:     - s_bl_2 + s_bl_3 - 0.25 c_bl_3 + 0.25 d_bl_3
                                 = 0
 Production_balance_bl3#84:     - s_bl_3 + s_bl_4 - 0.25 c_bl_4 + 0.25 d_bl_4
                                 = 0
 Production_balance_bl4#85:     - s_bl_4 + s_bl_5 - 0.25 c_bl_5 + 0.25 d_bl_5
                                 = 0
 Production_balance_bl5#86:     - s_bl_5 + s_bl_6 - 0.25 c_bl_6 + 0.25 d_bl_6
                                 = 0
 Production_balance_bl6#87:     - s_bl_6 + s_bl_7 - 0.25 c_bl_7 + 0.25 d_bl_7
                                 = 0
 Production_balance_bl7#88:     - s_bl_7 + s_bl_8 - 0.25 c_bl_8 + 0.25 d_bl_8
                                 = 0
 Production_balance_bl8#89:     - s_bl_8 + s_bl_9 - 0.25 c_bl_9 + 0.25 d_bl_9
                                 = 0
 Production_balance_bl9#90:     - s_bl_9 + s_bl_10 - 0.25 c_bl_10
                                + 0.25 d_bl_10  = 0
 c1#91:                         s_bl_1 >= 0
 c2#92:                         s_bl_2 >= 0
 c3#93:                         s_bl_3 >= 0
 c4#94:                         s_bl_4 >= 0
 c5#95:                         s_bl_5 >= 0
 c6#96:                         s_bl_6 >= 526.099156071528
 c7#97:                         s_bl_7 >= 4276.09915607153
 c8#98:                         s_bl_8 >= 8026.09915607153
 c9#99:                         s_bl_9 >= 11776.0991560715
 c10#100:                       s_bl_10 >= 15526.0991560715
 c11#101:                       15000 u_up_1 + z_1_1 >= 0
 c12#102:                       15000 u_up_2 + z_1_2 >= 0
 c13#103:                       15000 u_up_3 + z_1_3 >= 0
 c14#104:                       15000 u_up_4 + z_1_4 >= 0
 c15#105:                       15000 u_up_5 + z_1_5 >= 0
 c16#106:                       15000 u_up_6 + z_1_6 >= 0
 c17#107:                       15000 u_up_7 + z_1_7 >= 0
 c18#108:                       15000 u_up_8 + z_1_8 >= 0
 c19#109:                       15000 u_up_9 + z_1_9 >= 0
 c20#110:                       15000 u_up_10 + z_1_10 >= 0
 c112#111:                      15000 v_up_1 + z_2_1 >= 0
 c113#112:                      15000 v_up_2 + z_2_2 >= 0
 c114#113:                      15000 v_up_3 + z_2_3 >= 0
 c115#114:                      15000 v_up_4 + z_2_4 >= 0
 c116#115:                      15000 v_up_5 + z_2_5 >= 0
 c117#116:                      15000 v_up_6 + z_2_6 >= 0
 c118#117:                      15000 v_up_7 + z_2_7 >= 0
 c119#118:                      15000 v_up_8 + z_2_8 >= 0
 c120#119:                      15000 v_up_9 + z_2_9 >= 0
 c121#120:                      15000 v_up_10 + z_2_10 >= 0
 c31#121:                       - s_bl_0 + 0.25 z_1_1 >= -30000
 c32#122:                       - s_bl_1 + 0.25 z_1_2 >= -30000
 c33#123:                       - s_bl_2 + 0.25 z_1_3 >= -30000
 c34#124:                       - s_bl_3 + 0.25 z_1_4 >= -30000
 c35#125:                       - s_bl_4 + 0.25 z_1_5 >= -30000
 c36#126:                       - s_bl_5 + 0.25 z_1_6 >= -30000
 c37#127:                       - s_bl_6 + 0.25 z_1_7 >= -30000
 c38#128:                       - s_bl_7 + 0.25 z_1_8 >= -30000
 c39#129:                       - s_bl_8 + 0.25 z_1_9 >= -30000
 c40#130:                       - s_bl_9 + 0.25 z_1_10 >= -30000
 c41#131:                       s_bl_0 + 0.25 z_2_1 >= 0
 c42#132:                       s_bl_1 + 0.25 z_2_2 >= 0
 c43#133:                       s_bl_2 + 0.25 z_2_3 >= 0
 c44#134:                       s_bl_3 + 0.25 z_2_4 >= 0
 c45#135:                       s_bl_4 + 0.25 z_2_5 >= 0
 c46#136:                       s_bl_5 + 0.25 z_2_6 >= 526.099156071528
 c47#137:                       s_bl_6 + 0.25 z_2_7 >= 4276.09915607153
 c48#138:                       s_bl_7 + 0.25 z_2_8 >= 8026.09915607153
 c49#139:                       s_bl_8 + 0.25 z_2_9 >= 11776.0991560715
 c50#140:                       s_bl_9 + 0.25 z_2_10 >= 15526.0991560715
 c51#141:                       - y_1  = 0
 c52#142:                       - y_2  = 0
 c53#143:                       - y_3  = 0
 c54#144:                       - y_4  = 0
 c55#145:                       - y_5  = 0
 c56#146:                       - y_6  = 0
 c57#147:                       - y_7  = 0
 c58#148:                       - y_8  = 0
 c59#149:                       - y_9  = 0
 c60#150:                       - y_10  = 0
 c61#151:                       u_bl_1 >= 1
 c62#152:                       u_bl_2 >= 1
 c63#153:                       u_bl_3 >= 1
 c64#154:                       u_bl_4 >= 1
 c65#155:                       u_bl_5 >= 1
 c66#156:                       u_bl_6 >= 1
 c67#157:                       u_bl_7 >= 1
 c68#158:                       u_bl_8 >= 1
 c69#159:                       u_bl_9 >= 1
 c70#160:                       u_bl_10 >= 1
 c71#161:                       v_bl_1 >= 0
 c72#162:                       v_bl_2 >= 0
 c73#163:                       v_bl_3 >= 0
 c74#164:                       v_bl_4 >= 0
 c75#165:                       v_bl_5 >= 0
 c76#166:                       v_bl_6 >= 0
 c77#167:                       v_bl_7 >= 0
 c78#168:                       v_bl_8 >= 0
 c79#169:                       v_bl_9 >= 0
 c80#170:                       v_bl_10 >= 0
 c81#171:                       u_up_1 >= 0
 c82#172:                       u_up_2 >= 0
 c83#173:                       u_up_3 >= 0
 c84#174:                       u_up_4 >= 0
 c85#175:                       u_up_5 >= 0
 c86#176:                       u_up_6 >= 0
 c87#177:                       u_up_7 >= 0
 c88#178:                       u_up_8 >= 0
 c89#179:                       u_up_9 >= 0
 c90#180:                       u_up_10 >= 0
 c91#181:                       v_up_1 >= 1
 c92#182:                       v_up_2 >= 1
 c93#183:                       v_up_3 >= 1
 c94#184:                       v_up_4 >= 1
 c95#185:                       v_up_5 >= 1
 c96#186:                       v_up_6 >= 1
 c97#187:                       v_up_7 >= 1
 c98#188:                       v_up_8 >= 1
 c99#189:                       v_up_9 >= 1
 c100#190:                      v_up_10 >= 1
Bounds
      s_bl_0 Free
      s_bl_1 Free
      s_bl_2 Free
      s_bl_3 Free
      s_bl_4 Free
      s_bl_5 Free
      s_bl_6 Free
      s_bl_7 Free
      s_bl_8 Free
      s_bl_9 Free
      s_bl_10 Free
      c_bl_1 Free
      c_bl_2 Free
      c_bl_3 Free
      c_bl_4 Free
      c_bl_5 Free
      c_bl_6 Free
      c_bl_7 Free
      c_bl_8 Free
      c_bl_9 Free
      c_bl_10 Free
      d_bl_1 Free
      d_bl_2 Free
      d_bl_3 Free
      d_bl_4 Free
      d_bl_5 Free
      d_bl_6 Free
      d_bl_7 Free
      d_bl_8 Free
      d_bl_9 Free
      d_bl_10 Free
 0 <= u_bl_1 <= 1
 0 <= u_bl_2 <= 1
 0 <= u_bl_3 <= 1
 0 <= u_bl_4 <= 1
 0 <= u_bl_5 <= 1
 0 <= u_bl_6 <= 1
 0 <= u_bl_7 <= 1
 0 <= u_bl_8 <= 1
 0 <= u_bl_9 <= 1
 0 <= u_bl_10 <= 1
 0 <= u_up_1 <= 1
 0 <= u_up_2 <= 1
 0 <= u_up_3 <= 1
 0 <= u_up_4 <= 1
 0 <= u_up_5 <= 1
 0 <= u_up_6 <= 1
 0 <= u_up_7 <= 1
 0 <= u_up_8 <= 1
 0 <= u_up_9 <= 1
 0 <= u_up_10 <= 1
 0 <= v_bl_1 <= 1
 0 <= v_bl_2 <= 1
 0 <= v_bl_3 <= 1
 0 <= v_bl_4 <= 1
 0 <= v_bl_5 <= 1
 0 <= v_bl_6 <= 1
 0 <= v_bl_7 <= 1
 0 <= v_bl_8 <= 1
 0 <= v_bl_9 <= 1
 0 <= v_bl_10 <= 1
 0 <= v_up_1 <= 1
 0 <= v_up_2 <= 1
 0 <= v_up_3 <= 1
 0 <= v_up_4 <= 1
 0 <= v_up_5 <= 1
 0 <= v_up_6 <= 1
 0 <= v_up_7 <= 1
 0 <= v_up_8 <= 1
 0 <= v_up_9 <= 1
 0 <= v_up_10 <= 1
      y_1 Free
      y_2 Free
      y_3 Free
      y_4 Free
      y_5 Free
      y_6 Free
      y_7 Free
      y_8 Free
      y_9 Free
      y_10 Free
      z_1_1 Free
      z_1_2 Free
      z_1_3 Free
      z_1_4 Free
      z_1_5 Free
      z_1_6 Free
      z_1_7 Free
      z_1_8 Free
      z_1_9 Free
      z_1_10 Free
      z_2_1 Free
      z_2_2 Free
      z_2_3 Free
      z_2_4 Free
      z_2_5 Free
      z_2_6 Free
      z_2_7 Free
      z_2_8 Free
      z_2_9 Free
      z_2_10 Free
Binaries
 u_bl_1  u_bl_2  u_bl_3  u_bl_4  u_bl_5  u_bl_6  u_bl_7  u_bl_8  u_bl_9 
 u_bl_10  u_up_1  u_up_2  u_up_3  u_up_4  u_up_5  u_up_6  u_up_7  u_up_8 
 u_up_9  u_up_10  v_bl_1  v_bl_2  v_bl_3  v_bl_4  v_bl_5  v_bl_6  v_bl_7 
 v_bl_8  v_bl_9  v_bl_10  v_up_1  v_up_2  v_up_3  v_up_4  v_up_5  v_up_6 
 v_up_7  v_up_8  v_up_9  v_up_10 
End
