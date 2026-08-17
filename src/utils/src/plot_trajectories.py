"""
PLOT GROUND TRUTH, WHEEL ODOMETRY AND VISUAL ODOMETRY
"""


import pandas as pd
import os

import matplotlib.pyplot as plt

# Diretório atual
dir_path = os.path.dirname(os.path.abspath(__file__))
trajectory_dir = os.path.join(dir_path, "trajetorias")

dif = "s_SIFT_RATIO_CLAHE_03"

vo_files = f"./trajetorias/visual_odom_trajectory_{dif}.csv"
wheel_files = f"./trajetorias/odom_trajectory_{dif}.csv"
gt_file = f"./trajetorias/ground_truth_{dif}.csv"

plt.figure(figsize=(10, 8))

# Lê os arquivos
df_vo = pd.read_csv(vo_files)
df_wheel = pd.read_csv(wheel_files)
df_gt = pd.read_csv(gt_file)


scale = 1
# Plotar trajetórias VO
# For some reason, the 'x' axis is twisted in the visual odometry. If the plotted chart is wrong, try removing the negative sign in the following plot
plt.plot(-scale*df_vo['x'], scale*df_vo['y'], label=f'Visual Odometry')

# Plotar trajetórias Wheel
plt.plot(df_wheel['x'], df_wheel['y'], linestyle='--', label=f'Wheel Odometry')

# Plotar trajetórias e ground truth
offset_x = -df_gt['world_x_m'][0]
offset_y = -df_gt['world_y_m'][0]
plt.plot(df_gt['world_y_m'] + offset_y, df_gt['world_x_m'] + offset_x, color='black', label='Ground Truth')


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparação de Trajetórias VO vs Wheel')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'trajectories_comparison.png'))
plt.show()
