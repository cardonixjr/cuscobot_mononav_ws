import pandas as pd
import os

import matplotlib.pyplot as plt

# Diretório atual
dir_path = os.path.dirname(os.path.abspath(__file__))
trajectory_dir = os.path.join(dir_path, "trajetorias")

dif = "0407_ORB_04"

vo_files = f"/home/luciano/cuscobot_mononav_ws/trajetorias/visual_odom_trajectory_{dif}.csv"
wheel_files = f"/home/luciano/cuscobot_mononav_ws/trajetorias/odom_trajectory_{dif}.csv"
gt_file = f"/home/luciano/cuscobot_mononav_ws/src/utils/src/intelbras_rael/trajetorias/ground_truth_{dif}.csv"

plt.figure(figsize=(10, 8))

# Plotar trajetórias VO
df_vo = pd.read_csv(vo_files)
plt.plot(df_vo['y'], -df_vo['z'], label=f'VO: {os.path.basename(vo_files)}')

# Plotar trajetórias Wheel
df_wheel = pd.read_csv(wheel_files)
plt.plot(df_wheel['x'], df_wheel['y'], linestyle='--', label=f'Wheel: {os.path.basename(wheel_files)}')

# Plotar trajetórias e ground truth
df_gt = pd.read_csv(gt_file)
offset_x = df_gt['world_x_m'][0]
offset_y = -df_gt['world_y_m'][0]

plt.plot(df_gt['world_y_m'] + offset_y, -df_gt['world_x_m'] + offset_x, color='black', label='Ground Truth')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparação de Trajetórias VO vs Wheel')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'trajectories_comparison.png'))
plt.show()