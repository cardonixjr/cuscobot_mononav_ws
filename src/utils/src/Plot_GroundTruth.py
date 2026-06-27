import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminho para o arquivo ground_truth.csv
#dir_path = os.path.dirname(os.path.abspath(__file__))
#gt_dir = os.path.join(dir_path, "trajetorias")
#gt_file = os.path.join(gt_dir, "ground_truth_0703_TESTE.csv")

gt_file = "src/utils/trajetorias/ground_truth_ORB_25_04-1.csv"
dir_path = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(gt_file):

    # Plotar trajetórias ground truth
    df_gt = pd.read_csv(gt_file)
    offset_x = df_gt['x_world'][0]
    offset_y = -df_gt['y_world'][0]
    plt.figure(figsize=(8, 6))
    plt.plot(-df_gt['x_world'] + offset_x, df_gt['y_world'] + offset_y, color='black', label='Ground Truth')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajetória Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'ground_truth_plot.png'))
    plt.show()
#else:
#    print(f"Arquivo não encontrado em {gt_dir}")