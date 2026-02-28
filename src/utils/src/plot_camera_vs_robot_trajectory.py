import pandas as pd
import os

import matplotlib.pyplot as plt

name_mod = "TESTEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"

# Define paths
base_path = "src/visual_core/results"
vo_file = os.path.join(base_path, f"vo_scaled_trajectory-{name_mod}.csv")
wheel_file = os.path.join(base_path, f"wheel_trajectory-{name_mod}.csv")

# Read CSV files
vo_data = pd.read_csv(vo_file)
wheel_data = pd.read_csv(wheel_file)

# Create plot
plt.figure(figsize=(12, 8))

# Plot trajectories
plt.plot(vo_data.iloc[:, 1], -vo_data.iloc[:, 2], label='VO Scaled Trajectory', marker='o', markersize=3)
plt.plot(wheel_data.iloc[:, 0], wheel_data.iloc[:, 1], label='Wheel Trajectory', marker='s', markersize=3)

# Labels and legend
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajectory Comparison: Visual Odometry vs Wheel Odometry')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Show plot
plt.tight_layout()
plt.show()