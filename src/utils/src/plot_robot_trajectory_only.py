#!/usr/bin/env python3
"""
Plot only the robot odometry trajectory.
Applies the same 90° CCW rotation as plot_robot_person_together.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from pathlib import Path


def plot_robot_trajectory(robot_csv):
    """Plot robot odometry trajectory only."""
    try:
        robot_df = pd.read_csv(robot_csv)
        print(f"✅ Robot trajectory loaded: {len(robot_df)} points")
        print(f"   Columns: {list(robot_df.columns)}")
    except Exception as e:
        print(f"❌ Error loading robot trajectory: {e}")
        return

    robot_x_raw = robot_df['x'].values
    robot_y_raw = robot_df['y'].values

    # Apply 90° counter-clockwise rotation (same as plot_robot_person_together.py)
    # Transformation: x_new = -y_old, y_new = x_old
    robot_x = -robot_y_raw
    robot_y = robot_x_raw

    print(f"\n📐 Applied 90° counter-clockwise rotation to robot coordinates")
    print(f"   Original X range: [{robot_x_raw.min():.2f}, {robot_x_raw.max():.2f}] m")
    print(f"   Original Y range: [{robot_y_raw.min():.2f}, {robot_y_raw.max():.2f}] m")
    print(f"   Rotated X range: [{robot_x.min():.2f}, {robot_x.max():.2f}] m")
    print(f"   Rotated Y range: [{robot_y.min():.2f}, {robot_y.max():.2f}] m")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(robot_x, robot_y, 'r-o', label='Robot (Odometry)', linewidth=2, markersize=6, alpha=0.7)
    ax.scatter(robot_x[0], robot_y[0], c='green', s=200, marker='o', label='Start', zorder=5,
               edgecolor='black', linewidth=2)
    ax.scatter(robot_x[-1], robot_y[-1], c='red', s=200, marker='s', label='End', zorder=5,
               edgecolor='black', linewidth=2)

    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_title('Robot Odometry Trajectory (Top View)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    ax.axis('equal')

    timestamp = int(time.time())
    output_file = f"/home/mechro/cuscobot_ws/robot_trajectory_only_{timestamp}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_file}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        robot_csv = sys.argv[1]
    else:
        search_dirs = [
            Path("/home/mechro/cuscobot_ws/trajetorias"),
            Path("/home/mechro/cuscobot_ws/src/utils_package/trajetorias"),
            Path("/home/mechro/trajetorias"),
        ]

        robot_csv = None
        for search_dir in search_dirs:
            if search_dir.exists():
                robot_files = list(search_dir.glob("trajetoria_robo_*.csv"))
                if robot_files:
                    robot_csv = str(sorted(robot_files)[-1])
                    print(f"✅ Found robot trajectory: {robot_csv}")
                    break

        if not robot_csv:
            print("❌ No robot trajectory files found in any search directory")
            print(f"   Searched: {[str(d) for d in search_dirs]}")
            sys.exit(1)

        print("Using most recent file...\n")

    print(f"Plotting robot trajectory:")
    print(f"  Robot:  {robot_csv}")
    print()

    plot_robot_trajectory(robot_csv)
