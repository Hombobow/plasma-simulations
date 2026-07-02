import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the CSV file
data = pd.read_csv('trace.csv')

# Extract columns
t = data['t'].values
x = data['x'].values
v = data['v'].values

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Position vs Time
axes[0, 0].plot(t, x, 'b-o', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Time (s)', fontsize=12)
axes[0, 0].set_ylabel('Position (m)', fontsize=12)
axes[0, 0].set_title('Particle Position vs Time', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Velocity vs Time
axes[0, 1].plot(t, v, 'r-o', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Time (s)', fontsize=12)
axes[0, 1].set_ylabel('Velocity (m/s)', fontsize=12)
axes[0, 1].set_title('Particle Velocity vs Time', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Phase space (Position vs Velocity)
axes[1, 0].plot(x, v, 'g-o', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Position (m)', fontsize=12)
axes[1, 0].set_ylabel('Velocity (m/s)', fontsize=12)
axes[1, 0].set_title('Phase Space Trajectory', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Acceleration (numerical derivative of velocity)
acceleration = np.gradient(v, t)
axes[1, 1].plot(t, acceleration, 'purple', marker='o', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Time (s)', fontsize=12)
axes[1, 1].set_ylabel('Acceleration (m/s²)', fontsize=12)
axes[1, 1].set_title('Particle Acceleration vs Time', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=acceleration[0], color='k', linestyle='--', alpha=0.3, label='Constant acceleration')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('trace_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'trace_plot.png'")
plt.show()
