#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:50:12 2026

@author: ishaan
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt


x = np.linspace(0,1, 500)

n_particles = 500000
x_vals = []


for i in range(n_particles):
    tval = True
    while(tval): 
        R1 = rd.random()
        R2 = rd.random()
        if R2 * 1.3 < 1 + 0.3*(np.sin(2 * np.pi * R1)): 
            tval = False
            x_vals.append(R1)


    
    
# --- Plotting Configuration ---
plt.figure(figsize=(10, 6), dpi=100)

# 1. Plot the histogram of the particles (normalized)
plt.hist(x_vals, bins=100, density=True, alpha=0.6, color='skyblue', 
         edgecolor='steelblue', label='Particle Distribution (100 bins)')

plt.plot(x, 1 + 0.3 * np.sin(2 * np.pi * x), color='crimson', linewidth=2.5, 
         label=r'$1 + 0.3\sin(2\pi x)$ (Analytical)')
# Labels and styling
plt.title('Rejection Sampling: Particle Distribution vs. Analytical Curve', fontsize=14, pad=15)
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.xlim(0, 1)
#plt.ylim(0, 2.2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')

# Display the plot
plt.tight_layout()
plt.show()