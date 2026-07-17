from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def results(filename):
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Run the C++ program first to generate it.")

    df = pd.read_csv(csv_path)
    x = df['x'].to_numpy()
    phi_num = df['phi'].to_numpy()
    ni = 33
    dx = (x[ni - 1] - x[0]) / (ni - 1)

    # Replace this with your analytic formula
    phi_analytic = -np.sin(x) -np.cos(x) + (np.sin(1) + np.cos(1)) * x + 1
    
    plt.figure(figsize=(7, 5))
    plt.plot(x, phi_num, label='Numeric Solution', marker='o')
    plt.plot(x, phi_analytic, label='Analytic Solution', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('phi')
    plt.title('Numerical vs Analytic Solution of Poisson\'s Equation')
    plt.legend(loc = 0)
    plt.grid(True, alpha=0.3)
    plt.show()

def error(filename):
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Run the C++ program first to generate it.")
    
    df = pd.read_csv(csv_path)
    dx = df['dx'].to_numpy()
    avg_abs_error = df['avg_abs_error'].to_numpy()
    
    analytic = avg_abs_error[-1] * (dx / dx[-1]) ** 2

    plt.figure(figsize = (7, 6))
    plt.plot(dx, avg_abs_error, label = "Average error (|e|)", marker = 'o', linestyle = '-')
    plt.plot(dx, analytic, label = "Slope = 2", linestyle = '--')
    plt.xlabel("dx")
    plt.ylabel("Average Error (|e|)")
    plt.title("Error vs dx")
    plt.xlim(1e-3, 1e-1)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 2)
    plt.grid(True, which='both', alpha = 0.9, linestyle = "--")
    plt.show()

def main():
    results('phi_solution.csv')
    error('error_vs_dx.csv')

if __name__ == "__main__":
    main()