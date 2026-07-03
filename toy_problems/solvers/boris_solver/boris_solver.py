# import numpy for vector calculations and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import os

# define global variables for the particle's properties
Q = 1.0
M = 1.0

# define background fields
B_Z = 1.0
E_NON = 1.0

# analytical reference
def get_analytic_solution(t, v0): 
    # takes in a time array and the initial velocity magnitude
    # returns the exact positions using the cyclotron frequency

    omega_c = (Q * B_Z) / M
    r_c = v0 / omega_c

    x = r_c * (1.0 - np.cos(omega_c * t))
    y = r_c * np.sin(omega_c * t)
    z = np.zeros_like(t)

    return np.array([x, y, z])

def forward_euler_step(x, v, E, B, dt):
    # takes current state of the particles and fields
    # returns the position and velocity after one small timestep dt
    
    # F = q * (E + v x B)
    F_lorentz = Q * (E + np.cross(v, B))
    acceleration = F_lorentz / M

    x_next = x + v * dt
    v_next = v + acceleration * dt
    return x_next, v_next

def boris_step(x, v_minus_half, E, B, dt):
    # takes current state of the particles and fields
    # returns the position and velocity after one small timestep dt

    # Step 1: Half E field application
    v_minus = v_minus_half + (Q * E / M) * (dt / 2.0)

    # t vector = (q * B / m) * (dt / 2)
    t_vec = (Q * B / M) * (dt / 2.0)

    # s vector = 2 * t / (1 + |t|^2)
    t_mag_sq = np.dot(t_vec, t_vec)
    s_vec = (2.0 * t_vec) / (1 + t_mag_sq)

    
    # Step 2: Vector Rotation
    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Step 3: Half E field application
    v_next = v_plus + (Q * E / M) * (dt / 2.0)
    
    # Step 4: Update Position
    x_next = x + v_next * dt

    return x_next, v_next

def run_simulation(scheme_func, dt, t_max, x0, v0):
    num_steps = int(t_max / dt)
    time_array = np.linspace(0, t_max, num_steps + 1)

    # create arrays to hold the X, Y, Z history [N + 1, 3]
    x_hist = np.zeros((num_steps + 1, 3))
    v_hist = np.zeros((num_steps + 1, 3))

    # initialize positions
    x_hist[0] = x0

    # Define our static background fields
    E = np.array([E_NON, 0.0, 0.0])
    B = np.array([0.0, 0.0, B_Z])

    # rewind the velocity if using boris
    if scheme_func.__name__ == 'boris_step':
        F_init = Q * (E + np.cross(v0, B))
        v_current = v0 - (F_init / M) * (dt / 2.0)
    else:
        v_current = v0
    
    # store the true physical initial velocity at index 0
    v_hist[0] = v0

    for i in range(num_steps):
        # move one step
        x_next, v_next = scheme_func(x_hist[i], v_current, E, B, dt)

        # save the new position
        x_hist[i + 1] = x_next
        
        # update the velocity
        v_current = v_next

        # realign the velocity
        if scheme_func.__name__ == 'boris_step':
            v_hist[i + 1] = v_current + (Q * E / M) * (dt / 2.0)
        else:
            v_hist[i + 1] = v_current

    return time_array, x_hist, v_hist


# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

t_max = 20.0  
x0 = np.array([0.0, 0.0, 0.0])
v0 = np.array([0.0, 1.0, 0.0]) 
v0_mag = np.linalg.norm(v0)


# Step 1: Trajectory Verification
dt_plot = 0.1
t_arr, x_boris, _ = run_simulation(boris_step, dt_plot, t_max, x0, v0)
_, x_euler, _ = run_simulation(forward_euler_step, dt_plot, t_max, x0, v0)

# generate a high res exact analytical curve to overlay
t_fine = np.linspace(0, t_max, 1000)
x_analytic = get_analytic_solution(t_fine, v0_mag)

# constructing the plot
plt.figure(figsize=(8, 8))
plt.plot(x_analytic[0], x_analytic[1], 'k-', label='Analytic (Exact Theory)', linewidth=2.5)
plt.plot(x_boris[:, 0], x_boris[:, 1], 'b-o', label='Boris Method', markersize=4)
plt.plot(x_euler[:, 0], x_euler[:, 1], 'r:', label='Forward Euler')
plt.title(f'Particle Trajectory Comparison ($\Delta t$ = {dt_plot})')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.savefig('results/trajectory_verification.png')
plt.show()

# Step 2: Timestep Sensitivity Sweep
timesteps = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
boris_errors = []

for dt in timesteps:
    # run simulation for the specific dt
    t_arr, x_b, _ = run_simulation(boris_step, dt, t_max, x0, v0)
    
    # calculate exact positions matching those exact discrete timesteps
    x_true = get_analytic_solution(t_arr, v0_mag)
    x_true = x_true.T  # transpose shape to match x_b (Steps, 3)
    
    # L2 norm (euclidean distance error) at final timestamp t_max
    final_step_error = np.linalg.norm(x_b[-1] - x_true[-1])
    boris_errors.append(final_step_error)

# plotting the convergence curve on a log-log scale
plt.figure(figsize=(8, 6))
plt.loglog(timesteps, boris_errors, 'b-o', label='Boris Numerical Error', linewidth=2)

# generate a clean mathematical reference line representing true second-order slope: O(dt^2)
ref_slope = [1.5 * (dt**2) for dt in timesteps]
plt.loglog(timesteps, ref_slope, 'k:', label='Theoretical Second-Order $O(\Delta t^2)$')

plt.title('Timestep Sensitivity & Global Error Convergence')
plt.xlabel('Timestep Size ($\Delta t$) [Log Scale]')
plt.ylabel('Global Position Error at $t_{max}$ [Log Scale]')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('results/error_convergence_analysis.png')
plt.show()