#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <functional>
#include <sys/stat.h>

using namespace std;

// Vector3 structure for 3D vector operations
struct Vector3
{
    double x, y, z;

    Vector3() : x(0), y(0), z(0) {}
    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vector3 operator+(const Vector3 &v) const
    {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }

    Vector3 operator-(const Vector3 &v) const
    {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }

    Vector3 operator*(double scalar) const
    {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    Vector3 operator/(double scalar) const
    {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }

    double dot(const Vector3 &v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    Vector3 cross(const Vector3 &v) const
    {
        return Vector3(y * v.z - z * v.y,
                       z * v.x - x * v.z,
                       x * v.y - y * v.x);
    }

    double magnitude() const
    {
        return sqrt(x * x + y * y + z * z);
    }

    Vector3 normalized() const
    {
        double mag = magnitude();
        if (mag == 0)
            return Vector3(0, 0, 0);
        return *this / mag;
    }
};

// Global variables for particle properties
const double Q = 1.0; // Charge
const double M = 1.0; // Mass

// Background fields
const double B_Z = 1.0;   // Magnetic field in Z direction
const double E_NON = 0.0; // Non-physical electric field (normally 0)

// Analytical reference solution
vector<Vector3> get_analytic_solution(const vector<double> &time_array, double v0)
{
    vector<Vector3> result;

    double omega_c = (Q * B_Z) / M;
    double r_c = v0 / omega_c;

    for (double t : time_array)
    {
        double x = r_c * (1.0 - cos(omega_c * t));
        double y = r_c * sin(omega_c * t);
        double z = 0.0;
        result.push_back(Vector3(x, y, z));
    }

    return result;
}

// Forward Euler integration scheme
pair<Vector3, Vector3> forward_euler_step(const Vector3 &x, const Vector3 &v,
                                          const Vector3 &E, const Vector3 &B, double dt)
{
    // F = q * (E + v x B)
    Vector3 F_lorentz = (E + v.cross(B)) * Q;
    Vector3 acceleration = F_lorentz / M;

    Vector3 x_next = x + v * dt;
    Vector3 v_next = v + acceleration * dt;

    return {x_next, v_next};
}

// Boris integration scheme
pair<Vector3, Vector3> boris_step(const Vector3 &x, const Vector3 &v_minus_half,
                                  const Vector3 &E, const Vector3 &B, double dt)
{
    // Step 1: Half E field application
    Vector3 v_minus = v_minus_half + E * (Q / M * dt / 2.0);

    // t vector = (q * B / m) * (dt / 2)
    Vector3 t_vec = B * (Q / M * dt / 2.0);

    // s vector = 2 * t / (1 + |t|^2)
    double t_mag_sq = t_vec.dot(t_vec);
    Vector3 s_vec = t_vec * (2.0 / (1.0 + t_mag_sq));

    // Step 2: Vector Rotation
    Vector3 v_prime = v_minus + v_minus.cross(t_vec);
    Vector3 v_plus = v_minus + v_prime.cross(s_vec);

    // Step 3: Half E field application
    Vector3 v_next = v_plus + E * (Q / M * dt / 2.0);

    // Step 4: Update Position
    Vector3 x_next = x + v_next * dt;

    return {x_next, v_next};
}

// Run simulation with specified integration scheme
struct SimulationResult
{
    vector<double> time_array;
    vector<Vector3> x_hist;
    vector<Vector3> v_hist;
};

SimulationResult run_simulation(function<pair<Vector3, Vector3>(const Vector3 &, const Vector3 &, const Vector3 &, const Vector3 &, double)> scheme_func,
                                double dt, double t_max, const Vector3 &x0, const Vector3 &v0,
                                const std::string &scheme_name)
{
    int num_steps = static_cast<int>(t_max / dt);

    SimulationResult result;

    // Generate time array
    for (int i = 0; i <= num_steps; ++i)
    {
        result.time_array.push_back(i * dt);
    }

    // Initialize position and velocity arrays
    result.x_hist.resize(num_steps + 1);
    result.v_hist.resize(num_steps + 1);

    result.x_hist[0] = x0;

    // Define static background fields
    Vector3 E(E_NON, 0.0, 0.0);
    Vector3 B(0.0, 0.0, B_Z);

    // Rewind the velocity if using Boris scheme
    Vector3 v_current = v0;
    if (scheme_name == "boris_step")
    {
        Vector3 F_init = (E + v0.cross(B)) * Q;
        v_current = v0 - F_init * (dt / (2.0 * M));
    }

    // Store the true physical initial velocity at index 0
    result.v_hist[0] = v0;

    // Integration loop
    for (int i = 0; i < num_steps; ++i)
    {
        auto [x_next, v_next] = scheme_func(result.x_hist[i], v_current, E, B, dt);

        // Save new position
        result.x_hist[i + 1] = x_next;

        // Update velocity
        v_current = v_next;

        // Realign velocity if using Boris scheme
        if (scheme_name == "boris_step")
        {
            result.v_hist[i + 1] = v_current + E * (Q / M * dt / 2.0);
        }
        else
        {
            result.v_hist[i + 1] = v_current;
        }
    }

    return result;
}

// Save trajectory data to CSV file
void save_trajectory_to_csv(const string &filename, const SimulationResult &result)
{
    ofstream file("results/" + filename);
    file << fixed << setprecision(8);
    file << "time,x,y,z,vx,vy,vz\n";

    for (size_t i = 0; i < result.time_array.size(); ++i)
    {
        file << result.time_array[i] << ","
             << result.x_hist[i].x << ","
             << result.x_hist[i].y << ","
             << result.x_hist[i].z << ","
             << result.v_hist[i].x << ","
             << result.v_hist[i].y << ","
             << result.v_hist[i].z << "\n";
    }

    file.close();
    cout << "Saved trajectory to: " << filename << endl;
}

// Save error convergence data to CSV file
void save_convergence_to_csv(const string &filename,
                             const vector<double> &timesteps,
                             const vector<double> &errors)
{
    ofstream file("results/" + filename);
    file << fixed << setprecision(8);
    file << "timestep,error\n";

    for (size_t i = 0; i < timesteps.size(); ++i)
    {
        file << timesteps[i] << "," << errors[i] << "\n";
    }

    file.close();
    cout << "Saved convergence data to: " << filename << endl;
}

int main()
{
    // Create results directory if it doesn't exist
    mkdir("results", 0755);

    // Simulation parameters
    double t_max = 20.0;
    Vector3 x0(0.0, 0.0, 0.0);
    Vector3 v0(0.0, 1.0, 0.0);
    double v0_mag = v0.magnitude();

    // Step 1: Trajectory Verification
    cout << "\n=== Step 1: Trajectory Verification ===" << endl;

    double dt_plot = 0.1;
    auto boris_result = run_simulation(boris_step, dt_plot, t_max, x0, v0, "boris_step");
    auto euler_result = run_simulation(forward_euler_step, dt_plot, t_max, x0, v0, "forward_euler");

    // Generate high resolution analytical curve
    vector<double> t_fine;
    for (int i = 0; i <= 1000; ++i)
    {
        t_fine.push_back(i * t_max / 1000.0);
    }
    auto x_analytic = get_analytic_solution(t_fine, v0_mag);

    // Save trajectory comparisons
    save_trajectory_to_csv("boris_trajectory.csv", boris_result);
    save_trajectory_to_csv("euler_trajectory.csv", euler_result);

    // Save analytical solution
    ofstream analytic_file("results/analytic_trajectory.csv");
    analytic_file << fixed << setprecision(8);
    analytic_file << "time,x,y,z\n";
    for (size_t i = 0; i < t_fine.size(); ++i)
    {
        analytic_file << t_fine[i] << ","
                      << x_analytic[i].x << ","
                      << x_analytic[i].y << ","
                      << x_analytic[i].z << "\n";
    }
    analytic_file.close();
    cout << "Saved analytical trajectory to: analytic_trajectory.csv" << endl;

    // Step 2: Timestep Sensitivity Sweep
    cout << "\n=== Step 2: Timestep Sensitivity & Error Convergence ===" << endl;

    vector<double> timesteps = {0.4, 0.2, 0.1, 0.05, 0.025, 0.0125};
    vector<double> boris_errors;

    for (double dt : timesteps)
    {
        auto result = run_simulation(boris_step, dt, t_max, x0, v0, "boris_step");

        // Calculate exact positions at discrete timesteps
        auto x_true = get_analytic_solution(result.time_array, v0_mag);

        // L2 norm (Euclidean distance error) at final timestamp
        Vector3 error_vec = result.x_hist.back() - x_true.back();
        double final_step_error = error_vec.magnitude();
        boris_errors.push_back(final_step_error);

        cout << "dt = " << dt << ", error = " << final_step_error << endl;
    }

    // Save convergence analysis
    save_convergence_to_csv("error_convergence_analysis.csv", timesteps, boris_errors);

    // Print convergence analysis summary
    cout << "\n=== Error Convergence Analysis ===" << endl;
    cout << "Timestep Size | Global Position Error" << endl;
    cout << "------|------|" << endl;
    for (size_t i = 0; i < timesteps.size(); ++i)
    {
        cout << scientific << setprecision(4)
             << timesteps[i] << " | " << boris_errors[i] << endl;
    }

    cout << "\n=== Simulation Complete ===" << endl;
    cout << "Output files generated:" << endl;
    cout << "  - boris_trajectory.csv" << endl;
    cout << "  - euler_trajectory.csv" << endl;
    cout << "  - analytic_trajectory.csv" << endl;
    cout << "  - error_convergence_analysis.csv" << endl;

    return 0;
}
