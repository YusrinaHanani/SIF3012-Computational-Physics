# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Constants and Parameters
hbar = 1.0  # Planck's constant
m = 1.0     # Particle mass
alpha = 1.0 # alpha parameter
lmbda = 4.0 # lambda parameter

# x-value range parameters
x_min, x_max = -10, 10  # Original x range
h = 0.05                 # Step size
x = np.arange(x_min, x_max + h, h)  # x values

# Defining the Potential Function V(x)
def calculate_potential(x):
    """Calculate the potential V(x)"""
    coeff = (hbar**2 / (2 * m)) * alpha**2 * lmbda * (lmbda - 1)
    return coeff * (0.5 - 1 / (np.cosh(alpha * x)**2))

# Plotting the potential function V(x)
def plot_potential_no_lines():
    """Plot potential function as a smooth curve without reference lines"""
    potential = calculate_potential(x)
    plt.figure(figsize=(16, 10))
    plt.plot(x, potential, color="red", linewidth=2, label='Potential V(x)')
    plt.xlabel("x")
    plt.ylabel("Potential Energy V(x)")
    plt.title("Quantum Potential Function")
    plt.grid()
    plt.legend()
    plt.show()

def plot_potential_with_lines(eigenvalues):
    """Plot the potential function V(x) with reference lines for eigenvalues"""
    potential = calculate_potential(x)
    plt.figure(figsize=(16, 10))
    plt.plot(x, potential, color="blue", linewidth=2, label='Potential V(x)')
    
    # Define colors for eigenvalue lines
    line_colors = ['red', 'green', 'purple']
    
    for i, E in enumerate(eigenvalues):
        color = line_colors[i % len(line_colors)]  # Cycle through colors if more than 3 eigenvalues
        plt.axhline(E, color=color, linestyle='--', label=f'Eigenvalue E_{i} = {E:.3f}')
    plt.xlim(x_min, x_max)
    plt.xlabel("x")
    plt.ylabel("Potential Energy V(x)")
    plt.title("Quantum Potential Function with Energy Levels")
    plt.grid()
    plt.legend()
    plt.show()

# Numerov Method Implementation
def numerov_method(psi0, psi1, energy, x_values, step_size):
    """Integrate wavefunction using the Numerov method"""
    num_points = len(x_values)
    psi = np.zeros(num_points)
    psi[0], psi[1] = psi0, psi1

    potential_function = lambda xi: (2 * m / hbar**2) * (energy - calculate_potential(xi))

    for i in range(1, num_points - 1):
        k0 = potential_function(x_values[i - 1])
        k1 = potential_function(x_values[i])
        k2 = potential_function(x_values[i + 1])
        psi[i + 1] = (2 * (1 - (5 * step_size**2 * k1) / 12) * psi[i] -
                       (1 + (step_size**2 * k0) / 12) * psi[i - 1]) / (1 + (step_size**2 * k2) / 12)

    return psi

# Matching Condition for finding eigenvalues
def matching_condition(energy, x_values, step_size):
    """Calculate the matching condition for eigenvalue finding"""
    mid_index = len(x_values) // 2
    psi_left = numerov_method(0.0, 1e-5, energy, x_values, step_size)
    psi_right = numerov_method(0.0, 1e-5, energy, x_values[::-1], step_size)[::-1]

    ratio_left = (psi_left[mid_index + 1] - psi_left[mid_index - 1]) / (2 * step_size * psi_left[mid_index])
    ratio_right = (psi_right[mid_index + 1] - psi_right[mid_index - 1]) / (2 * step_size * psi_right[mid_index])

    return ratio_left - ratio_right

# Root-Finding for Eigenvalues
def find_eigenvalues(x_values, step_size, num_levels=3):
    """Find the first num_levels eigenvalues using root-finding"""
    eigenvalues = []
    E_start, E_end, dE = -2.0, 0.0, 2.0  # Initial energy range

    for _ in range(num_levels):
        energies = np.linspace(E_start, E_end, 100)

        for i in range(len(energies) - 1):
            E1, E2 = energies[i], energies[i + 1]

            if matching_condition(E1, x_values, step_size) * matching_condition(E2, x_values, step_size) < 0:
                result = root_scalar(matching_condition, args=(x_values, step_size), bracket=[E1, E2], method='brentq')
                eigenvalues.append(result.root)
                break

        E_start += dE  # Update energy range for the next level
        E_end += dE

    return eigenvalues

# Plotting the Potential and Wavefunctions
def plot_wavefunctions(eigenvalues):
    """Plot the wavefunctions corresponding to the eigenvalues"""
    plt.figure(figsize=(16, 10))
    plt.plot(x, calculate_potential(x), color='orange', label='Potential V(x)')

    wave_colors = ['purple', 'green', 'magenta']  # Different colors for wavefunctions

    for i, E in enumerate(eigenvalues):
        psi = numerov_method(0.0, 1e-5, E, x, h)
        normalized_psi = psi / np.max(np.abs(psi)) + E  # Normalize wavefunction
        plt.plot(x, normalized_psi, label=f'Eigenvalue E_{i} = {E:.3f}', color=wave_colors[i], linewidth=1.5)

    plt.xlabel("x")
    plt.ylabel("Wavefunction / Energy")
    plt.title("Wavefunctions for Quantum Potential")
    plt.xlim(x_min, x_max)
    plt.grid()
    plt.legend()
    plt.show()

# Exact Eigenvalues for Comparison
def exact_eigenvalues(num_levels):
    """Calculate exact eigenvalues for comparison"""
    return [(hbar**2 / (2 * m)) * alpha**2 * (lmbda * (lmbda - 1) / 2 - (lmbda - 1 - n)**2) for n in range(num_levels)]

# Main execution
# Plot 1: Potential without reference lines (smooth curve)
plot_potential_no_lines()

# Find eigenvalues
num_levels = 3
eigenvalues = find_eigenvalues(x, h, num_levels=num_levels)

# Print eigenvalues numerically
print("Numerically solved Eigenvalues:")
for i, E in enumerate(eigenvalues):
    print(f"E_{i} = {E:.6f}")

# Plot 2: Potential with reference lines for eigenvalues
plot_potential_with_lines(eigenvalues)

# Plot 3: Wavefunctions with eigenvalues
plot_wavefunctions(eigenvalues)