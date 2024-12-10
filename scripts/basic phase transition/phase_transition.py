"""
Simulating the phase transition in the Vicsek model by analyzing 
the order parameter (φ) and susceptibility (χ) for varying levels of noise (η). 
The results demonstrate how the system transitions from ordered to disordered 
states as noise increases.

Functions:
----------

initialize_particles(n_particles, box_length, speed):
    Initializes a list of particles with random positions and velocities.

    Parameters:
    - n_particles (int): Number of particles to initialize.
    - box_length (float): Length of the simulation box (assumed to be square).
    - speed (float): Constant speed of particles.

    Returns:
    - list of Particle: A list of initialized Particle objects.

simulate_phase_transition(noise_values, n_particles, box_length, speed, 
                          interaction_range, equilibration_steps, measurement_steps):
    Runs Vicsek model simulations for different noise levels to compute 
    the order parameter and susceptibility.

    Parameters:
    - noise_values (list or array of float): Noise levels (η) to simulate.
    - n_particles (int): Number of particles in the simulation.
    - box_length (float): Length of the simulation box.
    - speed (float): Constant speed of particles.
    - interaction_range (float): Interaction radius for the particles.
    - equilibration_steps (int): Number of steps for system equilibration.
    - measurement_steps (int): Number of steps for measuring order parameters.

    Returns:
    - np.array: Array of mean order parameters for each noise value.
    - np.array: Array of susceptibilities for each noise value.

plot_phase_transition(noise_values, order_params, susceptibility):
    Creates a plot of the phase transition, showing the order parameter 
    and susceptibility as functions of noise.

    Parameters:
    - noise_values (list or array of float): Noise levels (η) simulated.
    - order_params (array of float): Mean order parameter for each noise value.
    - susceptibility (array of float): Susceptibility for each noise value.

    Returns:
    - matplotlib.figure

"""


import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from vicsek.models.particle import Particle
from vicsek.models.vicsek import Vicsek


def initialize_particles(n_particles, box_length, speed):
    """Initialize particles with random positions and velocities."""
    particles = []
    for i in range(n_particles):
        # Random position within box
        position = np.random.uniform(0, box_length, size=2)
        # Random direction for velocity
        angle = np.random.uniform(0, 2 * np.pi)
        velocity = speed * np.array([np.cos(angle), np.sin(angle)])

        particles.append(Particle(
            position=position,
            velocity=velocity,
            name=f"p{i}",
            type="standard"
        ))
    return particles


def simulate_phase_transition(noise_values, n_particles=1000, box_length=32,
                              speed=0.03, interaction_range=1.0,
                              equilibration_steps=1000, measurement_steps=200):
    """Run simulations for different noise values and measure order parameter."""
    density = n_particles / (box_length ** 2)
    order_parameters = []
    order_fluctuations = []  # For susceptibility

    print(f"Running simulations with density = {density:.2f}")

    # Main progress bar for noise values
    for noise in tqdm(noise_values, desc="Noise values", position=0):
        # Initialize system
        particles = initialize_particles(n_particles, box_length, speed)
        model = Vicsek(
            length=box_length,
            particles=particles,
            interaction_range=interaction_range,
            speed=speed,
            noise_factor=noise
        )

        # Equilibration
        model.run(equilibration_steps)

        # Measurement with progress bar
        measurements = []
        for _ in range(measurement_steps):
            model.step()
            measurements.append(model.order_parameter())

        order_parameters.append(np.mean(measurements))
        order_fluctuations.append(n_particles * np.var(measurements))  # Susceptibility

    return np.array(order_parameters), np.array(order_fluctuations)


def plot_phase_transition(noise_values, order_params, susceptibility):
    """Create a plot showing order parameter and susceptibility."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Order parameter plot
    ax1.plot(noise_values, order_params, 'o-', color='blue', markersize=8)
    ax1.set_ylabel('Order Parameter φ', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Vicsek Model Phase Transition', fontsize=14)

    # Susceptibility plot
    ax2.plot(noise_values, susceptibility, 'o-', color='red', markersize=8)
    ax2.set_xlabel('Noise η', fontsize=12)
    ax2.set_ylabel('Susceptibility χ', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Simulation parameters
    N = 1024  # Number of particles
    L = 16  # Box length
    v0 = 0.03  # Particle speed

    # Create noise values with more points around expected transition
    noise_values = np.concatenate([
        np.linspace(0, 1.5, 10),
        np.linspace(1.5, 3.0, 20),
        np.linspace(3.0, 5.0, 10)
    ])

    # Run simulation
    order_params, susceptibility = simulate_phase_transition(
        noise_values=noise_values,
        n_particles=N,
        box_length=L,
        speed=v0,
        equilibration_steps=400,
        measurement_steps=300
    )

    # Create and save plot
    fig = plot_phase_transition(noise_values, order_params, susceptibility)
    plt.savefig('vicsek_phase_transition.png', dpi=300, bbox_inches='tight')
    plt.close()
