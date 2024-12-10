"""
Simulating the Vicsek model for particles and analyzing the equilibration behavior of the order parameter (φ) for varying levels of noise. 
The script generates a plot of how the order parameter evolves over time for different noise values.

Functions:
analyze_equilibration(noise_values, n_steps, n_particles, box_length, speed):
    Simulates the Vicsek model for given parameters and visualizes the order parameter's 
    evolution over time for different noise levels. Returns the figure object of the generated plot.

    Parameters:
    - noise_values (list of floats): List of noise values (η) to analyze.
    - n_steps (int): Number of simulation steps for each noise value.
    - n_particles (int): Number of particles in the simulation.
    - box_length (float): Length of the simulation box (assumed to be square).
    - speed (float): Constant speed of particles.

"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from vicsek.models.particle import Particle
from vicsek.models.vicsek import Vicsek


def analyze_equilibration(noise_values=[0.5, 2.25, 4.0], n_steps=1000,
                          n_particles=1024, box_length=16, speed=0.03):
    """Analyze how order parameter evolves over time for different noise values."""

    # Store order parameter evolution for each noise value
    evolution_data = {}

    for noise in tqdm(noise_values, desc="Processing noise values"):
        # Initialize particles
        particles = []
        for i in range(n_particles):
            position = np.random.uniform(0, box_length, size=2)
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
            particles.append(Particle(
                position=position,
                velocity=velocity,
                name=f"p{i}",
                type="standard"
            ))

        # Create model
        model = Vicsek(
            length=box_length,
            particles=particles,
            interaction_range=1.0,
            speed=speed,
            noise_factor=noise
        )

        # Track order parameter evolution
        order_params = []
        for _ in range(n_steps):
            model.step()
            order_params.append(model.order_parameter())

        evolution_data[noise] = order_params

    # Create plot
    plt.figure(figsize=(12, 8))
    for noise, order_params in evolution_data.items():
        plt.plot(order_params, label=f'η = {noise:.2f}', alpha=0.8)

    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Order Parameter φ', fontsize=12)
    plt.title('Order Parameter Evolution Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add threshold lines for visualization
    for noise, order_params in evolution_data.items():
        final_mean = np.mean(order_params[-100:])  # Mean of last 100 steps
        plt.axhline(y=final_mean, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    # Test with more noise values around the transition
    noise_values = [0.5, 1.5, 2.25, 3.0, 4.0]

    # Run analysis
    fig = analyze_equilibration(
        noise_values=noise_values,
        n_steps=1000,
        n_particles=1024,
        box_length=16
    )

    # Save plot
    plt.savefig('equilibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
