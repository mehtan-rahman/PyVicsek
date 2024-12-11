# PyVicsek

This is a Python library used to simulate systems using the Vicsek model. It currently has support for non periodic-boundary conditions, $N$-dimensional space, as well as particle heterogeneity.

## Dependencies and Installation

VicsekPy is best run on Python 3.9+ and is supported by [numpy](https://github.com/numpy/numpy), [scipy](https://github.com/scipy/scipy), [matplotlib](https://github.com/matplotlib/matplotlib), and [tqdm](https://github.com/tqdm/tqdm). Additionally, unit tests are made with [pytest](https://github.com/pytest-dev/pytest). These packages are automatically included in `requirements.txt`. 

To download VicsekPy, run the following commands in a UNIX terminal: 
```
pip install pyvicsek
```

## Examples

Here is a basic script to initialize and create a 2-D homogenous Vicsek system. 

```python
import vicsek as vs

N = 1024
L = 16
v = 0.03

particles = vs.initialize_random_particles(
    n_particles=N,
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='standard'
)

vicsek = vs.Vicsek(
    particles=particles,
    length=L,
    interaction_range=1.0,
    speed=v,
    base_noise=0.5,
)

anim = vicsek.animate(frames=200)
anim.save(filename='example.gif')
```

![](https://github.com/mehtan-rahman/PyVicsek/blob/main/scripts/homogeneous/example.gif)

Here is another example to initialize a 2-D heterogeneous Vicsek system. 

```python
import vicsek as vs

N = 1024
L = 16
v = 0.03

particles_A = vs.initialize_random_particles(
    n_particles=round(N * 0.5),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='A'
)

particles_B = vs.initialize_random_particles(
    n_particles=round(N * 0.5),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='B'
)

particles = particles_A + particles_B

alignment_weights = {
    ('A', 'A'): 1.0,
    ('B', 'B'): 1.0,
    ('A', 'B'): -1.0,
    ('B', 'A'): -1.0,
}

noise_weights = {
    ('A', 'A'): 0.5,  # low noise within groups
    ('B', 'B'): 0.5,
    ('A', 'B'): 2.0,  # high noise between groups
    ('B', 'A'): 2.0,
}

legend = {
    'A': 'red',
    'B': 'blue',
}


vicsek = vs.HeterogeneousVicsek(
    particles=particles,
    length=L,
    interaction_range=1.0,
    speed=v,
    base_noise=0.5,
    alignment_weights=alignment_weights,
    noise_weights=noise_weights
)

anim = vicsek.animate(frames=200, legend=legend)
anim.save(filename='hetero_example.gif')
```

![](https://github.com/mehtan-rahman/PyVicsek/blob/main/scripts/heterogeneous/antagonistic/antagonistic.gif)

You can find more information on the `Vicsek` class as well as other classes in the doc-strings located in the files of the `vicsek` directory. You can also find further examples in the `scripts` directory, where they are organized by system type.

## Roadmap
If given more time, we would like to spend some time working on the following things for this library: 
- Higher dimensional testing.
- Lyapunov analysis of phase dimensions (based on this [paper](https://doi.org/10.1103/PhysRevE.105.014213)). 
- Geometry support for particles and boundaries using [shapely](https://github.com/shapely/shapely). 
