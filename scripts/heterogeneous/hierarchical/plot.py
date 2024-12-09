import numpy as np
import matplotlib.pyplot as plt


legend = {
    'leader': 'red',
    'follower': 'blue',
}

noise_values = np.arange(0, 2, 0.1)
global_order = np.load('global_order.npy', allow_pickle=True)
global_fluctuations = np.load('global_fluctuations.npy', allow_pickle=True)
type_fluctuations = np.load('type_fluctuations.npy', allow_pickle=True).tolist()
type_orders = np.load('type_orders.npy', allow_pickle=True).tolist()
cross_correlations = np.load('cross_correlations.npy', allow_pickle=True).tolist()

print(cross_correlations)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

# Plot order parameters
ax1.plot(noise_values, global_order, 'k-', label='Global', linewidth=2)
for ptype, orders in type_orders.items():
    ax1.plot(noise_values, orders, color=f'{legend[ptype]}', label=f'{ptype}', linewidth=2)

ax1.set_xlabel('Noise')
ax1.set_ylabel('Order Parameter')
ax1.set_title('Order Parameters vs Noise')
ax1.legend()
ax1.grid(True)

# Plot fluctuations
ax2.plot(noise_values, global_fluctuations, 'k-', label='Global', linewidth=2)
for ptype, fluct in type_fluctuations.items():
    ax2.plot(noise_values, fluct, color=f'{legend[ptype]}', label=f'{ptype}', linewidth=2)

ax2.set_xlabel('Noise')
ax2.set_ylabel('Susceptibility')
ax2.set_title('Fluctuations vs Noise')
ax2.legend()
ax2.grid(True)

# Plot cross-correlations
for pair, corr in cross_correlations.items():
    try:
        if pair[0] != pair[1]:
            ax3.plot(noise_values, corr, '-', color='black')
    except ValueError:
        pass

ax3.set_xlabel('Noise')
ax3.set_ylabel('Cross-correlation')
ax3.set_title('Velocity Cross-correlations vs Noise')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig('plot.png')
plt.show()

