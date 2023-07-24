import matplotlib.pyplot as plt
import numpy as np

# The metrics are recorded for the 50th epoch (last epoch)

# A is the noise level, it is the y axis
# Am = A for multi hand
Am = [0.831679, 0.829200, 0.82574, 0.82246, 0.8167, 0.80418, 0.79858]
# As = A for single hand
As = [0.89474, 0.893280, 0.89512, 0.8858, 0.88606, 0.88724, 0.88456, 0.87289, 0.862669, 0.8375]

# C is the noise level, it is the x axis
# Cm = C for multi hand
Cm = np.array([0.0, 1.01, 1.98, 2.02, 4.82, 7.20, 10.13])
# Cs = C for single hand
Cs = np.array([0.0, 0.20, 0.35, 0.55, 1.00, 1.50, 2.01, 3.49, 5.01, 9.50])

# Regression for multi hand
Pm = np.polyfit(Cm, Am, 1)
Rm = np.corrcoef(Cm, Am)
# Regression for single hand
Ps = np.polyfit(Cs, As, 1)
Rs = np.corrcoef(Cs, As)


# Plotting
plt.plot(Cm, Am, 'o', label='multi-hand exp', color='red')
plt.plot(Cs, As, 'o', label='single-hand exp', color='blue')
# cf. https://youtu.be/4cNWBBOEck0
plt.plot(Cm, Pm[0] * Cm + Pm[1], label='multi-hand regression', color='orange')
plt.plot(Cs, Ps[0] * Cs + Ps[1], label='single-hand regression', color='turquoise')

plt.xlabel('Noise level (%)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy (at epoch 50, on noiseless test set) \nas a function of the noise level in the training set')
plt.grid()
plt.legend()
plt.text(0.42, 0.35, f"multi-hand regression: y = {Pm[0]:.3f}x + {Pm[1]:.3f}\nr = {Rm[0,1]:.3f}", transform=plt.gca().transAxes)
plt.text(0.42, 0.25, f"single-hand regression: y = {Ps[0]:.3f}x + {Ps[1]:.3f}\nr = {Rs[0,1]:.3f}", transform=plt.gca().transAxes)

# make the plot bigger
plt.gcf().set_size_inches(10, 6)

plt.savefig('regressions.png')
#plt.show()


