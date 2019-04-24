import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import odeint


class Attractors:
    def __init__(self, length=10000, step=0.001, sample=0.1, discard=1000):
        self.length = length
        self.step = step
        self.sample = sample
        self.discard = discard

    def rossler(self, x0=None, a=0.2, b=0.2, c=5.7):
        """Generate time series using the Rössler oscillator.

        Generates time series using the Rössler oscillator.

        Parameters
        ----------
        length : int, optional (default = 10000)
            Length of the time series to be generated.
        x0 : array, optional (default = random)
            Initial condition for the flow.
        a : float, optional (default = 0.2)
            Constant a in the Röessler oscillator.
        b : float, optional (default = 0.2)
            Constant b in the Röessler oscillator.
        c : float, optional (default = 5.7)
            Constant c in the Röessler oscillator.
        step : float, optional (default = 0.001)
            Approximate step size of integration.
        sample : int, optional (default = 0.1)
            Sampling step of the time series.
        discard : int, optional (default = 1000)
            Number of samples to discard in order to eliminate transients.

        Returns
        -------
        t : array
            The time values at which the points have been sampled.
        x : ndarray, shape (length, 3)
            Array containing points in phase space.
        """

        def _rossler(x, t):
            return [-(x[1] + x[2]), x[0] + a * x[1], b + x[2] * (x[0] - c)]

        sample = int(self.sample / self.step)
        t = np.linspace(0, (sample * (self.length + self.discard)) * self.step,
                        sample * (self.length + self.discard))

        if not x0:
            x0 = (-9.0, 0.0, 0.0) + 0.25 * (-1 + 2 * np.random.random(3))

        return (t[self.discard * sample::sample],
                odeint(_rossler, x0, t)[self.discard * sample::sample])


def drawRossler():
    attractor = Attractors()
    data = attractor.rossler()[1]
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='Rössler oscillator', linewidth=0.5)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


drawRossler()
