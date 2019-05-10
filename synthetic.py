import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.integrate import odeint

from embedding import *


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

    def lorenz(self, x0=None, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
        """Generate time series using the Lorenz system.

        Generates time series using the Lorenz system.

        Parameters
        ----------
        length : int, optional (default = 10000)
            Length of the time series to be generated.
        x0 : array, optional (default = random)
            Initial condition for the flow.
        sigma : float, optional (default = 10.0)
            Constant sigma of the Lorenz system.
        beta : float, optional (default = 8.0/3.0)
            Constant beta of the Lorenz system.
        rho : float, optional (default = 28.0)
            Constant rho of the Lorenz system.
        step : float, optional (default = 0.001)
            Approximate step size of integration.
        sample : int, optional (default = 0.03)
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

        def _lorenz(x, t):
            return [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1],
                    x[0] * x[1] - beta * x[2]]

        if not x0:
            x0 = (0.0, -0.01, 9.0) + 0.25 * (-1 + 2 * np.random.random(3))

        sample = int(self.sample / self.step)
        t = np.linspace(0, (sample * (self.length + self.discard)) * self.step,
                        sample * (self.length + self.discard))

        return (t[self.discard * sample::sample],
                odeint(_lorenz, x0, t)[self.discard * sample::sample])


def drawRossler():
    attractor = Attractors()
    data = attractor.rossler()[1]
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='Аттрактор Рёсслера', linewidth=0.5)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def Rossler_delay():
    attractor = Attractors()
    x = attractor.rossler()[1][:, 0]

    _embedding = Embedding(x)

    lag = np.arange(250)
    r = _embedding.autocorrelation(maxtau=250)
    i = _embedding.time_delayed_mutual_information(maxtau=250)

    i_delay = _embedding.locmin(i)
    r_delay = np.argmax(np.round(r, 1) == 0.0)

    print(r'Minima of delayed mutual information = %s' % i_delay)
    print(r'Autocorrelation time = %d' % r_delay)

    plt.figure(1)

    plt.subplot(211)
    # plt.title(r'Оценка временного лага $\tau$ для аттрактора Рёсслера')
    # plt.ylabel(r'Delayed mutual information')
    plt.plot(lag, i, label='функция взаимной информации')
    plt.plot(i_delay, i[i_delay], 'o', label='локальные минимумы')
    plt.legend(loc='upper right')

    plt.subplot(212)
    plt.xlabel(r'Временной лаг $\tau$')
    # plt.ylabel(r'Autocorrelation')
    plt.plot(lag, r, label='функция автокорреляции')
    plt.plot(r_delay, r[r_delay], 'o', label='первое нулевое значение')
    plt.ylim(top=2.0, bottom=-2.0)
    plt.legend(loc='upper right')

    plt.figure(2)
    # plt.subplot(121)
    plt.title(r'Временной лаг $\tau$= %d' % i_delay[0])
    plt.xlabel(r'$x(t)$')
    plt.ylabel(r'$x(t + \tau)$')
    plt.plot(x[:-i_delay[0]], x[i_delay[0]:])

    # plt.subplot(122)
    # plt.title(r'Time delay = %d' % r_delay)
    # plt.xlabel(r'$x(t)$')
    # plt.ylabel(r'$x(t + \tau)$')
    # plt.plot(x[:-r_delay], x[r_delay:])

    plt.show()


def Rossler_dimension1():
    attractor = Attractors()
    x = attractor.rossler()[1][:, 0]

    _embedding = Embedding(x)

    dim = np.arange(1, 10 + 1)
    f1, f2, f3 = _embedding.fnn(x, tau=14, dim=dim, window=10, metric='cityblock')
    _embedding.plot_fnn(dim, f1, f2, f3)


def Rossler_dimension2():
    attractor = Attractors()
    x = attractor.rossler()[1][:, 0]

    _embedding = Embedding(x)

    dim = np.arange(1, 10 + 2)
    E, Es = _embedding.afn(x, tau=14, dim=dim, window=45, metric='chebyshev')
    E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]
    _embedding.plot_afn(dim, E1, E2)


if __name__ == '__main__':
    # drawRossler()
    Rossler_delay()
    # Rossler_dimension1()
    # Rossler_dimension2()
