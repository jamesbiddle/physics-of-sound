import numpy as np
from numpy import sin, cos, arccos, arcsin, pi
from numpy.random import rand
from math import floor, ceil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def sample_sin(k, a=None):
    if a is None:
        a = pi / k

    y = rand()
    x = arccos(1 - (1 - cos(k * a)) * y) / k
    return x


class Particle:
    def __init__(self, x0, y0, p, k, w):
        self.x0 = x0
        self.y0 = y0
        self.p = p
        self.k = k
        self.w = w

        self.x = x0
        self.y = y0

    def set_pos(self, t):
        self.x = self.x0 - (self.p / self.k) * sin(self.k * self.x0 - self.w * t)

    def __str__(self):
        return f'x0 = {self.x0}, y0 = {self.y0}'


class Animator:
    def __init__(self):
        self.simulations = {}
        self.scatters = []

    def add_simulation(self, key, *args, **kwargs):
        self.simulations[key] = Simulation(*args, **kwargs)

    def remove_simulation(self, key):
        del self.simulations[key]

    def run(self):
        n_sim = len(self.simulations)
        # n_rows = ceil(np.sqrt(n_sim))
        # n_cols = ceil(n_sim / n_rows)

        # self.fig, self.axes = plt.subplots(n_rows, n_cols)
        self.fig, self.axes = plt.subplots(n_sim)
        self.setup_plot()
        self.animation =animation.FuncAnimation(self.fig,
                                                sim.update,
                                                interval=5,
                                                repeat=False,
                                                frames=50,
                                                # save_count=100,
                                                blit=True)
        # plt.show()
        return self.animation

    def setup_plot(self):
        for ax, key in zip(self.axes.flatten(), self.simulations):
            sim = self.simulations[key]
            x, y = sim.initial_positions()
            self.scatters.append(ax.scatter(x, y))
            ax.set_xlim(0, sim.width)

    def update(self, j):
        for i, (ax, key) in enumerate(zip(self.axes.flatten(), self.simulations)):
            sim = self.simulations[key]
            data = sim.update()
            self.scatters[i].set_offsets(data)

        return self.scatters


class Simulation:
    def __init__(self, p0=1, l=1, f=1, n=100, width=1, height=1):
        """Initialise the simulation

        :param p0: Pressure of the sound wave
        :param l: Wavelength
        :param f: frequency
        :param n: (Optional, default=100) Number of particles
        :param width: (Optional, default=1) Width of the box
        :param height: (Optional, default=1) Height of the box

        """
        self.width = width
        self.height = height
        self.n = n
        self.particles = []

        self.p0 = p0
        self.l = l
        self.f = f

        self.k = 2 * pi / l
        self.w = 2 * pi * f
        self.t = 0
        self.interval = self.width / 100

        for i in range(self.n):
            x, y = self.random_position()
            self.particles.append(Particle(x, y, self.p0, self.k, self.w))

    def initial_positions(self):
        x = [p.x0 for p in self.particles]
        y = [p.y0 for p in self.particles]
        return x, y

    def update(self):
        self.t += self.interval
        # Use periodicity to prevent overflow
        T = 2 * pi / self.w
        if self.t > T:
            self.t = 0

        for p in self.particles:
            p.set_pos(self.t)

        x = [p.x for p in self.particles]
        y = [p.y for p in self.particles]

        data = np.array([x, y]).T
        return data

    def random_position(self):
        y = rand() * self.height
        x = rand() * (self.width + 2 * self.p0) - self.p0

        # How many half-wavelengths fit in the box
        # n_regions = floor(self.width / (self.l / 2))
        # residual_width = self.width - n_regions * self.l / 2
        # residual_frac = residual_width / self.width
        # if n_regions > 0:
        #     offset = (self.width - residual_width) / n_regions
        # else:
        #     offset = 0
        # r = floor(rand() * (n_regions + residual_frac))
        # if r < n_regions:
        #     x = sample_sin(self.k) + r * offset
        # else:
        #     x = sample_sin(self.k, residual_width) + n_regions * offset

        return x, y


if __name__ == '__main__':
    # sim = Simulation(0.1, 1, 0.5, 2000, width=2)
    sim = Animator()
    sim.add_simulation(1, p0=0.1, l=1, f=0.5, n=2000, width=2)
    # sim.add_simulation(2, p0=0.5, l=1, f=0.5, n=2000, width=2)
    sim.add_simulation(3, p0=0.5, l=0.5, f=1, n=2000, width=2)
    anim = sim.run()
    out_dir = './output/'
    filename = os.path.join(out_dir, 'animation.gif')
    anim.save(filename,
              writer='imagemagick', fps=30)

    # print(sim.random_position())
    # sim.show_init()
