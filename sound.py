import numpy as np
from numpy import sin, cos, arccos, arcsin, pi
from numpy.random import rand
from math import floor
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def sample_sin(k, a=None):
    if a is None:
        a = pi / k

    y = rand()
    x = arccos(1 - (1 - cos(k * a)) * y) / k
    return x


class Particle:
    def __init__(self, x0, y0, k, w):
        self.x0 = x0
        self.y0 = y0
        self.k = k
        self.w = w

        self.x = x0
        self.y = y0

    def set_pos(self, t):
        self.x = self.x0 - (self.w / self.k) * sin(self.k * self.x0 - self.w * t)

    def __str__(self):
        return f'x0 = {self.x0}, y0 = {self.y0}'


class Simulation:
    def __init__(self, l, f, n=100, width=1, height=1):
        """Initialise the simulation

        :param a: Amplitude of the sound wave
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

        self.l = l
        self.f = f

        self.k = 2 * pi / l
        self.w = 2 * pi * f
        self.a = self.w / self.k
        self.t = 0
        self.interval = self.width / 100

        for i in range(self.n):
            x, y = self.random_position()
            self.particles.append(Particle(x, y, self.k, self.w))

        self.fig, self.ax = plt.subplots()

    def show_init(self):
        self.setup_plot()

    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        x = [p.x0 for p in self.particles]
        y = [p.y0 for p in self.particles]
        self.scat = self.ax.scatter(x, y)
        self.ax.set_xlim(0, self.width)
        return self.scat,

    def update(self, i):
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
        self.scat.set_offsets(data)

        return self.scat,

    def random_position(self):
        y = rand() * self.height
        x = rand() * (self.width + 2 * self.a) - self.a

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
    sim = Simulation(1, 0.1, 1000, width=2)
    # print(sim.random_position())
    # sim.show_init()
    sim.run()
    plt.show()
