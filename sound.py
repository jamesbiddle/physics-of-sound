import numpy as np
from numpy import sin, cos, pi
from shutil import which
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess
import os
from math import gcd


def find_lcm(array):
    lcm = 1
    for i in array:
        lcm = lcm*i//gcd(lcm, i)
    return lcm


class Particle:
    def __init__(self, x0, y0, p, k, w):
        """Initialise the particle

        :param x0: Initial x position
        :param y0: Initial y position
        :param p: Pressure
        :param k: Wavenumber
        :param w: angular frequency

        """

        self.x0 = x0
        self.y0 = y0
        self.p = p
        self.k = k
        self.w = w

        self.x = x0
        self.y = y0

    def set_pos(self, t):
        """Set the x position of the particle at time t

        :param t: Time to evaluate the position

        """
        self.x = self.x0 + (self.p / self.k) * cos(self.k * self.x0 - self.w * t)

    def __str__(self):
        return f'x0 = {self.x0}, y0 = {self.y0}'


class Animator:
    def __init__(self):
        self.simulations = {}
        self.scatters = {}
        self.plots = {}

        self.key = 1

    def add_simulation(self, *args, **kwargs):
        self.simulations[self.key] = Simulation(*args, **kwargs)
        self.key += 1

    def reset(self):
        self.__init__()

    def save(self, filename, fps=30, dpi=None, compress=True):
        """Save the simulation as a gif

        :param filename: Output name
        :param fps: Framerate of the gif
        :param dpi: (optional, default=None) Gif dpi. If None, use imagemagick default
        :param compress: (Optional, default=True) Compress with gifsicle

        """
        if not which('convert'):
            print('Gif writer not found, cannot output animation.')
            return
        anim = self.run()
        if dpi is None:
            anim.save(filename,
                      writer='imagemagick', fps=fps)
        else:
            anim.save(filename,
                      writer='imagemagick', fps=fps, dpi=dpi)

        if compress:
            bashcmd = f'gifsicle -i {filename} -O3 --colors 128 -o {filename}'
            subprocess.run(bashcmd, shell=True)

        print(f'Saved {filename}')

    def run(self):
        self._setup_plot()
        frames = self._get_interval()
        self.animation = animation.FuncAnimation(self.fig,
                                                 self._update,
                                                 interval=40,
                                                 repeat=False,
                                                 frames=frames,
                                                 blit=True)

        return self.animation

    def _get_interval(self, max_frames=200):
        periods = [1 / self.simulations[key].f
                   for key in self.simulations]
        intervals = [self.simulations[key].interval
                     for key in self.simulations]
        frames = [round(T / i) for (T, i) in zip(periods, intervals)]
        frames = find_lcm(frames)
        frames = min(max_frames, frames)
        return frames

    def _setup_plot(self):
        """Initialise the plot once all simulations have been added

        """
        n_sim = len(self.simulations)
        heights = n_sim * [4, 1]
        default_height = 4.8
        default_width = 6.4
        size = [default_width, (n_sim / 2) * default_height]

        self.fig = plt.figure(constrained_layout=True, figsize=size)
        gs = self.fig.add_gridspec(2 * n_sim, 1,
                                   height_ratios=heights)
        for i, key in enumerate(self.simulations):
            ax_sim, ax_func = self._add_sim_grid(gs, i)

            # Setup particle sim
            sim = self.simulations[key]
            x, y = sim.initial_positions()
            n_part = sim.n
            colours = n_part * [(0, 0, 1)]
            colours[-1] = (1, 0, 0)
            self.scatters[key] = ax_sim.scatter(x, y, c=colours)
            ax_sim.set_xlim(0, sim.width)
            ax_sim.get_xaxis().set_visible(False)
            ax_sim.get_yaxis().set_visible(False)

            # Setup pressure function
            y = sim.pressure()
            self.plots[key], = ax_func.plot(sim.x_range, y)
            ax_func.set_ylim(-1, 1)
            ax_func.set_xlim(0, sim.width)
            ax_func.set_ylabel('Pressure')
            ax_func.set_xlabel('Position')

    def _add_sim_grid(self, gs, i):
        """Add a simulation to the animation

        :param gs: Gridspec to use
        :param i: Which simulation is being added
        :returns: list of axes

        """
        ax1 = self.fig.add_subplot(gs[2 * i, 0])
        ax2 = self.fig.add_subplot(gs[2 * i + 1, 0])
        return [ax1, ax2]

    def _update(self, j):
        for key in self.simulations:
            sim = self.simulations[key]
            data = sim.update()
            self.scatters[key].set_offsets(data)

            x = sim.x_range
            y = sim.pressure()
            self.plots[key].set_data(x, y)

        return list(self.scatters.values()) + list(self.plots.values())


class Simulation:
    def __init__(self, p0=0.5, l=1, f=1, n=500, width=2, height=1):
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
        self.interval = 0.01
        self.x_range = np.linspace(0, self.width, num=100)

        for i in range(self.n - 1):
            x, y = self._random_position()
            self.particles.append(Particle(x, y, self.p0, self.k, self.w))

        # Always put 1 particle in the middle to highlight
        self.particles.append(Particle(self.width / 2, self.height / 2,
                                       self.p0, self.k, self.w))

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

    def _random_position(self):
        y = rand() * self.height
        x = rand() * (self.width + 2 * self.p0) - self.p0
        return x, y

    def pressure(self):
        return self.p0 * sin(self.k * self.x_range - self.w * self.t)


if __name__ == '__main__':
    out_dir = './output/'

    sim = Animator()
    sim.add_simulation(p0=0.2)
    sim.add_simulation(p0=0.5)
    sim.add_simulation(p0=1)
    filename = os.path.join(out_dir, 'sound_p0_comp.gif')
    sim.save(filename, dpi=100)
    sim.reset()

    sim.add_simulation(l=0.2, p0=0.8)
    sim.add_simulation(l=1, p0=0.8)
    sim.add_simulation(l=2, p0=0.8)
    filename = os.path.join(out_dir, 'sound_l_comp.gif')
    sim.save(filename, dpi=100)
    sim.reset()

    sim.add_simulation(f=0.5, p0=0.8)
    sim.add_simulation(f=1, p0=0.8)
    sim.add_simulation(f=2, p0=0.8)
    filename = os.path.join(out_dir, 'sound_f_comp.gif')
    sim.save(filename, dpi=100)
    sim.reset()

    sim.add_simulation(f=0.5, p0=0.8, l=2)
    sim.add_simulation(f=1, p0=0.8, l=1)
    sim.add_simulation(f=2, p0=0.8, l=0.5)
    filename = os.path.join(out_dir, 'sound_samev.gif')
    sim.save(filename, dpi=100)
    sim.reset()
