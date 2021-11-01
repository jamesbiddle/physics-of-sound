import numpy as np
from numpy import sin, cos, pi
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import os


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
        self.x = self.x0 + (self.p / self.k) * cos(self.k * self.x0 - self.w * t)

    def __str__(self):
        return f'x0 = {self.x0}, y0 = {self.y0}'


class Animator:
    def __init__(self):
        self.simulations = {}
        self.scatters = {}
        self.plots = {}
        self.widgets = {}

        self.key = 1

    def add_simulation(self, *args, **kwargs):
        self.simulations[self.key] = Simulation(*args, **kwargs)
        self.key += 1

    def remove_simulation(self, key):
        del self.simulations[key]

    def run(self):
        # Close all currently running simulations
        plt.close('all')
        self.setup_plot()
        self.p_animation = animation.FuncAnimation(self.fig,
                                                   self.p_update,
                                                   interval=20,
                                                   repeat=True,
                                                   repeat_delay=5,
                                                   frames=100,
                                                   blit=True)

        self.f_animation = animation.FuncAnimation(self.fig,
                                                   self.f_update,
                                                   interval=20,
                                                   repeat=True,
                                                   repeat_delay=5,
                                                   frames=100,
                                                   blit=True)

        # plt.show()
        return self.fig

    def get_interval(self):
        periods = [2 * pi / s.w for s in self.simulations]

    def setup_plot(self):
        n_sim = len(self.simulations)
        widths = [1, 15]
        heights = n_sim * [4, 1]

        self.fig = plt.figure(constrained_layout=True)
        gs = self.fig.add_gridspec(2 * n_sim, 2,
                                   width_ratios=widths,
                                   height_ratios=heights)
        for i, key in enumerate(self.simulations):
            ax_sim, ax_func, ax_slider = self.add_sim_grid(gs, i)

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

            # Set slider bars
            self.widgets[key] = self.add_slider(ax_slider, key)

    def add_slider(self, ax, key):
        sim = self.simulations[key]
        p_slider = Slider(
                ax=ax,
                label='Pressure',
                valmin=0,
                valmax=1,
                valinit=sim.p0,
                orientation='vertical'
            )

        def update(val):
            slider = self.widgets[key]
            sim.p0 = slider.val
            for p in sim.particles:
                p.p = slider.val
            self.fig.canvas.draw_idle()

        p_slider.on_changed(update)
        return p_slider

    def add_sim_grid(self, gs, i):
        ax1 = self.fig.add_subplot(gs[2 * i, 1])
        ax2 = self.fig.add_subplot(gs[2 * i + 1, 1])
        ax3 = self.fig.add_subplot(gs[2 * i:2 * i + 1, 0])
        return [ax1, ax2, ax3]

    def p_update(self, j):
        for key in self.simulations:
            sim = self.simulations[key]
            data = sim.update()
            self.scatters[key].set_offsets(data)

        return list(self.scatters.values())

    def f_update(self, j):
        for key in self.simulations:
            sim = self.simulations[key]
            x = sim.x_range
            y = sim.pressure()
            self.plots[key].set_data(x, y)

        return list(self.plots.values())


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
        self.interval = self.width / 100
        self.x_range = np.linspace(0, self.width)

        for i in range(self.n - 1):
            x, y = self.random_position()
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

    def random_position(self):
        y = rand() * self.height
        x = rand() * (self.width + 2 * self.p0) - self.p0
        return x, y

    def pressure(self):
        return self.p0 * sin(self.k * self.x_range - self.w * self.t)


if __name__ == '__main__':
    # sim = Simulation(0.1, 1, 0.5, 2000, width=2)
    sim = Animator()
    sim.add_simulation()
    anim = sim.run()
    plt.show()

    out_dir = './output/'
    filename = os.path.join(out_dir, 'animation.gif')
    # anim.save(filename,
    #           writer='imagemagick', fps=30)
