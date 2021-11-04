import numpy as np
from numpy import sin, cos, pi
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from ipywidgets import interact, interact_manual
from ipywidgets import FloatSlider
from math import ceil
import os
from math import gcd


def find_lcm(array):
    lcm = 1
    for i in array:
        lcm = lcm*i//gcd(lcm, i)
    return lcm


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

        self.anim = None
        self.key = 1

    def add_simulation(self, *args, **kwargs):
        self.simulations[self.key] = Simulation(*args, **kwargs)
        self.key += 1

    def remove_simulation(self, key):
        del self.simulations[key]

    def run(self):
        self.setup_plot()
        # for sim in self.simulations():
        interact(self.animate,
                 p0=FloatSlider(min=0, max=1, step=0.1, value=0.5, continuous_update=False),
                 # l=FloatSlider(min=0.1, max=1, step=0.1, value=0.5, continuous_update=False),
                 f=FloatSlider(min=0.1, max=1, step=0.1, value=0.5, continuous_update=False))

    def animate(self, p0=None, l=None, f=None):
        if self.anim is not None:
            self.anim.event_source.stop()
        # frames = self._get_interval()
        frames = 200
        self.anim = animation.FuncAnimation(self.fig,
                                            self.update,
                                            fargs=(p0, l, f),
                                            interval=40,
                                            repeat=True,
                                            # frames=frames,
                                            blit=True)
        self.anim.frame_seq = self.anim.new_frame_seq()
        self.anim.event_source.start()
        plt.show()
        # return self.anim

    def _get_interval(self, max_frames=200):
        periods = [1 / self.simulations[key].f
                   for key in self.simulations]
        print(periods)
        intervals = [self.simulations[key].interval
                     for key in self.simulations]
        frames = [round(T / i) for (T, i) in zip(periods, intervals)]
        print(frames)
        frames = find_lcm(frames)
        print(frames)
        return min(frames, max_frames)

    def setup_plot(self):
        n_sim = len(self.simulations)
        # widths = [1, 15]
        heights = n_sim * [4, 1]

        self.fig = plt.figure(constrained_layout=True)
        gs = self.fig.add_gridspec(2 * n_sim,
                                   # width_ratios=widths,
                                   height_ratios=heights)
        for i, key in enumerate(self.simulations):
            ax_sim, ax_func = self.add_sim_grid(gs, i)

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

            # # Set slider bars
            # self.widgets[key] = self.add_slider(ax_slider, key)

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
        ax1 = self.fig.add_subplot(gs[2 * i, 0])
        ax2 = self.fig.add_subplot(gs[2 * i + 1, 0])
        return [ax1, ax2]

    def update(self, i, p0=None, l=None, f=None):
        for key in self.simulations:
            sim = self.simulations[key]
            sim.update_parameters(p0, l, f)
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
        self.interval = 0.05
        self.x_range = np.linspace(0, self.width, 100 * self.width)

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

    def update_parameters(self, p=None, l=None, f=None):
        if p is None:
            p = self.p0
        if l is None:
            l = self.l
        if f is None:
            f = self.f

        k = 2 * pi / l
        w = 2 * pi * f

        self.p0 = p
        self.l = l
        self.f = f
        self.k = k
        self.w = w

        for part in self.particles:
            part.p = p
            part.w = w
            part.k = k


if __name__ == '__main__':
    sim = Animator()
    sim.add_simulation(f=3)
    sim.run()
    # sim.add_simulation(f=2)
    # sim.setup_plot()
    # anim = sim.animate(0.5, 0.5, 0.5)
    # plt.show()

    # out_dir = './output/'
    # filename = os.path.join(out_dir, 'animation.html')
    # html = anim.to_html5_video()
    # with open(filename, 'w') as f:
    #     f.write(html)
    # plt.close('all')
    # anim.save(filename,
    #           writer='imagemagick', fps=30)
