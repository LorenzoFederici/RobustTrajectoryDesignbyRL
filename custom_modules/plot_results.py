import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import load_results, ts2xy
import pykep

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, out_folder = None, title='Learning_Curve', save_plot = True):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param out_folder: (str) the location where the plot is saved (default: log_folder)
    :param title: (str) the title of the task to plot
    :param save_plot: (bool) save the plot as pdf?
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    matplotlib.rc('font', size=14)
    matplotlib.rc('text', usetex=True)
    #fig1 = plt.figure() #figsize=(10, 10))
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Cumulative reward')
    #plt.xscale('log')
    plt.yscale('symlog')
    plt.grid()
    #plt.title("Learning curve smoothed")
    if (save_plot):
        if (out_folder is None):
            plt.savefig(log_folder + title + ".pdf", dpi=300)
        else:
            plt.savefig(out_folder + title + ".pdf", dpi=300)


def plot_kepler_new(r0, v0, r0_nom, v0_nom, tof, mu, N=60, units=1, color='b', label=None, axes=None, file_out=None):
    """
    ax = plot_kepler(r0, v0, tof, mu, N=60, units=1, color='b', label=None, axes=None):

    - axes:     3D axis object created using fig.gca(projection='3d')
    - r0:       initial position (cartesian coordinates)
    - v0:		initial velocity (cartesian coordinates)
    - r0_nom:   initial nominal position (cartesian coordinates)
    - v0_nom:	initial nominal velocity (cartesian coordinates)
    - tof:		propagation time
    - mu:		gravitational parameter
    - N:		number of points to be plotted along one arc
    - units:	how many times the differences between actual trajectory and nominal
        trajectory must be exaggerated
    - color:	matplotlib color to use to plot the line
    - label: 	adds a label to the plotted arc.
    - file_out: output file where to print the trajectory

    Plots the result of a keplerian propagation

    Example::

        import pykep as pk
        pi = 3.14
        pk.orbit_plots.plot_kepler(r0 = [1,0,0], v0 = [0,1,0], tof = pi/3, mu = 1)
    """

    from pykep.core import propagate_lagrangian
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    from copy import deepcopy

    if axes is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        ax = axes

    # We define the integration time ...
    dt = tof / (N - 1)

    # ... and calculate the cartesian components for r and v
    x = [0.0] * N
    y = [0.0] * N
    z = [0.0] * N
    vx = [0.0] * N
    vy = [0.0] * N
    vz = [0.0] * N

    # We calculate the spacecraft position and velocity at each dt
    r = deepcopy(r0)
    v = deepcopy(v0)
    r_nom = deepcopy(r0_nom)
    v_nom = deepcopy(v0_nom)
    for i in range(N):
        x[i] = r_nom[0] + (r[0] - r_nom[0])*units
        y[i] = r_nom[1] + (r[1] - r_nom[1])*units
        z[i] = r_nom[2] + (r[2] - r_nom[2])*units
        vx[i] = v_nom[0] + (v[0] - v_nom[0])*units
        vy[i] = v_nom[1] + (v[1] - v_nom[1])*units
        vz[i] = v_nom[2] + (v[2] - v_nom[2])*units
        r, v = propagate_lagrangian(r, v, dt, mu)
        r_nom, v_nom = propagate_lagrangian(r_nom, v_nom, dt, mu)

        if file_out is not None:
            file_out.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
                % (x[i], y[i], z[i], vx[i], vy[i], vz[i]))
        

    # And we plot
    ax.plot(x, y, z, c=color, label=label)
    return ax


def plot_taylor_new(r0, v0, m0, r0_nom, v0_nom, m0_nom, thrust, thrust_nom, tof, mu, veff, \
    N=60, units=1, color='b', label=None, axes=None):
    """
    ax = plot_taylor(r0, v0, m0, thrust, tof, mu, veff, N=60, units=1, color='b', legend=False, axes=None):

    - axes:		    3D axis object created using fig.gca(projection='3d')
    - r0:		    initial position (cartesian coordinates)
    - v0:		    initial velocity (cartesian coordinates)
    - m0: 		    initial mass
    - r0_nom:       initial nominal position (cartesian coordinates)
    - v0_nom:	    initial nominal velocity (cartesian coordinates)
    - m0_nom:       initial nominal mass
    - thrust:	    cartesian components for the constant thrust
    - thrust_nom:   cartesian components for the constant nominal thrust
    - tof:		    propagation time
    - mu:		    gravitational parameter
    - veff:	        the product Isp * g0
    - N:		    number of points to be plotted along one arc
    - units:	    the length unit to be used in the plot
    - color:	    matplotlib color to use to plot the line
    - label 	    adds a label to the plotted arc.

    Plots the result of a taylor propagation of constant thrust

    Example::

	import pykep as pk
	import matplotlib.pyplot as plt
	pi = 3.14

	fig = plt.figure()
	ax = fig.gca(projection = '3d')
	pk.orbit_plots.plot_taylor([1,0,0],[0,1,0],100,[1,1,0],40, 1, 1, N = 1000, axes = ax)
	plt.show()
    """

    from pykep.core import propagate_taylor
    import matplotlib.pyplot as plt
    from copy import deepcopy

    if axes is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        ax = axes

    # We define the integration time ...
    dt = tof / (N - 1)

    # ... and calcuate the cartesian components for r
    x = [0.0] * N
    y = [0.0] * N
    z = [0.0] * N

    # We calculate the spacecraft position at each dt
    r = deepcopy(r0)
    v = deepcopy(v0)
    m = deepcopy(m0)
    r_nom = deepcopy(r0_nom)
    v_nom = deepcopy(v0_nom)
    m_nom = deepcopy(m0_nom)
    for i in range(N):
        x[i] = r_nom[0] + (r[0] - r_nom[0])*units
        y[i] = r_nom[1] + (r[1] - r_nom[1])*units
        z[i] = r_nom[2] + (r[2] - r_nom[2])*units
        r, v, m = propagate_taylor(r, v, m, thrust, dt, mu, veff, -10, -10)
        r_nom, v_nom, m_nom = propagate_taylor(r_nom, v_nom, m_nom, thrust_nom, dt, mu, veff, -10, -10)

    # And we plot
    ax.plot(x, y, z, c=color, label=label)
    return ax

