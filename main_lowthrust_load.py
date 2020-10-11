import os
import warnings
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
import gym
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, LstmPolicy
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, DDPG, SAC, TD3, TRPO
from custom_modules.custom_policies import CustomPolicy_3x64, CustomPolicy_4x128, CustomLSTMPolicy
from custom_modules.learning_schedules import linear_schedule
from custom_modules.plot_results import plot_results, plot_kepler_new, plot_taylor_new
from custom_modules.set_axes_equal_3d import set_axes_equal
import argparse
import gym_lowthrust
import pykep as pk
from pykep.core import epoch_from_string, ic2par, propagate_lagrangian
from pykep.planet import jpl_lp

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# Input data
plot_rewards = True
nominal = False

#Input settings file
parser = argparse.ArgumentParser()
parser.add_argument('--settings', type=str, default="sol_saved/sol_1/settings.txt", \
    help='Input settings file')
parser.add_argument('--n_sol', type=int, default=1, \
    help='Number of solution to post-process')
args = parser.parse_args()
settings_file = args.settings
n_sol = args.n_sol

#Read settings and assign environment and model parameters
with open(settings_file, "r") as input_file: # with open context
    input_file_all = input_file.readlines()
    for line in input_file_all: #read line
        line = line.split()
        if (len(line) > 2):
            globals()[line[0]] = line[1:]
        else:
            globals()[line[0]] = line[1]
        
input_file.close() #close file

#Environment parameters
impulsive = bool(int(impulsive)) #type of thrust model
action_coord = bool(int(action_coord)) #type of coordinates for actions
obs_type = int(obs_type) #type of observations
random_obs = bool(int(random_obs)) #random observations
stochastic = bool(int(stochastic)) #stochastic environment
mission_type = bool(int(mission_type)) # False = interception, True = rendezvous
if (mission_type == False):
    str_mission = "interception"
else:
    str_mission = "rendezvous"
NSTEPS = int(NSTEPS) # Total number of trajectory segments
if isinstance(eps_schedule, list):
    eps_schedule = [float(i) for i in eps_schedule] #epsilon-constraint schedule
else:
    eps_schedule = [float(eps_schedule)]
lambda_con = float(lambda_con) #weight of constraint violation in reward
sigma_r = float(sigma_r) #standard deviation of position
sigma_v = float(sigma_v) #standard deviation of velocity
sigma_u_rot = float(sigma_u_rot) #standard deviation of control rotation
sigma_u_norm = float(sigma_u_norm) #standard deviation of control modulus
MTE = bool(int(MTE)) #at least one MTE occurs?
pr_MTE = float(pr_MTE) #probability of having a MTE at k-th step

#Model parameters
num_cpu = int(num_cpu) #number of environments
policy = globals()[policy]

#Input Model and Output folders
in_folder = out_folder_root = "./sol_saved/sol_" + str(n_sol) + "/"
monitor_folder = in_folder + algorithm + "/"
if (stochastic == False) and (random_obs == False):
    logname = "best_model"
else:
    logname = "final_model"
trained_model = in_folder + logname
plot_folder = in_folder

#Physical constants
amu = 132712440018.             # km^3/s^2, Gravitational constant of the central body
rconv = 149600000.              # position, km
vconv = np.sqrt(amu/rconv)      # velocity, km/s
tconv = rconv/vconv             # time, s
mconv = 1000.                   # mass, kg
aconv = vconv/tconv             # acceleration, km/s^2
fconv = mconv*aconv             # force, kN
dim_conv = np.array([tconv, rconv, rconv, rconv, \
    vconv, vconv, vconv, mconv]).astype(np.float64)

#Spacecraft data (nondimensional)
Tmax = float(Tmax)/fconv             # Maximum thrust
Isp = float(Isp).                    # s, Specific impulse
ueq = (pk.G0/1000.*Isp)/vconv   # Equivalent ejection velocity

#Read Mission file
t_nom = [] #nominal trajectory: time
rx_nom = [] #nominal trajectory: x
ry_nom = [] #nominal trajectory: y
rz_nom = [] #nominal trajectory: z
vx_nom = [] #nominal trajectory: vx
vy_nom = [] #nominal trajectory: vy
vz_nom = [] #nominal trajectory: vz
m_nom = [] #nominal trajectory: m
mission_folder = "missions/"
mission_file = mission_folder + mission_name + ".dat" #File with mission data
with open(mission_file, "r") as f: # with open context
    f.readline()
    file_all = f.readlines()
    for line in file_all: #read line
        line = line.split()
        state_adim = np.array(line).astype(np.float64) #non-dimensional data
        #state_dim = np.multiply(state_adim, dim_conv) #dimensional data
        
        #save data
        t_nom.append(state_adim[0])
        rx_nom.append(state_adim[1])
        ry_nom.append(state_adim[2])
        rz_nom.append(state_adim[3])
        vx_nom.append(state_adim[4])
        vy_nom.append(state_adim[5])
        vz_nom.append(state_adim[6])
        m_nom.append(state_adim[7])

f.close() #close file

#Mission data

#Time-of-flight
tf =  t_nom[-1] - t_nom[0]   #s, Time-of-flight
dt = tf/NSTEPS               #s, time-step

#Reference initial state
m0 = m_nom[0] #kg, initial spacecraft mass
r0 = [rx_nom[0], ry_nom[0], rz_nom[0]] #km, initial spacecraft position
v0 = [vx_nom[0], vy_nom[0], vz_nom[0]] #km/s, initial spacecraft velocity

#Target initial state
rTf = [rx_nom[-1], ry_nom[-1], rz_nom[-1]]      #km, final target position
vTf = [vx_nom[-1], vy_nom[-1], vz_nom[-1]]      #km/s, final target velocity
coeT = ic2par(r = rTf, v = vTf)                 #orbital elements of the target
PT = 2.*np.pi*np.sqrt(coeT[0]**3)               #s, target orbital period
rT0, vT0 = propagate_lagrangian(r0 = rTf, v0 = vTf, tof = PT - tf) #km, km/s, initial target position and velocity

#Epsilon constraint schedule
eps_schedule = [eps_schedule[-1]]

# Environment creation
env0 = gym.make(env_name, impulsive = impulsive, 
    action_coord = action_coord, obs_type = obs_type, \
    random_obs = False, stochastic = False, \
    mission_type = mission_type, NSTEPS = NSTEPS, \
    NITER = NSTEPS, eps_schedule = eps_schedule, \
    lambda_con = lambda_con, \
    Tmax = Tmax, ueq = ueq, tf = tf, amu = 1., m0 = m0, \
    r0 = r0, v0 = v0, \
    rTf = rTf, vTf = vTf, \
    sigma_r = sigma_r, sigma_v = sigma_v, \
    sigma_u_rot = sigma_u_rot, sigma_u_norm = sigma_u_norm, \
    MTE = MTE, pr_MTE = pr_MTE)
env = DummyVecEnv([lambda: env0])

# Load model
model = PPO2.load(trained_model, env=env, policy=policy)

# Print graph and results
f_out = open(in_folder + "Simulation.txt", "w") # open file
f_out.write("Environment simulation\n\n")
f_out_EM = open(in_folder + "Traj_Earth_Mars.txt", "w") # open file
f_out_EM.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
        % ("# x", "y", "z", "vx", "vy", "vz"))
if impulsive == True:   
    f_out_traj = open(in_folder + "Trajectory.txt", "w") # open file
    f_out_u = open(in_folder + "control.txt", "w") # open file
    f_out_traj.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
        % ("# x", "y", "z", "vx", "vy", "vz"))
    f_out_u.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
        % ("# t", "x", "y", "z", "DVx", "DVy", "DVz", "DVnorm", "DVmax", "m"))
np.set_printoptions(precision=3)

#Plot episode rewards
if (plot_rewards):
    for rank in range(num_cpu-1):
        plot_results(log_folder = monitor_folder + "env_" + str(rank) + "/", save_plot = False)
    plot_results(log_folder = monitor_folder + "env_" + str(num_cpu-1) + "/", \
        out_folder = monitor_folder)

#Create figures
matplotlib.rc('font', size=18)
matplotlib.rc('text', usetex=True)
fig1 = plt.figure(figsize=(10, 10)) #figsize=(10, 10)
ax = fig1.gca(projection='3d')

fig2 = plt.figure()
ax2 = fig2.gca()

#Departure orbit
plot_kepler_new(r0 = np.array(r0), v0 = np.array(v0), r0_nom = np.array(r0), v0_nom = np.array(v0), tof = tf, \
    mu = 1., N = 400, axes = ax, color='k', file_out = f_out_EM)
f_out_EM.write("\n\n");

#Arrival orbit
plot_kepler_new(r0 = np.array(rT0), v0 = np.array(vT0), r0_nom = np.array(rT0), v0_nom = np.array(vT0), tof = tf, \
    mu = 1., N = 400, axes = ax, color='k', file_out = f_out_EM)
f_out_EM.close()

#Nominal solution
if nominal == True:
    ax.plot(np.array(rx_nom), np.array(ry_nom), np.array(rz_nom), '--', color='red', \
        linewidth = 2.5, label='$\\mbox{nominal}$')

#Reset environment
obs = env0.reset()
cumulative_reward = 0.


for i in range(NSTEPS):
    
    #Get current action
    action, _states = model.predict(obs, deterministic=True)

    #Get new observation
    obs, reward, done, info = env0.step(action)

    #Spacecraft state, time and control
    r = np.array([info["rx"], info["ry"], info["rz"]])
    v = np.array([info["vx"], info["vy"], info["vz"]])
    m = info["m"]
    t = info["t"]
    u = np.array([info["ux"], info["uy"], info["uz"]])
    if impulsive == True:
        u_max = Tmax/m*dt
    else:
        u_max = Tmax

    #Print trajectory information
    f_out.write("t_step = " + str(np.round(t/dt)) + "\n")
    f_out.write("norm(r) = " + str(norm(r)) + "\n")
    f_out.write("norm(v) = " + str(norm(v)) + "\n")
    f_out.write("norm(u/u_max) = " + str(norm(u)/u_max) + "\n")
    f_out.write("m = " + str(m) + "\n")
    f_out.write("cum_reward = " + str(cumulative_reward) + "\n\n")

    #Plot trajectory segment
    if impulsive == True:
        if (i == 0) and (nominal == True):
            plot_kepler_new(r0 = r, v0 = (v + u), r0_nom = r, v0_nom = (v + u), tof = dt, \
                mu = 1., N = 10, axes = ax, label = '$\\pi^{unp}$', file_out = f_out_traj)
        else:
            plot_kepler_new(r0 = r, v0 = (v + u), r0_nom = r, v0_nom = (v + u), tof = dt, \
                mu = 1., N = 10, axes = ax, file_out = f_out_traj)
    else:
        pk.orbit_plots.plot_taylor(r0 = r, v0 = v, m0 = m, thrust = u, tof = dt, mu = 1., \
            veff = ueq, axes=ax)

    #Add u vector
    if impulsive == True:
        f_out_u.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
            % (t, r[0], r[1], r[2], u[0], u[1], u[2], norm(u), u_max, m))
        ax.quiver(r[0], r[1], r[2], u[0], u[1], u[2]/3, length=20)
    else:
        ax.quiver(r[0], r[1], r[2], u[0], u[1], u[2]/3, length=5)

    #Plot control
    points = 60
    t_vec = np.linspace(i*dt/tf, (i+1)*dt/tf, points)
    if impulsive == False:
        u_vec = norm(u)/Tmax*np.ones(points)
        ux_vec = u[0]/Tmax*np.ones(points)
        uy_vec = u[1]/Tmax*np.ones(points)
        uz_vec = u[2]/Tmax*np.ones(points)
    else:
        u_vec = norm(u)
    if i == 0:
        if impulsive == True:
            ax2.stem([t_vec[0]], [u_vec], '-k')
        else:
            ax2.plot(t_vec, u_vec, '-k', label = "$ || T || $")
            ax2.plot(t_vec, ux_vec, '-b', label = "$ T_x $")
            ax2.plot(t_vec, uy_vec, '-r', label = "$ T_y $")
            ax2.plot(t_vec, uz_vec, '-g', label = "$ T_z $")
    else:
        if impulsive == True:
            #ax2.stem([t_vec[0]], [u_max], '--k')
            ax2.stem([t_vec[0]], [u_vec], '-k')
        else:
            ax2.plot(t_vec, u_vec, '-k')
            ax2.plot(t_vec, ux_vec, '-b')
            ax2.plot(t_vec, uy_vec, '-r')
            ax2.plot(t_vec, uz_vec, '-g')

    #Update cumulative reward
    cumulative_reward += reward

#Final state
t = env0.t
r = env0.rk
v = env0.vkm
m = env0.mk

rTf = np.array(rTf)
vTf = np.array(vTf)

if impulsive == True:
    #Final state, after DV
    u_max = Tmax/m*dt
    uf = min(norm(vTf - v), u_max)*(vTf - v)/norm(vTf - v)
    rf = r
    vf = v + uf
    mf = m*np.exp(-norm(uf)/ueq)
else:
    #Final state
    rf = r
    vf = v
    mf = m

#Print final state
f_out.write("t_step = " + str(np.round(t/dt)) + "\n")
f_out.write("norm(r) = " + str(norm(r)) + "\n")
f_out.write("norm(v) = " + str(norm(v)) + "\n")
if impulsive == True:
    f_out.write("norm(u/u_max) = " + str(norm(uf)/u_max) + "\n")
f_out.write("m = " + str(m) + "\n")
f_out.write("cum_reward = " + str(cumulative_reward) + "\n\n")

if impulsive == True:
    #Add final u vector
    ax.quiver(rf[0], rf[1], rf[2], uf[0], uf[1], uf[2]/3, \
        length=20)
    f_out_traj.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (rf[0], rf[1], rf[2], vf[0], vf[1], vf[2]))
    f_out_u.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (tf, rf[0], rf[1], rf[2], uf[0], uf[1], uf[2], norm(uf), u_max, mf))

f_out.write("Final position: r = " + str(rf*rconv) + " km\n")
f_out.write("Final velocity: v = " + str(vf*vconv) + " km/s\n")
f_out.write("Final target position: rT = " + str(rTf*rconv) + " km\n")
f_out.write("Final target velocity: vT = " + str(vTf*vconv) + " km/s\n")
f_out.write("Final relative position error: dr = " + str(norm(rf - rTf)/norm(rTf)) + "\n")
f_out.write("Final relative velocity error: dv = " + str(norm(vf - vTf)/norm(vTf)) + "\n")
f_out.write("Final mass: m = " + str(mf*mconv) + " kg\n")
f_out.write("Final time: t = " + str(t*tconv/pk.DAY2SEC) + " days\n\n")

f_out.close()
if impulsive == True:
    f_out_traj.close()
    f_out_u.close()

#Final target position
ax.plot([rTf[0]], [rTf[1]], [rTf[2]], '.g', markersize=12)

# Get rid of colored axes planes
# First remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

#Set plot properties
fontP = FontProperties()
fontP.set_size(15)
ax.legend(prop=fontP, loc='center right', ncol = 1)

#3D plot
ax.set_zlim([-0.5, 0.5])
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False) 
ax.zaxis.set_rotate_label(False) 
ax.set_xlabel('$x/\\bar{r}$', labelpad=15, rotation = 0)
ax.set_ylabel('$y/\\bar{r}$', labelpad=15, rotation = 0)
ax.set_zlabel('$z/\\bar{r}$', labelpad=15, rotation = 0)
ax.tick_params(axis="z",direction="out", pad=5)
fig1.savefig(plot_folder + "trajectory3D.pdf", dpi=300, bbox_inches='tight')

#2D projection on x,y plane
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 2])
set_axes_equal(ax)
ax.set_zlim([0, 0])
ax.view_init(90, 270)
ax.set_zticks([])
ax.zaxis.axes._draw_grid = False
ax.xaxis.axes._draw_grid = True
ax.yaxis.axes._draw_grid = True
ax.xaxis._axinfo['juggled'] = (2,0,1)
ax.yaxis._axinfo['juggled'] = (2,1,0)
ax.tick_params(axis="y",direction="out", labeltop=True, pad=15)
ax.tick_params(axis="x",direction="out", pad=3)
ax.set_xlabel('$x/\\bar{r}$', labelpad=15, rotation = 0)
ax.set_ylabel('$y/\\bar{r}$', labelpad=35, rotation = 0)
ax.set_zlabel('')

fig1.savefig(plot_folder + "trajectory2D.pdf", dpi=300, bbox_inches='tight') #dpi=300

#Control figure
plt.xlabel('$t/t_f$')
if impulsive == True:
    plt.ylabel('$\\Delta V/\\bar{v}$')
else:
    plt.ylabel('Thrust')
plt.grid()
fig2.savefig(plot_folder + "control.pdf", dpi=300, bbox_inches='tight')


print("Results printed, graphs plotted.")