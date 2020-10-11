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
policy = globals()[policy]

#Input Model and Output folders
in_folder = out_folder_root = "./sol_saved/sol_" + str(n_sol) + "/"
monitor_folder = in_folder + algorithm + "/"
logname = "final_model" #"best_model"
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
Isp = float(Isp)                     # s, Specific impulse
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
f_out = open(in_folder + "MCanalysis.txt", "w") # open file
f_out.write("%12s %12s %12s %12s\n" % ("# mass [kg]", "dr [%]", "dv [%]", "reward"))
f_out_traj_MC = open(in_folder + "Trajectory_MC.txt", "w") # open file
f_out_traj_MC.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
        % ("# x_rel", "y_rel", "z_rel", "vx_rel", "vy_rel", "vz_rel"))

#Create figures
matplotlib.rc('font', size=18)
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{textcomp} \usepackage{mathabx}')
fig1 = plt.figure(figsize=(10, 10))
ax = fig1.gca(projection='3d')

if (MTE == True) and (pr_MTE == 0):
    fig2 = plt.figure()
    ax2 = fig2.gca()
    fig3 = plt.figure()
    ax3 = fig3.gca()

#Departure orbit
pk.orbit_plots.plot_kepler(r0 = np.array(r0), v0 = np.array(v0), tof = tf, mu = 1., \
    color='k', axes=ax)

#Arrival orbit
rTf = np.array(rTf)
vTf = np.array(vTf)
pk.orbit_plots.plot_kepler(r0 = np.array(rT0), v0 = np.array(vT0), tof = tf, mu = 1., \
    color='r', axes=ax)

#Nominal trajectory

#Reset environment
obs = env0.reset()
cumulative_reward = 0.
r_nom = []
v_nom = []
m_nom = []
u_nom = []
u_nom_norm = []
for i in range(NSTEPS):
    
    #Get current action
    action, _states = model.predict(obs, deterministic=True)

    #Get new observation
    obs, reward, done, info = env0.step(action)

    #Nominal spacecraft state, time and control
    r_nom.append(np.array([info["rx"], info["ry"], info["rz"]]))
    v_nom.append(np.array([info["vx"], info["vy"], info["vz"]]))
    m_nom.append(info["m"])
    u_nom.append(np.array([info["ux"], info["uy"], info["uz"]]))
    u_nom_norm.append(norm(u_nom[-1]))

    #Update cumulative reward
    cumulative_reward += reward

#Final state
t = env0.t
r = env0.rk
v = env0.vkm
m = env0.mk

if impulsive == True:
    #Final state, after DV
    u_max = env0.max_action()
    uf = min(norm(vTf - v), u_max)*(vTf - v)/norm(vTf - v)
    rf = r
    vf = v + uf
    mf = env0.Tsiolkowsky(uf)

    uf_nom = uf

else:
    #Final state
    rf = r
    vf = v
    mf = m

dr = norm(rf - rTf)/norm(rTf)
dv = norm(vf - vTf)/norm(vTf)

rf_nom = rf
vf_nom = vf
mf_nom = mf

f_out.write("%12.7f %12.7f %12.7f %12.7f\n" % (mf*mconv, dr, dv, cumulative_reward))

#MonteCarlo analysis
print("MC simulation")
Nsim = 500
if stochastic == True:
    env0.stochastic = True
if random_obs == True:
    env0.random_obs = True

mf_vec = []
dr_vec = []
dv_vec = []
rew_vec = []
if (MTE == True) and (pr_MTE == 0):
    tk_mte = []
    mf_mte = []
    dr_mte = []
    dv_mte = []

for sim in range(Nsim):
    
    print("Simulation number %d / %d" % (sim+1, Nsim))

    #New seed
    env0.seed(1000 + sim)
    
    #Reset environment
    obs = env0.reset()
    cumulative_reward = 0.

    if MTE == True and pr_MTE == 0:
        new_MTE = False

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

        #Plot trajectory segment
        if impulsive == True:
            if sim == 0 and i==0:
                plot_kepler_new(r0 = r, v0 = (v + u), r0_nom = r_nom[i], v0_nom = (v_nom[i] + u_nom[i]), \
                    tof = dt, mu = 1., units = 5., N=10, color='#A9A9A9', \
                    label="perturbed", axes=ax, file_out=f_out_traj_MC)
            else:
                plot_kepler_new(r0 = r, v0 = (v + u), r0_nom = r_nom[i], v0_nom = (v_nom[i] + u_nom[i]), \
                    tof = dt, mu = 1., units = 5., N=10, color='#A9A9A9', \
                    axes=ax, file_out=f_out_traj_MC)
        else:
            if sim == 0 and i==0:
                plot_taylor_new(r0 = r, v0 = v, m0 = m, r0_nom = r_nom[i], v0_nom = v_nom[i], \
                    m0_nom = m_nom[i], thrust = u, thrust_nom = u_nom[i], tof = dt, mu = 1., \
                    veff = ueq, units = 5., color='#A9A9A9', label="perturbed", axes=ax)
            else:
                plot_taylor_new(r0 = r, v0 = v, m0 = m, r0_nom = r_nom[i], v0_nom = v_nom[i], \
                    m0_nom = m_nom[i], thrust = u, thrust_nom = u_nom[i], tof = dt, mu = 1., \
                    veff = ueq, units = 5., color='#A9A9A9', axes=ax)
        
        if (MTE == True) and (pr_MTE == 0):
            tk = t/dt
            if (norm(u) == 0.) and (tk not in tk_mte):
                tk_mte.append(tk)
                new_MTE = True

        #Update cumulative reward
        cumulative_reward += reward

    #Final state
    t = env0.t
    r = env0.rk
    v = env0.vkm
    m = env0.mk

    if impulsive == True:
        #Final state, after DV
        u_max = env0.max_action()
        uf = min(norm(vTf - v), u_max)*(vTf - v)/norm(vTf - v)
        rf = r
        vf = v + uf
        mf = env0.Tsiolkowsky(uf)
    else:
        #Final state
        rf = r
        vf = v
        mf = m

    dr = norm(rf - rTf)/norm(rTf)
    dv = norm(vf - vTf)/norm(vTf)
    
    mf_vec.append(mf*mconv)
    dr_vec.append(dr)
    dv_vec.append(dv)
    rew_vec.append(cumulative_reward)

    if (MTE == True) and (pr_MTE == 0) and (new_MTE == True and pr_MTE == 0):
        mf_mte.append(mf)
        dr_mte.append(dr*1000)
        dv_mte.append(dv*1000)
    
    f_out.write("%12.7f %12.7f %12.7f %12.7f\n" % (mf*mconv, dr, dv, cumulative_reward))
    f_out_traj_MC.write("\n\n");

f_out.close()
f_out_traj_MC.close()

if (MTE == True) and (pr_MTE == 0):
    sort_index = np.argsort(tk_mte)
    tk_mte.sort()
    mf_mte_sort = [mf_mte[i] for i in sort_index]
    dr_mte_sort = [dr_mte[i] for i in sort_index]
    dv_mte_sort = [dv_mte[i] for i in sort_index]
    mf_mte = mf_mte_sort
    dr_mte = dr_mte_sort
    dv_mte = dv_mte_sort

SR = 0
for j in range(Nsim):
    if (dr_vec[j] < eps_schedule[-1]) and (dv_vec[j] < eps_schedule[-1]):
        SR += 1

#Print statistics
f_out_stats = open(in_folder + "MCstats.txt", "w") # open file
f_out_stats.write("%15s\t%12s\t%12s\t%12s\t%12s\n" % ("#", "mass [kg]", "dr [%]", "dv [%]", "reward"))
f_out_stats.write("%15s\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" % ("mean", \
    np.mean(mf_vec), np.mean(dr_vec), np.mean(dv_vec), np.mean(rew_vec)))
f_out_stats.write("%15s\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" % ("sigma", \
    np.std(mf_vec), np.std(dr_vec), np.std(dv_vec), np.std(rew_vec)))
f_out_stats.write("%15s\t%12s\t%12.7f\t%12.7f\t%12s\n" % ("N.feasibles[%]", \
    "- -", float(sum(map(lambda x : x<=eps_schedule[-1], dr_vec))/Nsim), \
        float(sum(map(lambda x : x<=eps_schedule[-1], dv_vec))/Nsim), float(SR/Nsim)))
f_out_stats.close()

#Plot nominal trajectory
for i in range(NSTEPS):
    if impulsive == True:
        if i==0:
            pk.orbit_plots.plot_kepler(r0 = r_nom[i], v0 = (v_nom[i] + u_nom[i]), tof = dt, \
                color='b', label="nominal", mu = 1., axes=ax)
        else:
            pk.orbit_plots.plot_kepler(r0 = r_nom[i], v0 = (v_nom[i] + u_nom[i]), tof = dt, \
                color='b', mu = 1., axes=ax)
        ax.quiver(r_nom[i][0], r_nom[i][1], r_nom[i][2], u_nom[i][0],  u_nom[i][1], u_nom[i][2]/3, length=20)
    else:
        if i==0:
            pk.orbit_plots.plot_taylor(r0 = r_nom[i], v0 = v_nom[i], m0 = m_nom[i], \
                thrust = u_nom[i], tof = dt, mu = 1., veff = ueq, \
                color='b', label="nominal", axes=ax)
        else:
            pk.orbit_plots.plot_taylor(r0 = r_nom[i], v0 = v_nom[i], m0 = m_nom[i], \
                thrust = u_nom[i], tof = dt, mu = 1., veff = ueq, \
                color='b', axes=ax)
        ax.quiver(r_nom[i][0], r_nom[i][1], r_nom[i][2], u_nom[i][0],  u_nom[i][1], u_nom[i][2]/3, length=5)

#Final target position
ax.plot([rTf[0]], [rTf[1]], [rTf[2]], '.g', markersize=12, label='Final target position')

if impulsive == True:
    #Add final u vector
    ax.quiver(rf_nom[0], rf_nom[1], rf_nom[2], uf_nom[0], uf_nom[1], uf_nom[2]/3, \
        length=20, label='$\\Delta V$')

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
fontP.set_size(18)
ax.legend(prop=fontP)
ax.get_legend().remove() 
set_axes_equal(ax)

#3D plot
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False) 
ax.zaxis.set_rotate_label(False) 
ax.set_xlabel('$x/\\bar{r}$', labelpad=15, rotation = 0)
ax.set_ylabel('$y/\\bar{r}$', labelpad=15, rotation = 0)
ax.set_zlabel('$z/\\bar{r}$', labelpad=15, rotation = 0)
ax.tick_params(axis="z",direction="out", pad=5)
fig1.savefig(plot_folder + "MC3D.pdf", dpi=300, bbox_inches='tight')

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
fig1.savefig(plot_folder + "MC2D.pdf", dpi=300, bbox_inches='tight')


if (MTE == True) and (pr_MTE == 0):

    ax2.set_xlabel('MTE location $\\hat{k}$')
    ax2.set_ylabel('Constraint violation [\\textperthousand]')

    ax2.plot(tk_mte, np.zeros(len(tk_mte)), '-k', linewidth=1)

    if dr_mte[0] > dv_mte[0]:
        markerline, stemlines, baseline = ax2.stem([tk_mte[0]], [dr_mte[0]], '-r', label='$\\Delta r_f/r_\\Mars$', use_line_collection=True)
        plt.setp(markerline, 'marker', 'o', 'markerfacecolor', 'red', 'markeredgecolor', 'red')
        markerline, stemlines, baseline = ax2.stem([tk_mte[0]], [dv_mte[0]], '-b', label='$\\Delta v_f/v_\\Mars$', use_line_collection=True)
        plt.setp(markerline, 'marker', 'o', 'markerfacecolor', 'blue', 'markeredgecolor', 'blue')
    else:
        markerline, stemlines, baseline = ax2.stem([tk_mte[0]], [dv_mte[0]], '-b', label='$\\Delta v_f/v_\\Mars$', use_line_collection=True)
        plt.setp(markerline, 'marker', 'o', 'markerfacecolor', 'blue', 'markeredgecolor', 'blue')
        markerline, stemlines, baseline = ax2.stem([tk_mte[0]], [dr_mte[0]], '-r', label='$\\Delta r_f/r_\\Mars$', use_line_collection=True)
        plt.setp(markerline, 'marker', 'o', 'markerfacecolor', 'red', 'markeredgecolor', 'red')

    for i in range(1, len(tk_mte)):
        if dr_mte[i] > dv_mte[i]:
            markerline, stemlines, baseline = ax2.stem([tk_mte[i]], [dr_mte[i]], '-r', use_line_collection=True)
            plt.setp(markerline, 'marker', 'o', 'markerfacecolor', 'red', 'markeredgecolor', 'red')
            markerline, stemlines, baseline = ax2.stem([tk_mte[i]], [dv_mte[i]], '-b', use_line_collection=True)
            plt.setp(markerline, 'marker', 'o', 'markerfacecolor', 'blue', 'markeredgecolor', 'blue')
        else:
            markerline, stemlines, baseline = ax2.stem([tk_mte[i]], [dv_mte[i]], '-b', use_line_collection=True)
            plt.setp(markerline, 'marker', 'o', 'markerfacecolor', 'blue', 'markeredgecolor', 'blue')
            markerline, stemlines, baseline = ax2.stem([tk_mte[i]], [dr_mte[i]], '-r', use_line_collection=True)
            plt.setp(markerline, 'marker', 'o', 'markerfacecolor', 'red', 'markeredgecolor', 'red')

    ax2.plot(tk_mte, eps_schedule[-1]*1000*np.ones(len(tk_mte)), '--k', linewidth=2)
    ax2.legend()

    fig2.savefig(plot_folder + "MTE_error.pdf", dpi=300, bbox_inches='tight')

    ax3.set_xlabel('MTE location $\\hat{k}$')
    ax3.set_ylabel('$m_f/\\bar{m}$')

    markerline, stemlines, baseline = ax3.stem(tk_mte, mf_mte, '-g', use_line_collection=True, bottom = 0.57, basefmt='k-')
    plt.setp(markerline, 'marker', 'o', 'markerfacecolor', 'g', 'markeredgecolor', 'g')

    fig3.savefig(plot_folder + "MTE_mass.pdf", dpi=300, bbox_inches='tight')


print("Results printed, graphs plotted.")