import os
import sys
import warnings
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import gym
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, LstmPolicy
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, A2C, DDPG, SAC, TD3, TRPO
from stable_baselines.common.callbacks import EvalCallback
from custom_modules.custom_policies import CustomPolicy_3x64, CustomPolicy_4x128, CustomLSTMPolicy
from custom_modules.learning_schedules import linear_schedule
from custom_modules.env_fun import make_env
from custom_modules.plot_results import plot_results
from custom_modules.set_axes_equal_3d import set_axes_equal
import argparse
import time
import gym_lowthrust
import pykep as pk
from pykep.core import epoch_from_string, ic2par, propagate_lagrangian
from pykep.planet import jpl_lp

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

if __name__ == '__main__':

    #Input data
    postprocess = True
    MonteCarlo = True
    tensorboard = False
    eval_environment = False

    #Input settings file
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default="settings_unp.txt", \
        help='Input settings file')
    parser.add_argument('--input_model', type=str, default="final_model", \
        help='Input model to load')
    args = parser.parse_args()
    settings_file = "./settings_files/" + args.settings
    input_model = "./settings_files/" + args.input_model

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

    #Settings
    load_model = bool(int(load_model)) #load external model

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
    seed = 0 #int(time.time()) #pseudorandom number generator seed

    #Model parameters
    if load_model == True:
        policy = globals()[policy]
    num_cpu = int(num_cpu) #number of environments
    learning_rate_in = float(learning_rate_in) #initial learning rate
    clip_range_in = float(clip_range_in) #initial clip range
    if learning_rate == "lin":
        learning_rate = linear_schedule(learning_rate_in)
    elif learning_rate == "const":
        learning_rate = learning_rate_in
    if clip_range == "lin":
        clip_range = linear_schedule(clip_range_in)
    elif clip_range == "const":
        clip_range = clip_range_in
    ent_coef = float(ent_coef)
    gamma = float(gamma)
    lam = float(lam)
    noptepochs = int(noptepochs)
    nminibatches = int(nminibatches)
    niter = int(float(niter)) #number of iterations
    n_steps = int(NSTEPS*nminibatches) # batch size
    niter_per_cpu = niter / num_cpu # Steps per CPU

    #Output folders and log files
    n_sol = 1
    out_folder_root = "./sol_saved/sol_"
    out_folder = out_folder_root + str(n_sol) + "/"
    while os.path.exists(out_folder):
        n_sol += 1
        out_folder = out_folder_root + str(n_sol) + "/"
    os.makedirs(out_folder, exist_ok=True)
    trained_model_name = "final_model"
    if tensorboard == True:
        tensorboard_log = out_folder
    else:
        tensorboard_log = None
    trained_model_log = out_folder + trained_model_name
    monitor_folder = out_folder + algorithm + "/"
    os.system("cp " + settings_file + " " + out_folder)
    os.system("mv " + out_folder + args.settings + " " + out_folder + "settings.txt")

    #Physical constants
    amu = 132712440018.             # km^3/s^2, Gravitational constant of the central body
    rconv = 149600000.              # position, km
    vconv = np.sqrt(amu/rconv)      # velocity, km/s
    tconv = rconv/vconv             # time, s
    mconv = 1000.                   # mass, kg
    aconv = vconv/tconv             # acceleration, km/s^2
    fconv = mconv*aconv             # force, kN

    #Spacecraft data (nondimensional)
    Tmax = 0.5e-3/fconv             # Maximum thrust
    Isp = 2000.                     # s, Specific impulse
    ueq = (pk.G0/1000.*Isp)/vconv   # Equivalent ejection velocity

    #Read Mission file
    t_nom = []  #nominal trajectory: time
    rx_nom = [] #nominal trajectory: x
    ry_nom = [] #nominal trajectory: y
    rz_nom = [] #nominal trajectory: z
    vx_nom = [] #nominal trajectory: vx
    vy_nom = [] #nominal trajectory: vy
    vz_nom = [] #nominal trajectory: vz
    m_nom = []  #nominal trajectory: m
    mission_folder = "missions/"
    mission_file = mission_folder + "Earth_Mars.dat" #File with reference trajectory
    with open(mission_file, "r") as f: # with open context
        f.readline()
        file_all = f.readlines()
        for line in file_all: #read line
            line = line.split()
            state_adim = np.array(line).astype(np.float64) #non-dimensional data
            
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

    #Reference initial state
    m0 = m_nom[0] #kg, initial spacecraft mass
    r0 = [rx_nom[0], ry_nom[0], rz_nom[0]] #km, initial spacecraft position
    v0 = [vx_nom[0], vy_nom[0], vz_nom[0]] #km/s, initial spacecraft velocity

    #Target initial state
    rTf = [rx_nom[-1], ry_nom[-1], rz_nom[-1]]      #km, final target position
    vTf = [vx_nom[-1], vy_nom[-1], vz_nom[-1]]      #km/s, final target velocity
    
    # Create the environment

    #Create the vectorized environment
    for rank in range(num_cpu):
        os.makedirs(monitor_folder + "env_" + str(rank) + "/", exist_ok=True)

    if num_cpu <= 1:
         env = DummyVecEnv([make_env(env_name, rank, seed, monitor_folder + "env_" + str(rank) + "/", \
            impulsive, action_coord, obs_type, random_obs, stochastic, mission_type, NSTEPS, \
            niter_per_cpu, eps_schedule, lambda_con, \
            Tmax, ueq, tf, 1., m0, \
            r0, v0, \
            rTf, vTf, \
            sigma_r, sigma_v, \
            sigma_u_rot, sigma_u_norm, \
            MTE, pr_MTE) for rank in range(num_cpu)])
    else:
        env = SubprocVecEnv([make_env(env_name, rank, seed, monitor_folder + "env_" + str(rank) + "/", \
            impulsive, action_coord, obs_type, random_obs, stochastic, mission_type, NSTEPS, \
            niter_per_cpu, eps_schedule, lambda_con, \
            Tmax, ueq, tf, 1., m0, \
            r0, v0, \
            rTf, vTf, \
            sigma_r, sigma_v, \
            sigma_u_rot, sigma_u_norm, \
            MTE, pr_MTE) for rank in range(num_cpu)], start_method='spawn')
    
    #Create the evaluation environment
    if eval_environment == True:
        eps_schedule_eval = [eps_schedule[-1]]
        eval_env = gym.make(env_name, impulsive = impulsive, action_coord = action_coord, \
            obs_type = obs_type, random_obs = False, stochastic = False, \
            mission_type = mission_type,
            NSTEPS = NSTEPS, NITER = niter_per_cpu / nminibatches, \
            eps_schedule = eps_schedule_eval, lambda_con = lambda_con, \
            Tmax = Tmax, ueq = ueq, tf = tf, amu = 1., m0 = m0, \
            r0 = r0, v0 = v0, \
            rTf = rTf, vTf = vTf, \
            sigma_r = sigma_r, sigma_v = sigma_v, \
            sigma_u_rot = sigma_u_rot, sigma_u_norm = sigma_u_norm, \
            MTE = MTE, pr_MTE = pr_MTE)
        eval_callback = EvalCallback(eval_env, n_eval_episodes = 1, \
                                best_model_save_path=out_folder, \
                                log_path=out_folder, eval_freq=n_steps, \
                                deterministic=True)

    # Create the model
    if algorithm == "PPO": 
        if load_model == False:
            model = PPO2(policy=policy, env=env, 
                        n_steps=n_steps, nminibatches=nminibatches,
                        gamma=gamma, ent_coef=ent_coef,
                        lam=lam, noptepochs=noptepochs,
                        learning_rate=learning_rate,
                        cliprange=clip_range,
                        tensorboard_log=tensorboard_log, verbose=1)
        else:
            model = PPO2.load(input_model, policy=policy, env=env, 
                        n_steps=n_steps, nminibatches=nminibatches,
                        gamma=gamma, ent_coef=ent_coef,
                        lam=lam, noptepochs=noptepochs,
                        learning_rate=learning_rate,
                        cliprange=clip_range,
                        verbose=1)
            model.tensorboard_log = tensorboard_log
    elif algorithm == "A2C":
        model = A2C(policy=policy, env=env, n_steps=n_steps, verbose=1,
                    tensorboard_log=tensorboard_log)
    elif algorithm == "DDPG":
        model = DDPG(policy='MlpPolicy', env=env, verbose=1,
                    tensorboard_log=tensorboard_log)
    elif algorithm == "SAC":
        model = SAC(policy='MlpPolicy', env=env, verbose=1,
                    learning_rate=learning_rate,
                    tensorboard_log=tensorboard_log)
    elif algorithm == "TD3":
        model = TD3(policy='MlpPolicy', env=env, verbose=1,
                    learning_rate=learning_rate,
                    tensorboard_log=tensorboard_log)
    elif algorithm == "TRPO":
        model = TRPO(policy=policy, env=env, verbose=1,
                    tensorboard_log=tensorboard_log)

    # Learning
    if eval_environment == True:
        model.learn(total_timesteps=niter, callback=eval_callback)
    else:
        model.learn(total_timesteps=niter)

    # Save solution
    model.save(trained_model_log)
    print("End Training.")

    # Post-process
    if postprocess == True:
        print("Post-processing")
        os.system('python main_lowthrust_load.py --settings ' + out_folder + "settings.txt " + " --n_sol " + str(n_sol))
        if MonteCarlo == True:
            print("Monte Carlo")
            os.system('python main_lowthrust_MC.py --settings ' + out_folder + "settings.txt " + " --n_sol " + str(n_sol))

    os.system("killall -9 python")