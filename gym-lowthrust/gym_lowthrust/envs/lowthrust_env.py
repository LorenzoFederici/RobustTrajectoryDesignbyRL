# Filter tensorflow version warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import numpy as np
from numpy import sqrt, log, exp, cos, sin, arccos, cross, dot, array
from numpy.linalg import norm
from numpy.random import uniform as np_uniform
from numpy.random import normal as np_normal
from numpy.random import randint as np_randint
import pykep as pk
from pykep.core import propagate_lagrangian, propagate_taylor, ic2par, ic2eq

""" RL ENVIRONMENT CLASS """
class LowThrustEnv(gym.Env):
    """
    Reinforcement Learning environment,
    for a mass-optimal, time-fixed, low-thrust transfer 
    with fixed initial and final conditions.
    The mission can be an interception (final target position), 
    or a rendezvous (final target position and velocity).
    The Sims-Flanagan method is used to model the low-thrust trajectory, that
    is divided into a number of segments.
    Two different thrust models are implemented:
    - impulsive thrust, with the magnitude of any impulse limited by the amount 
        of DV that could be accumulated over the duration of the 
        corresponding segment;
    - piecewise constant thrust.
    The environment can be deterministic or stochastic (i.e., with state and
    control uncertainties).

    
    Class inputs:
        - (bool) impulsive: True = impulsive thrust;
                False = piecewise constant thrust
        - (bool) action_coord: False = action in cartesian coordinates in ECI; 
                True = action in cartesian coordinates in RTN
        - (int) obs_type: 0 = observations are S/C position and 
                velocity in ECI;
                1 = observations are S/C COEs (classical orbital elements)
                2 = observations are S/C MEEs (modified equinoctial elements)
                3 = observations are S/C position and 
                velocity in ECI + current action;
        - (bool) random_obs: False = deterministic observations, True = random observations
        - (bool) stochastic: True = stochastic environment,
                        False = deterministic environment
        - (bool) mission_type: False = interception, True = rendezvous
        - (int) NSTEPS: number of trajectory segments
        - (inr) NITER: number of training iterations
        - (list) eps_schedule: list of tolerance values for constraint satisfaction
        - (float) lambda_con: weight of constraint violation in reward
        - (float) Tmax: maximum engine thrust, kN
        - (float) ueq: equivalent engine ejection velocity, km/s
        - (float) amu: gravitational constant of the central body, km^3/s^2
        - (float) tf: total mission time, s
        - (float) m0: initial mass, kg
        - (list) r0: initial position, km
        - (list) v0: initial velocity, km/s
        - (list) rTf: final target position, km
        - (list) vTf: final target velocity, km/s
        - (float) sigma_r: standard deviation on position, km
        - (float) sigma_v: standard deviation on velocity, km/s
        - (float) sigma_u_rot : standard deviation on control vector rotation, rad
        - (float) sigma_u_norm : standard deviation on control vector modulus (percentage)
        - (bool) MTE: at least one MTE occurs?
        - (float) pr_MTE: probability of having a new MTE after the last one

    RL ENVIRONMENT (MDP)
    Observations: 
        Type: Box(8)
        if (obs_type == 0):
        Num	Observation                Min     Max
        0	rx                        -max_r   max_r
        1	ry                        -max_r   max_r
        2	rz                        -max_r   max_r
        3	vx                        -max_v   max_v
        4	vy                        -max_v   max_v
        5	vz                        -max_v   max_v
        6	m                           0       1
        7   t                           0       1
        else if (obs_type == 1):
        Num	Observation                Min     Max
        0	a                           0      max_a
        1	e                           0      1
        2	ainc/pi                     0      1
        3	gom/2pi                     0      1
        4	pom/2pi                     0      1
        5	AE/2pi                      0      1
        6	m                           0      1
        7   t                           0      1 
        else if (obs_type == 2):
        Num	Observation                Min     Max
        0	p                           0      max_p
        1	f                          -1      1
        2	g                          -1      1
        3	h                          -1      1
        4	k                          -1      1
        5	L/2pi                       0      1
        6	m                           0      1
        7   t                           0      1 
        else if (obs_type == 3):
        Num	Observation                Min     Max
        0	rx                        -max_r   max_r
        1	ry                        -max_r   max_r
        2	rz                        -max_r   max_r
        3	vx                        -max_v   max_v
        4	vy                        -max_v   max_v
        5	vz                        -max_v   max_v
        6	m                           0       1
        7   t                           0       1
        8   ax                         -1       1
        9   ay                         -1       1
       10   az                         -1       1
        
    Actions:
        Type: Box(3)
        Num	Action
        0	ux                          -1      1
        1	uy                          -1      1
        2	uz                          -1      1

    Reward:
        [at any time step]                              -mp
        [at final time]                                 -norm(r-rTf)
        [at final time, if mission_type is True]        -norm(v-vTf)

    Starting State:
        Start at state: (r0, v0, m0)
    
    Episode Termination:
        - At time tf
        - Position and/or velocity reach the environment boundaries

    """

    metadata = {'render.modes': ['human']}

    """ Class constructor """
    def __init__(self, impulsive, action_coord, obs_type, \
        random_obs, stochastic, mission_type, 
        NSTEPS, NITER, \
        eps_schedule, lambda_con, \
        Tmax, ueq, tf, amu, m0, \
        r0, v0, \
        rTf, vTf, \
        sigma_r, sigma_v, \
        sigma_u_rot, sigma_u_norm, \
        MTE, pr_MTE):

        super(LowThrustEnv, self).__init__()

        """ Class attributes """
        self.impulsive = bool(impulsive)
        self.action_coord = bool(action_coord)
        self.obs_type = int(obs_type)
        self.random_obs = bool(random_obs)
        self.stochastic = bool(stochastic)
        self.mission_type = bool(mission_type)
        self.NSTEPS = float(NSTEPS)
        self.NITER = float(NITER)
        self.eps_schedule = array(eps_schedule)
        self.lambda_con = float(lambda_con)
        self.Tmax = float(Tmax)
        self.ueq = float(ueq)
        self.tf = float(tf)
        self.amu = float(amu)
        self.m0 = float(m0)
        self.r0 = array(r0)
        self.v0 = array(v0)
        self.rTf = array(rTf)
        self.vTf = array(vTf)
        self.sigma_r = float(sigma_r)               # 6.6845e-15, 6.6845e-9
        self.sigma_v = float(sigma_v)               # 1.6787e-4, 1.6787e-3
        self.sigma_u_rot = float(sigma_u_rot)       # 1 deg
        self.sigma_u_norm = float(sigma_u_norm)     # 0.1
        self.MTE = bool(MTE)              
        self.pr_MTE = float(pr_MTE)                 # 0.025
        
        """ Time variables """
        self.dt = self.tf / self.NSTEPS     # time step, s
        self.training_steps = 0.            # number of training steps
        self.mdot_max = sqrt(3.)*self.Tmax/self.ueq  # maximum mass flow rate

        """ Environment boundaries """
        coe0 = ic2par(r = self.r0, v = self.v0, mu = self.amu)    # initial classical orbital element of the S/C
        coeT = ic2par(r = self.rTf, v = self.vTf, mu = self.amu)  # classical orbital element of the target
        rpT = coeT[0]*(1 - coeT[1])                               # periapsis radius of the target, km
        raT = coeT[0]*(1 + coeT[1])                               # apoapsis radius of the target, km
        vpT = sqrt(self.amu*(2./rpT - 1/coeT[0]))                 # periapsis velocity of the target, km/s
        self.min_r = 0.1*min(norm(self.r0), rpT)                  # minimum radius, km
        self.max_r = 4.*max(norm(self.r0), raT)                   # maximum radius, km
        self.max_v = 4.*max(norm(self.v0), vpT)                   # maximum velocity, km/s
        self.max_a = 4.*max(coe0[0], coeT[0])                     # maximum semi-major axis, km
        self.max_p = 4.*max(coe0[0]*(1-coe0[1]**2), \
            coeT[0]*(1-coeT[1]**2))                               # maximum semi-major axis, km
        self.max_mte = 3                                          # maximum number of MTEs

        """ OBSERVATION SPACE """
        if self.obs_type == 0:
            # Lower bounds
            x_lb = np.array([-self.max_r, -self.max_r, -self.max_r, \
                -self.max_v, -self.max_v, -self.max_v, \
                0., 0.])
            # Upper bounds
            x_ub = np.array([+self.max_r, +self.max_r, +self.max_r, \
                +self.max_v, +self.max_v, +self.max_v, \
                1., 1.])
        elif self.obs_type == 1:
            # Lower bounds
            x_lb = np.array([0., 0., 0., \
                0., 0., 0., \
                0., 0.])
            # Upper bounds
            x_ub = np.array([self.max_a, 1., 1., \
                1., 1., 1., \
                1., 1.])
        elif self.obs_type == 2:
            # Lower bounds
            x_lb = np.array([0., -1., -1., \
                -1., -1., 0., \
                0., 0.])
            # Upper bounds
            x_ub = np.array([self.max_p, 1., 1., \
                1., 1., 1., \
                1., 1.])
        else:
            # Lower bounds
            x_lb = np.array([-self.max_r, -self.max_r, -self.max_r, \
                -self.max_v, -self.max_v, -self.max_v, \
                0., 0., -1., -1., -1.])
            # Upper bounds
            x_ub = np.array([+self.max_r, +self.max_r, +self.max_r, \
                +self.max_v, +self.max_v, +self.max_v, \
                1., 1., 1., 1., 1.])
        
        self.observation_space = spaces.Box(x_lb, x_ub, dtype=np.float64)

        """ ACTION ASPACE """
        # Lower bounds
        a_lb = np.array([-1., -1., -1.])
        # Upper bounds
        a_ub = np.array([1., 1., 1.])

        self.action_space = spaces.Box(a_lb, a_ub, dtype=np.float64)
        
        """ Environment initialization """
        self.viewer = None
        self.state = None
    
    """ Set seed """
    def seed(self, seed=None):
        """
        :return seed: current seed of pseudorandom
            numbers generator
        """
        set_global_seeds(seed)
        
        return [seed]

    """ Get Reward """
    def getReward(self, done, action):
        """
        :param done: episode is terminated?
        :param action: current action
        :return reward: current reward
        """
        
        # Frequent reward: current propellant consumption
        reward = - self.mpk

        # Penalty: current action greater than maximum admissible
        reward -= 100.*max(0., norm(action) - 1.)

        #Delayed reward: final constraint violation
        if done: 
            
            # Constraint violation on final position
            r_viol = norm(self.rk - self.rTf)/norm(self.rTf)
            
            # Constraint violation on final velocity
            v_viol = 0.

            # Rendezvous mission
            if self.mission_type == True:
                if self.impulsive == True:
                    #Final burn and propellant consumed
                    DVf = self.vTf - self.vkm
                    mpf = self.mk - self.Tsiolkowsky(DVf)
                    
                    #Propellant consumption
                    reward -= mpf

                    #Constraint violation on velocity
                    v_viol = max(0., (norm(DVf) - self.max_action())/norm(self.vTf))
                else:
                    #Constraint violation on velocity
                    v_viol = norm(self.vkm - self.vTf)/norm(self.vTf)
            
            # Total constraint violation
            
            #Tolerance
            eps = self.eps_constraint()

            #Violation
            c_viol = max(0., max(r_viol, v_viol) - eps)
            reward -= self.lambda_con*c_viol
                
        return reward
    
    """ Episode termination """
    def isDone(self):
        """
        :return done: terminate the episode

        """
        done = 0
       
        if self.tk == self.NSTEPS:
            done = 1
            self.steps_beyond_done = 0
        elif self.tk > self.NSTEPS:
            self.steps_beyond_done += 1
            print("beyond_done = ", self.steps_beyond_done)

        return bool(done)

    """ Safe Episode termination """
    def safeStop(self, r, v, m):
        """
        :param r: current position, km
        :param v: current velocity, km/s
        :param m: current mass, kg
        :return bool: terminate the episode?

        """

        if ((norm(r) < self.min_r) or (norm(r) > self.max_r)):
            return True
        elif (norm(v) > self.max_v):
            return True
        elif (self.impulsive == False) and ((m - self.mdot_max*self.dt) <= 0):
            return True
        else:
            return False
    
    """ Get epsilon value """
    def eps_constraint(self):
        """
        :return epsilon: epsilon value at current training step,
            given eps_schedule, decreasing with a piecewise constant scedule

        """
        
        n_values = len(self.eps_schedule)
        n_steps_per_value = self.NITER / n_values

        for i in range(n_values):
            if (self.training_steps <= (i+1)*n_steps_per_value):
                return self.eps_schedule[i]

        return self.eps_schedule[n_values-1]
    
    """ Maximum action """
    def max_action(self):
        """
        :return action_max: maximum value of the action

        """
        
        if self.impulsive == True:
            action_max = self.Tmax/self.mk*self.dt
        else:
            action_max = self.Tmax

        return action_max
    
    """ Tsiolkowsky equation """
    def Tsiolkowsky(self, DV):
        """
        :param DV: current DV, km/s
        :return mk1: mass at the end of the DV, evaluated through 
            Tsiolkowsky equation, kg

        """
        
        mk1 = self.mk*exp(-norm(DV)/self.ueq)

        return mk1
    
    """ RTN to ECI """
    def RTNtoECI(self, vec_rtn):
        """
        :return vec_eci: vec in eci frame

        """
        
        vers_r = self.rk/norm(self.rk)
        vers_n = cross(self.rk,self.vkm)/norm(cross(self.rk,self.vkm))
        vers_t = cross(vers_n, vers_r)  

        vec_eci = vec_rtn[0]*vers_r + \
            vec_rtn[1]*vers_t + \
            vec_rtn[2]*vers_n

        return vec_eci
    
    """ Get observation errors at step tk """
    def obsErrors(self):
        """
        :return drk, dvk: errors on position (km), and
            velocity (km/s) at step tk

        """

        #Position error
        drk = np_normal(0., self.sigma_r, 3)

        #Velocity error
        dvk = np_normal(0., self.sigma_v, 3)

        return drk, dvk
    
    """ Get state errors at step tk """
    def stateErrors(self):
        """
        :return drk, dvk: errors on position (km), and
            velocity (km/s) at step tk

        """

        #Position error
        drk = np_normal(0., self.sigma_r, 3)

        #Velocity error
        dvk = np_normal(0., self.sigma_v, 3)

        return drk, dvk
    
    """ Perturbate the action at step tk """
    def perturbAction(self, action):
        """
        :param action: current action
        :return action_pert: perturbed action

        """
        #MTE
        if self.MTE == True:
            if (self.tk == self.tk_mte) and (self.n_mte < self.max_mte):
                self.n_mte += 1
                pr = np_uniform(0., 1.)
                if (pr < self.pr_MTE) and (self.tk < self.NSTEPS - 2):
                    self.tk_mte += 1
                return array([0., 0., 0.])

        #Modulus error
        du_norm = np_normal(1., self.sigma_u_norm)
        np.clip(du_norm, 1. - 10.*self.sigma_u_norm, 1. + 10.*self.sigma_u_norm)

        #Rotation error
        du_rot = np_normal(0., self.sigma_u_rot, 3)
        np.clip(du_rot, -3.*self.sigma_u_rot, 3.*self.sigma_u_rot)

        #Rotation matrix
        Arot = array([[1., -du_rot[2], du_rot[1]], 
                  [du_rot[2], 1., -du_rot[0]], 
                  [-du_rot[1], du_rot[0], 1.]])

        #Perturbed action
        action_pert = du_norm*(Arot.dot(action))

        return action_pert
    
    """ Propagation step """
    def propagation_step(self, action, tof):
        """
        :param action: current action
        :param tof: time of flight
        :return uk: control at the current time step
        :return rk1: position at the beginning of next time step
        :return vk1m: velocity at the beginning of next time step
        :return mk1: mass at the beginning of next time step

        """
        # Maximum value of the control
        uk_max = self.max_action()

        # uk
        if (self.action_coord == False):
            # control in ECI
            uk = uk_max*action
        else:
            # control in RTN
            uk_rtn = uk_max*action
            uk = self.RTNtoECI(uk_rtn)          
        
        if self.impulsive == True:
            # Velocity after DV
            vkp = self.vkm + uk

            # Position and velocity at the next time step
            rk1_list, vk1m_list = propagate_lagrangian(r0 = self.rk, v0 = vkp, tof = tof, mu = self.amu)

            # Spacecraft mass at the next time step
            mk1 = self.Tsiolkowsky(uk)
        else:
            # Position, velocity and mass at the next time step
            rk1_list, vk1m_list, mk1 = propagate_taylor(r0 = self.rk, v0 = self.vkm, \
                m0 = self.mk, thrust = uk, tof = tof, mu = self.amu, veff = self.ueq, \
                log10tol = -10, log10rtol = -10)


        rk1 = array(rk1_list)
        vk1m = array(vk1m_list)               

        return rk1, vk1m, mk1, uk

    """ Do forward step in the MDP """
    def step(self, action):
        """
        :param action: current action
        :return obs, reward, done, info

        """
        # Invalid action
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Perturbate the action
        if self.stochastic == True:
            action = self.perturbAction(action)

        # State at next time step and current control
        rk1, vk1m, mk1, uk = self.propagation_step(action, self.dt)

        # Perturbate the state
        if self.stochastic == True and self.tk < self.NSTEPS-1:
            drk, dvk = self.stateErrors()
            rk1 += drk
            vk1m += dvk

        # Info (state at the beginning of the segment)
        self.sol['rx'] = self.rk[0]
        self.sol['ry'] = self.rk[1]
        self.sol['rz'] = self.rk[2]
        self.sol['vx'] = self.vkm[0]
        self.sol['vy'] = self.vkm[1]
        self.sol['vz'] = self.vkm[2]
        self.sol['ux'] = uk[0]
        self.sol['uy'] = uk[1]
        self.sol['uz'] = uk[2]
        self.sol['m'] = self.mk
        self.sol['t'] = self.t
        info = self.sol
            
        # Update the spacecraft state
        self.rk = rk1
        self.vkm = vk1m
        self.mpk = self.mk - mk1
        self.mk = mk1
        self.tk += 1.
        self.tk_left -= 1.
        self.t = self.tk*self.dt
        self.t_left = self.tk_left*self.dt           

        #Errors on observations
        rk_obs = self.rk
        vkm_obs = self.vkm
        if self.random_obs == True:
            drk, dvk = self.obsErrors()
            rk_obs += drk
            vkm_obs += dvk

        # Observations
        if self.obs_type == 0:
            obs = np.array([rk_obs[0], rk_obs[1], rk_obs[2], \
                vkm_obs[0], vkm_obs[1], vkm_obs[2], \
                self.mk, self.t]).astype(np.float64)
        elif self.obs_type == 1:
            coesk = ic2par(r = rk_obs, v = vkm_obs, mu = self.amu)
            obs = np.array([coesk[0], coesk[1], coesk[2]/np.pi, \
                coesk[3]/(2*np.pi), coesk[4]/(2*np.pi), coesk[5]/(2*np.pi), \
                self.mk, self.t]).astype(np.float64)
        elif self.obs_type == 2:
            meesk = ic2eq(r = rk_obs, v = vkm_obs, mu = self.amu)
            obs = np.array([meesk[0], meesk[1], meesk[2], \
                meesk[3], meesk[4], meesk[5]/(2*np.pi), \
                self.mk, self.t]).astype(np.float64)
        else:
            obs = np.array([rk_obs[0], rk_obs[1], rk_obs[2], \
                vkm_obs[0], vkm_obs[1], vkm_obs[2], \
                self.mk, self.t, action[0], action[1], action[2]]).astype(np.float64)
        
        # Update training steps
        self.training_steps += 1.

        # Episode termination
        done = (self.isDone() or self.safeStop(self.rk, self.vkm, self.mk))

        # Reward
        reward = self.getReward(done, action)

        return obs, float(reward), done, info

    """ Initialize the episode """
    def reset(self):
        """
        :return obs: observation vector

        """

        # Environment variables
        self.rk = self.r0                   # position at the k-th time step, km
        self.vkm = self.v0                  # velocity at the k-th time step, before DV, km/s
        self.mk = self.m0                   # mass at the k-th time step, kg
        self.mpk = 0.                       # propellant mass expelled at the k-th time step
        self.tk = 0.                        # k-th time step
        self.tk_left = self.NSTEPS          # steps-to-go
        self.t = 0.                         # time from departure
        self.t_left = self.tf               # time-to-go
        
        if (self.stochastic == True):
            #Select the segment with MTE
            pr = np_uniform(0., 1.)
            if (self.MTE == True) and (pr < 1):
                self.tk_mte = np_randint(0, int(self.NSTEPS))
                self.n_mte = 0 #number of MTEs so far
            else:
                self.tk_mte = -1

        # Reset parameters
        self.sol = {'rx': [], 'ry': [], 'rz': [],
                    'vx': [], 'vy': [], 'vz': [],
                    'ux': [], 'uy': [], 'uz': [],
                    'm': [],
                    't': []}
        self.steps_beyond_done = 0
        self.done = False

        rk_obs = self.rk
        vkm_obs = self.vkm

        # Observations
        if self.obs_type == 0:
            obs = np.array([rk_obs[0], rk_obs[1], rk_obs[2], \
                vkm_obs[0], vkm_obs[1], vkm_obs[2], \
                self.mk, self.t]).astype(np.float64)
        elif self.obs_type == 1:
            coesk = ic2par(r = rk_obs, v = vkm_obs, mu = self.amu)
            obs = np.array([coesk[0], coesk[1], coesk[2]/np.pi, \
                coesk[3]/(2*np.pi), coesk[4]/(2*np.pi), coesk[5]/(2*np.pi), \
                self.mk, self.t]).astype(np.float64)
        elif self.obs_type == 2:
            meesk = ic2eq(r = rk_obs, v = vkm_obs, mu = self.amu)
            obs = np.array([meesk[0], meesk[1], meesk[2], \
                meesk[3], meesk[4], meesk[5]/(2*np.pi), \
                self.mk, self.t]).astype(np.float64)
        else:
            obs = np.array([rk_obs[0], rk_obs[1], rk_obs[2], \
                vkm_obs[0], vkm_obs[1], vkm_obs[2], \
                self.mk, self.t, 0., 0., 0.]).astype(np.float64)

        return obs

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass