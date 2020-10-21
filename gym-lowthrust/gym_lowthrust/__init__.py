from gym.envs.registration import register

register(
    id = 'lowthrust-v0', #this variable is what we pass into gym.make() to call our environment.
    entry_point = 'gym_lowthrust.envs:LowThrustEnv',
    kwargs = {'impulsive' : True, 'action_coord' : False, 'obs_type' : 0, \
        'random_obs' : False, 'stochastic' : False, 'mission_type' : True, \
        'NSTEPS' : 40., 'NITER' : 1e6, \
        'eps_schedule' : [1e-1, 1e-2, 1e-3], 'lambda_con' : 50., \
        'Tmax' : 0.08431824475898621, 'ueq' : 0.658507386720677, \
        'tf' : 5.99979199, 'amu' : 1., 'm0' : 1., \
        'r0' : [-0.94050597, -0.3450161, 6.55e-06], \
        'v0' : [0.32817739, -0.94271519, 1.456e-05], \
        'rTf' : [-1.15429151, 1.18288422, 0.05313444], \
        'vTf' : [-0.55154179, -0.49893452, 0.00309385], \
        'sigma_r' : 0., 'sigma_v' : 0., \
        'sigma_u_rot' : 0., 'sigma_u_norm' : 0., \
        'MTE' : False, 'pr_MTE' : 0.}
)