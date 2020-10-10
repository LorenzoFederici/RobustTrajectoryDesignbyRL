Environment for a low-thrust interplanetary transfer with RL, 
from fixed initial conditions, towards a given final target.
The mission can be an interception, or a rendezvous.

After you have installed your package with:
pip3 install -e gym-lowthrust
you can create an instance of the environment with:
gym.make('gym_lowthrust:lowthrust-v0')