# Reinforcement Learning for Robust Low-Thrust Interplanetary Trajectory Design

This code implements a Reinforcement Learning (RL) framework to deal with the robust design of mass-optimal, time-fixed, low-thrust interplanetary transfers
with fixed initial and final conditions. Specifically, the mission can be an interception (fixed final position), or a rendezvous (fixed final position and velocity).
The Sims-Flanagan method is used to model the low-thrust trajectory, that is divided into a number of segments. Two different thrust models are implemented:
1. impulsive thrust, with the magnitude of any impulse limited by the amount of velocity variation that could be accumulated over the duration of the corresponding segment;
2. piecewise constant thrust.

The environment can be deterministic or stochastic (i.e., with state, observation and/or control uncertainties, as well as with a single or multiple consecutive missed thrust events).

![My Image](https://github.com/LorenzoFederici/RobustTrajectoryDesignbyRL/blob/main/images/traj_compare.png?raw=true)

The program is based on [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/), an open-source library containing a set of improved implementations of RL algorithms based on [OpenAI Baselines](https://github.com/openai/baselines).
All necessary information can be found in the corresponding GitHub repository.

Moreover, the program uses [OpenMPI](https://www.open-mpi.org/) message passing interface implementation to run different environment realizations in parallel on several CPU cores during training.

## Installation

To correctly set up the code on Linux (Ubuntu), please follow the present instructions:

1. First, you need Python3 and the system packages CMake, OpenMPI and zlib. You can install all these packages with:
    ```
    $ sudo apt-get update
    $ sudo apt-get install cmake openmpi-bin libopenmpi-dev python3-dev zlib1g-dev
    ```
2. After installing [Anaconda](https://www.anaconda.com/distribution/), you can create a virtual environment with a specific Python version (3.6.10) by using conda:
    ```
    $ conda create --name myenv python=3.6.10
    ```
    where `myenv` is the name of the environment, and activate the environment with:
    ```
    $ conda activate myenv
    ```
    When the conda environment is active, your shell prompt is prefixed with (myenv).
    The environment can be deactivated with:
    ```
    $ conda deactivate
    ```

3. Install all required packages in the virtual environment by using pip and the requirement file in the current repository:

    ```
    (myenv)$ cd /path-to-cloned-GitHub-repository/RobustTrajectoryDesignbyRL/
    (myenv)$ pip install -r requirements.txt
    ```
4. Install the gym RL environment with pip:
    ```
    (myenv)$ pip install -e gym-lowthrust
    ```
Now, you can verify that all packages were successfully compiled and installed by running the following command:

```
(myenv)$ python main_lowthrust_validate.py
```
If this command executes without any error, then your RobustTrajectoryDesignbyRL installation is ready for use.


Eventually, a LaTeX distribution should be installed on the computer in order to correctly visualize all graphs and plots. As an example, to install [Tex Live LaTeX](https://www.tug.org/texlive/) distribution on Ubuntu use the following command.
```
$ sudo apt-get install texlive-full
```

## User Guide

The program is composed by 3 main Python scripts (`main_lowthrust_(...).py`), the RL environment (in `gym-lowthrust/`), created by using the [OpenAI Gym](https://gym.openai.com/) toolkit, and a number of additional Python modules (in `custom_modules/`) with useful functions.

Specifically:

1. `main_lowthrust_input.py` is the main file, that must be called to train the agent in a specific RL environment. 

    All the environment and program settings must be included in an external text file with ad-hoc formatting, to be placed in folder `settings_files/` and given as input to the script.
For example, to start training the agent with the settings specified in file `settings_files/settings.txt`, use the following command:
    ```
    (myenv)$ python main_lowthrust_input.py --settings settings.txt
    ```
    The information that must be included in the settings file, together with their formatting, is specified in file `settings_files/README.md`. 
    The settings files that are already present in folder `settings_files/` are those used to obtain the results presented in paper https://arxiv.org/abs/2008.08501. You can look at them to better understand how to prepare new "legal" settings files from scratch. 

    The specific low-thrust mission to study must be specified in a .dat file to be placed in folder `missions/`.
    The file should contain at least three rows, the first containing the columns headers, and the other two specifying the time (`t`), the spacecraft position (`x`, `y`, `z`) velocity (`vx`, `vy`, `vz`) and mass (`m`) at the beginning and at the end of the mission, respectively (actually, the final mass can be a fictitious number). All quantities are assumed to be non-dimensional with respect to the reference values reported at lines 133-139 of `main_lowthrust_input.py`.
    The name of the mission file must be specified in the settings file.
    For example, file `Earth_Mars.dat` contains the optimal spacecraft state an any time for the Earth-Mars rendezvous mission used as case study in paper https://arxiv.org/abs/2008.08501.
    
    It is also possible to re-train with the same or different settings a pretrained RL model. In this case,
   the pretrained model file, once named `final_model.zip`, must be placed in `settings_files/` folder, and the program called with the command:
    ```
    (myenv)$ python main_lowthrust_input.py --settings settings.txt --input_model model
    ```

    At the end of the training, the RL model and all output files are saved in directory `sol_saved/sol_(i)/`, where number (i) depends on how many solutions are already contained in this folder.
    The model corresponding to the policy trained in the unperturbed environment of paper https://arxiv.org/abs/2008.08501 can be found in directory `sol_saved/sol_1/`.

2. `main_lowthrust_load.py` is the file that allows generating the plots of the robust trajectory and of the control that corresponds to a given model (policy).
Let us suppose that we want to realize the plots for the model saved in folder `sol_saved/sol_1/`. It is sufficient to run the command:
    ```
    (myenv)$ python main_lowthrust_load.py --settings sol_saved/sol_1/settings.txt --n_sol 1
    ```
    The graphs are saved in the same directory.
3. `main_lowthrust_MC.py` is the file that allows performing a Monte Carlo simulation of a given policy in a stochastic environment.
Let us suppose that we want to realize a MC by using the model saved in folder `sol_saved/sol_1/`. It is sufficient to run the command:
    ```
    (myenv)$ python main_lowthrust_MC.py --settings sol_saved/sol_1/settings.txt --n_sol 1
    ```
    The graphs and output files are saved in the same directory.


Enjoy the code!
