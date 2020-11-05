## How to prepare a correct settings file

The settings file contains a number of parameters that the program needs to know in order to understand what kind of mission to simulate, what RL algorithm to use, and so on.
The parameters must be organized in two columns, the first with the names of the parameters, and the second with their values. The order in which they are written in the file is not important.

The list of parameters to include is the following:

|        Name        |  Type  |                                  Meaning                                 |                                                   Value                                                   |
|:------------------:|:------:|:------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|
|    `load_model`    |  bool  |                 Do you want to train a pre-trained model?                |                                            `1` (yes), `0` (no)                                            |
|     `env_name`     | string |                          Name of the environment                         |                                               `lowthrust-v0`                                              |
|   `mission_name`   | string |       Name of the mission file (without .dat) in folder `missions/`      |                                               The file name                                               |
|     `impulsive`    |  bool  |        The spacecraft has impulsive or piece-wise constant thrust?       |                                      `1` (impulsive), `0` (constant)                                      |
|       `Tmax`       |  float |                     Maximum thrust of S/C engine (kN)                    |                                              A positive float                                             |
|        `Isp`       |  float |                    Specific impulse of S/C engine (s)                    |                                              A positive float                                             |
|   `action_coord`   |  bool  |   What is the reference frame used to specify the direction of thrust?   |                                           `0` (J200), `1` (RTN)                                           |
|     `obs_type`     |  bool  |             Parameters of the spacecraft used as observations            |                        `0` (r and v), `1` (COEs), `2` (MEEs), `3`(r, v and action)                        |
|    `random_obs`    |  bool  |                      Are the observations uncertain?                     |                                            `0` (no), `1` (yes)                                            |
|    `stochastic`    |  bool  |                      Is the environment stochastic?                      |                                            `0` (no), `1` (yes)                                            |
|   `mission_type`   |  bool  |                              Type of mission                             |                                    `0` (interception), `1` (rendezvous)                                   |
|      `NSTEPS`      |   int  |                 Number of time-steps per training episode                |                                             An integer number                                             |
|   `eps_schedule`   |  list  |       The schedule used for the constraint satisfaction tolerance        |                                           A list of floats << 1                                           |
|    `lambda_con`    |  float |     Weight of terminal constraint violation term in the reward signal    |                                                  A float                                                  |
|      `sigma_r`     |  float |          Standard deviation of S/C (observed) position (non-dim)         |                                                  A float                                                  |
|      `sigma_v`     |  float |          Standard deviation of S/C (observed) velocity (non-dim)         |                                                  A float                                                  |
|    `sigma_u_rot`   |  float | Standard deviation of Euler angles defining thrust/DV misalignment (rad) |                                              A float in 0,pi                                              |
|   `sigma_u_norm`   |  float |      Standard deviation of the error on thrust/DV norm (percentage)      |                                               A float in 0,1                                              |
|        `MTE`       |  bool  |                             Does a MTE occur?                            |                                            `0` (no), `1` (yes)                                            |
|      `pr_MTE`      |  float |            Probability of having a new MTE after the last one            |                                               A float in 0,1                                              |
|      `num_cpu`     |   int  |             Number of parallel environments used for training            |                                         An integer greater than 1                                         |
|     `algorithm`    | string |                               RL algorithm                               |                                 `PPO`, `A2C`, `DDPG`, `SAC`, `TD3`, `TRPO`                                |
| `learning_rate_in` |  float |                    Initial value of the learning rate                    |                                              A positive float                                             |
|   `clip_range_in`  |  float |              Initial value of the clip range (only for PPO)              |                                              A positive float                                             |
|   `learning_rate`  | string |                           Type of learning rate                          |                                     `lin` (linear), `const` (constant)                                    |
|    `clip_range`    | string |                            Type of clip range (only for PPO)                           |                                     `lin` (linear), `const` (constant)                                    |
|     `ent_coef`     |  float |         Coefficient of the entropy term in the performance index (only for PPO, A2C, SAC, TRPO)         |                                              A positive float                                             |
|       `gamma`      |  float |                    Discount factor for future rewards                    |                                               A float in 0,1                                              |
|        `lam`       |  float |                Generalized Advantage Estimator coefficient (only for PPO, TRPO)               |                                               A float in 0,1                                              |
|    `noptepochs`    |   int  |          Number of stochastic gradient ascent epochs per update (only for PPO)         |                                                 An integer                                                |
|   `nminibatches`   |   int  |                      Number of episodes per rollout                      |                                                 An integer                                                |
|      `policy`      | string |                     Actor-Critic Deep Neural Network                     | `MlpPolicy`, `MlpLstmPolicy`, `LstmPolicy`, `CustomPolicy_3x64`, `CustomPolicy_4x128`, `CustomLSTMPolicy`, `CustomPolicy_2x81`, `CustomPolicy_3_var` |
|       `niter`      |   int  |                      Total number of training steps                      |                                              A (huge) integer                                             |
