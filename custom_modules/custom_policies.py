from stable_baselines.common.policies import FeedForwardPolicy, LstmPolicy, register_policy
import tensorflow as tf

# Custom MLP policy of three layers of size 64 each
class CustomPolicy_3x64(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_3x64, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64, 64, 64],
                                                          vf=[64, 64, 64])],
                                           feature_extraction="mlp")
                                        
# Custom MLP policy of two layers of size 64 each + a shared layer of size 64
class CustomPolicy_2x64_shared(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_2x64_shared, self).__init__(*args, **kwargs,
                                           net_arch=[64, dict(pi=[64, 64],
                                                          vf=[64, 64])],
                                           feature_extraction="mlp")

# Custom MLP policy of three layers of size 128 each 
# + one shared layer of size 128
class CustomPolicy_4x128(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_4x128, self).__init__(*args, **kwargs,
                                           net_arch=[128, dict(pi=[128, 128, 128],
                                                            vf=[128, 128, 128])],
                                           feature_extraction="mlp")

# Custom LSTM policy with two MLP layers of size 64 each + a shared LSTM layer of size 4
class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=4, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=['lstm', dict(pi=[64, 64],
                                            vf=[64, 64])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

# Register the policy, it will check that the name is not already taken
register_policy('CustomPolicy_3x64', CustomPolicy_3x64)
register_policy('CustomPolicy_2x64_shared', CustomPolicy_2x64_shared)
register_policy('CustomPolicy_4x128', CustomPolicy_4x128)
register_policy('CustomLSTMPolicy', CustomLSTMPolicy)