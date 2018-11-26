## Credit to https://github.com/Microsoft/TextWorld/blob/master/notebooks/Building%20a%20simple%20agent.ipynb
## for providing useful starter code for writing this agent


import numpy as np
import textworld


class RandomAgent(textworld.Agent):
    """ Agent that randomly selects action object pairs. """
    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def reset(self, env):
        env.activate_state_tracking()
        env.compute_intermediate_reward()  

    def act(self, game_state, reward, done):
        #Since we are only considering action object pairs
        return self.rng.choice(game_state.admissible_commands)

