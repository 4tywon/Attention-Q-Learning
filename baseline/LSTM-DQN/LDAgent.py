import textworld


class LSTM_DQN_Agent(textworld.Agent):
    def __init__(self):


    def reset(self, env):
        env.activate_state_tracking()  
        env.compute_intermediate_reward()  

    def act(self, game_state, reward, done):

        return game_state.admissible_commands[0]
    
    def finish(self, game_state, reward, done):

        pass