## Credit to https://github.com/Microsoft/TextWorld/blob/master/notebooks/Building%20a%20simple%20agent.ipynb
## for providing useful code for this routine

import numpy as np
import textworld
from randomAgent import RandomAgent

def print_state(gs, ind):
    print("[STATE "+  str(ind) + "]")
    print("[DESCRIPTION]" )
    print(gs.description)
    print("[INVENTORY]")
    print(gs.inventory)
    print("[REWARDS]")
    print(gs.intermediate_reward)
    print("\n\n")

def print_command(com):
    print("-----------------------<COMMAND>--------------------------")
    print(com)
    print("-----------------------<COMMAND>--------------------------")


def run_random_agent(agent, game, max_step=500, nb_episodes=10, verbose = False):
    env = textworld.start(game)  
    print(game.split("/")[-1])
    
    avg_moves, avg_scores = [], []
    for no_episode in range(nb_episodes):
        agent.reset(env)          
        game_state = env.reset() 
        if verbose:
            print_state(game_state, 0)
        reward = 0
        done = False
        for no_step in range(max_step):
            command = agent.act(game_state, reward, done)
            game_state, reward, done = env.step(command)
            if verbose:
                print_command(command)
                print_state(game_state, no_step)
            if done:
                break
        print("Episode " + str(no_episode))
        avg_moves.append(game_state.nb_moves)
        avg_scores.append(game_state.score)

    env.close()
    print("  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / 1.".format(np.mean(avg_moves), np.mean(avg_scores)))


run_random_agent(RandomAgent(), game="../../games/test-game.ulx", verbose = True)
