import gym 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow_probability as tfp
from Actor_Critic import Agent

def ploting(x,history,figure_file):
    episodes_avg = np.zeros(len(history))
    for i in range(len(history)):
        episodes_avg[i] = np.mean(history[i])

    return episodes_avg




if __name__ == "__main__" : 
    env = gym.make('CartPole-v1')
    agent = Agent(env.action_space.n,units=[128,128])

    num_games = 1500

    best_score = env.reward_range[0]
    score_history = []


    for i in range(num_games):
        state = env.reset()
        done = False
        score = 0 

        while not done :
            env.render()
            action = agent.choose_action(state)

            state_,reward , done , _ = env.step(action)
            score+=reward
            agent.train(state,reward,state_,done)
            state = state_
        
        score_history.append(score)
        
        if score > best_score : 
            best_score = score
        
        print(f"episode :{i}\tscore :{score}\tbest score :{best_score}")


