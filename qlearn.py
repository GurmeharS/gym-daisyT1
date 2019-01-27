#!pip -q install gym
import numpy as np
import pandas as pd
import gym
import gym_daisyT1
import random
LastEpisode = False
env=gym.make('daisyT1-v0')
env.render()

action_size = (env.action_space.n)
print("Action size ",action_size)
state_size = len(env.observation_space)
print("State size ",state_size)

qtable = np.zeros((state_size,action_size))
#qtable[0,0]=10
print(qtable)

total_episodes = 1000 #total episodes
total_test_episodes = 1 #total test episodes
max_steps = env.TOTAL_STEPS #max steps per episode

learning_rate = 0.7 #learning rate
gamma = 0.618 #discount rate

epsilon = 1.0 #exploration rate
max_epsilon = 1.0 #max expl. probability at start
min_epsilon = 0.01 #min expl. probability 
decay_rate = 0.18 #exp. decay for expl.

for episode in range(total_episodes):
    state = env.reset()
    step=0
    done=False
    for step in range(env.TOTAL_STEPS):
        exp_exp_tradeoff = random.uniform(0,1)
        #print(exp_exp_tradeoff)
        #exploitation
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        
        #random choice - exploration
        else:
            action = env.action_space.sample()
        #take action and record state s' and reward

        new_state, reward, done, info = env.step(action)
        #action >0 here, <40
        #if abs(action) > 20: print("wow")
        #Update Q(s,a):= Q(s,a)+
        qtable[state,action] = qtable[state,action] + learning_rate*(reward + gamma*np.max(qtable[new_state,:]) - qtable[state,action])

        #print([state,action])
        state = new_state
        
        if done==True:
            break
    
    episode += 1
    #print(epsilon)
    #reduce epsilon to move towards exploitation
    epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
#print(env.memory)

#print(qtable)
env.reset()
rewards=[]

for episode in range(total_test_episodes):
    state = env.reset()
    step=0
    done=False
    total_rewards = 0
    finalActions = []
    print('*******************************')
    print('Episode is ', episode)
    
    for step in range(max_steps):
        #env.render()
        #take action that gives maximum expected future reward given state
        action=np.argmax(qtable[state,:])

        #print(qtable[state,:])
        #print(action)

        new_state, reward, done, info = env.step(action)
        
        total_rewards += reward
        finalActions += [np.argmax(qtable[state,:])-env.MAX_ACC]
        if done:
            rewards.append(total_rewards)
            print('Score ', total_rewards)
            break
        state=new_state
print(env.MAX_VEL_ARR)
print(finalActions)
df = pd.DataFrame(finalActions, columns=['a'])
df.to_csv("instructions_1.csv", sep='\t')
env.close()
print('Score over time: ' + str(sum(rewards)/total_test_episodes))