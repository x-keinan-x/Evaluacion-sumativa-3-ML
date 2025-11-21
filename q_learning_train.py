import numpy as np
import random

def q_learning_train(env, num_episodes=5000, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999):
    Q_table = np.zeros((env.state_space_size, env.action_space_size))
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            
            if random.random() < epsilon:
                action = random.randint(0, env.action_space_size - 1) 
            else:
                action = np.argmax(Q_table[state, :]) 

            
            next_state, reward, done = env.step(action)
            
            
            max_future_Q = np.max(Q_table[next_state, :])
            
            Q_table[state, action] = Q_table[state, action] + alpha * (
                reward + gamma * max_future_Q - Q_table[state, action]
            )
            
            state = next_state
            total_reward += reward

        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        
    print("Q-Learning completado.")
    return Q_table, episode_rewards

