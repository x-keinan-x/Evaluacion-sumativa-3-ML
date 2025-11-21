import numpy as np
import random
from collections import deque


class NumPyQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1 

        
        self.model_weights = np.zeros((self.state_size, self.action_size))
        
        
        self.target_model_weights = np.zeros((self.state_size, self.action_size))
        self.update_target_model()

    def update_target_model(self):
        """Copia los pesos del modelo principal al modelo objetivo."""
        self.target_model_weights = self.model_weights.copy()

    def remember(self, state, action, reward, next_state, done):
        """Almacena la experiencia en el buffer."""
        
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Elige la acción usando epsilon-greedy (lectura directa de la matriz)."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        
        return np.argmax(self.model_weights[state, :])

    def replay(self, batch_size):
        """Entrena el 'modelo' (actualiza la matriz de pesos) muestreando experiencias."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            
            if done:
                target = reward
            else:
                
                target_q_values = self.target_model_weights[next_state, :]
                target = reward + self.gamma * np.max(target_q_values)

            
            current_q = self.model_weights[state, action]
            
            
            new_q = current_q + self.learning_rate * (target - current_q)
            
            
            self.model_weights[state, action] = new_q
        
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def dqn_train_numpy(env, num_episodes=5000, batch_size=32, target_update_freq=10):
    agent = NumPyQAgent(env.state_space_size, env.action_space_size)
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            
            agent.replay(batch_size)
        
        
        if episode % target_update_freq == 0:
            agent.update_target_model()
            
        episode_rewards.append(total_reward)
        
    print("DQN (NumPy) completado.")
   
    return agent.model_weights, episode_rewards

def evaluate_policy(env, Q_matrix, num_steps=1000):
    """
    Evalúa la matriz Q (política) para calcular métricas de negocio.
    Asume que la matriz Q es la Q-Table (Q-Learning/SARSA) o la matriz de pesos (NumPy DQN).
    """
    total_maintenance_actions = 0
    unnecessary_maintenance_actions = 0
    total_failures = 0
    
    
    for _ in range(num_steps):
        state = env.reset()
        done = False
        
        while not done:
            
            action = np.argmax(Q_matrix[state, :])
            
            
            next_state, reward, done = env.step(action)
            
            
            if action == 1: 
                total_maintenance_actions += 1
                
                if state < 3:
                    unnecessary_maintenance_actions += 1
            
            
            if reward == env.R_FALLA:
                total_failures += 1
                
            state = next_state
            
            if done:
                break
                
    
    if total_maintenance_actions > 0:
        unnecessary_rate = unnecessary_maintenance_actions / total_maintenance_actions
    else:
        unnecessary_rate = 0.0 
        
    return unnecessary_rate, total_failures