import numpy as np
import random
from collections import deque

class bomba_de_agua:
    def __init__(self):
        
        self.state_space_size = 6
        
        self.action_space_size = 3
        self.state = 0 
        self.current_hours = 0
        self.hours_limit = 100 

        
        self.R_FALLA = -1000.0
        self.R_MANTENIMIENTO_NECESARIO = 100.0
        self.R_MANTENIMIENTO_PREMATURO = -50.0
        self.R_PARADA_EMERGENCIA = -300.0
        self.R_OPERACION = -1.0 
        
        self.P_fallo = np.array([0.001, 0.005, 0.01, 0.05, 0.10, 0.20])
        
        
        self.P_deterioro = 0.3
    
    def reset(self):
        """Reinicia el entorno al estado óptimo (Baja, Nueva)."""
        self.state = 0
        self.current_hours = 0
        return self.state

    def step(self, action):
        """Realiza una acción y devuelve el nuevo estado, la recompensa y si el episodio ha terminado."""
        current_state = self.state
        reward = 0
        done = False
        
        

        if action == 0:  
            self.current_hours += 1
            reward = self.R_OPERACION
            
            
            if random.random() < self.P_fallo[current_state]:
                reward = self.R_FALLA
                self.state = self.reset() 
                done = True
            
            
            elif not done:
                
                is_old = 1 if self.current_hours > self.hours_limit else 0

                
                vibration_level = current_state % 3 
                if random.random() < self.P_deterioro and vibration_level < 2:
                    vibration_level += 1
                
                
                self.state = vibration_level + (is_old * 3)
            
        elif action == 1: 
            
            if current_state >= 3: 
                reward = self.R_MANTENIMIENTO_NECESARIO
            else: 
                reward = self.R_MANTENIMIENTO_PREMATURO
            
            self.state = self.reset() 
            done = True
            
        elif action == 2: 
            reward = self.R_PARADA_EMERGENCIA
            self.state = self.reset() 
            done = True
            
        return self.state, reward, done