# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from bomba_de_agua import bomba_de_agua
from q_learning_train import q_learning_train
from sarsa_train import sarsa_train
from dqn_train import dqn_train_numpy,NumPyQAgent, evaluate_policy 


st.set_page_config(layout="wide", page_title="Comparación de Algoritmos RL - Mantenimiento de Bomba")

st.title("⚙️ Comparación de Algoritmos RL para Mantenimiento Preventivo")
st.markdown("---")


def smooth_data(data, window=100):
    """Calcula el promedio móvil para suavizar la curva de aprendizaje."""
    return pd.Series(data).rolling(window=window, min_periods=1).mean().tolist()



st.header("1. Parámetros de Entrenamiento")

col1, col2, col3 = st.columns(3)

with col1:
    num_episodes = st.slider("Número de Episodios", 1000, 15000, 5000, 500)
with col2:
    alpha = st.slider("Tasa de Aprendizaje (Alpha)", 0.01, 0.5, 0.1, 0.01)
    gamma = st.slider("Factor de Descuento (Gamma)", 0.8, 0.99, 0.95, 0.01)
with col3:
    epsilon_decay = st.slider("Decaimiento de Epsilon", 0.99, 0.9999, 0.999, 0.0001)
    smoothing_window = st.slider("Ventana de Suavizado (Episodios)", 10, 500, 100, 10)

st.markdown("---")



if st.button("🚀 Ejecutar y Comparar Modelos RL"):
    st.header("2. Resultados del Entrenamiento")
    st.warning(f"Entrenando los 3 modelos con {num_episodes} episodios. Esto puede tardar unos minutos...", icon="⏳")
    
    
    env_q = bomba_de_agua()
    Q_Q, rewards_Q = q_learning_train(env_q, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)
    unnecessary_rate_q, failures_q = evaluate_policy(env_q, Q_Q) 

    
    env_sarsa = bomba_de_agua()
    Q_SARSA, rewards_SARSA = sarsa_train(env_sarsa, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)
    unnecessary_rate_sarsa, failures_sarsa = evaluate_policy(env_sarsa, Q_SARSA) 

    
    env_dqn = bomba_de_agua()
    Q_DQN_numpy, rewards_DQN = dqn_train_numpy(env_dqn, num_episodes=num_episodes) 
    unnecessary_rate_dqn, failures_dqn = evaluate_policy(env_dqn, Q_DQN_numpy) 

    st.success("✅ Entrenamiento Finalizado. Generando gráficos...")
    
    st.session_state['rewards_Q'] = rewards_Q
    st.session_state['rewards_SARSA'] = rewards_SARSA
    st.session_state['rewards_DQN'] = rewards_DQN
    
    st.session_state['unnecessary_rate_q'] = unnecessary_rate_q
    st.session_state['failures_q'] = failures_q
    
    st.session_state['unnecessary_rate_sarsa'] = unnecessary_rate_sarsa
    st.session_state['failures_sarsa'] = failures_sarsa
    
    st.session_state['unnecessary_rate_dqn'] = unnecessary_rate_dqn
    st.session_state['failures_dqn'] = failures_dqn
    

    st.session_state['smoothing_window'] = smoothing_window
    
    
    df_raw = pd.DataFrame({
        'Episodio': range(num_episodes),
        'Q-Learning': rewards_Q,
        'SARSA': rewards_SARSA,
        'DQN': rewards_DQN
    })
    
    
    df_smoothed = pd.DataFrame({
        'Episodio': range(num_episodes),
        'Q-Learning': smooth_data(rewards_Q, smoothing_window),
        'SARSA': smooth_data(rewards_SARSA, smoothing_window),
        'DQN': smooth_data(rewards_DQN, smoothing_window)
    })
    
    
    st.subheader("3.1. Curvas de Aprendizaje Suavizadas (Recompensa Promedio)")
    st.info(f"El valor de la recompensa acumulada debe aumentar y estabilizarse. Se usa una ventana de suavizado de {smoothing_window} episodios.")

    df_melted_smoothed = df_smoothed.melt('Episodio', var_name='Algoritmo', value_name='Recompensa Promedio')
    fig_smoothed = px.line(df_melted_smoothed, x='Episodio', y='Recompensa Promedio', color='Algoritmo', 
                           title='Comparación de la Curva de Aprendizaje (Promedio Móvil)',
                           labels={'Recompensa Promedio': 'Recompensa Promedio (Suavizada)'})
    st.plotly_chart(fig_smoothed, use_container_width=True)

    
    st.subheader("3.2. Recompensa Acumulada Bruta por Episodio")
    st.info("Muestra la alta varianza del aprendizaje (Exploración/Explotación).")

    df_melted_raw = df_raw.melt('Episodio', var_name='Algoritmo', value_name='Recompensa Bruta')
    fig_raw = px.line(df_melted_raw, x='Episodio', y='Recompensa Bruta', color='Algoritmo', 
                      title='Recompensa Bruta por Episodio (Varianza)',
                      
                      labels={'Recompensa Bruta': 'Recompensa Total'})
    st.plotly_chart(fig_raw, use_container_width=True)

    

st.subheader("3.3. Tabla Comparativa de Desempeño Final")


if 'rewards_Q' in st.session_state:
    
    
    rewards_Q = st.session_state['rewards_Q']
    rewards_SARSA = st.session_state['rewards_SARSA']
    rewards_DQN = st.session_state['rewards_DQN']
    
    
    smoothing_window = st.session_state.get('smoothing_window', 100)
    unnecessary_rate_q = st.session_state['unnecessary_rate_q']
    unnecessary_rate_sarsa = st.session_state['unnecessary_rate_sarsa']
    unnecessary_rate_dqn = st.session_state['unnecessary_rate_dqn']
    failures_q = st.session_state['failures_q']
    failures_sarsa = st.session_state['failures_sarsa']
    failures_dqn = st.session_state['failures_dqn']

    
    q_avg = np.mean(rewards_Q[-smoothing_window:])
    sarsa_avg = np.mean(rewards_SARSA[-smoothing_window:])
    dqn_avg = np.mean(rewards_DQN[-smoothing_window:])

    
    comparison_data = {
        'Algoritmo': ['Q-Learning', 'SARSA', 'DQN'],
        f'Recompensa Promedio Final (últimos {smoothing_window})': [f'{q_avg:.2f}', f'{sarsa_avg:.2f}', f'{dqn_avg:.2f}'],
        'Recompensa Máxima Absoluta': [np.max(rewards_Q), np.max(rewards_SARSA), np.max(rewards_DQN)],
        
        
        'Tasa Mantenimiento Innecesario (TEST)': [f'{unnecessary_rate_q:.2%}', f'{unnecessary_rate_sarsa:.2%}', f'{unnecessary_rate_dqn:.2%}'],
        'Fallos Totales (TEST)': [failures_q, failures_sarsa, failures_dqn]
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)

    st.markdown("---")
    st.success("Conclusión: La **Tasa de Mantenimiento Innecesario** debe ser la más baja para el algoritmo óptimo, minimizando paradas de producción.")

else:
    
    st.info("Presiona 'Ejecutar y Comparar Modelos RL' para generar la tabla.")

    