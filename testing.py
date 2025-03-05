import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt

# Load dataset function
def load_and_preprocess_dataset(file_path):
    dataset = pd.read_csv(file_path)
    if 'ModulationType' in dataset.columns:
        dataset.drop(columns=['ModulationType'], inplace=True)
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    dataset.fillna(0, inplace=True)
    return dataset

# QoS Calculation
def calculate_qos(end_to_end_delay, packet_delivery_rate, tau_t=90, w_d=0.5, w_p=0.5, pdr_threshold=0.8):
    if end_to_end_delay < tau_t and packet_delivery_rate > pdr_threshold:
        return w_d * (tau_t - end_to_end_delay) / tau_t + w_p * (packet_delivery_rate - pdr_threshold) / pdr_threshold
    return 0

# Define DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array([state], dtype=np.float32)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return []
        minibatch = random.sample(self.memory, batch_size)
        loss_values = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.array([next_state], dtype=np.float32), verbose=0)[0])
            target_f = self.model.predict(np.array([state], dtype=np.float32), verbose=0)
            target_f[0][action] = target
            history = self.model.fit(np.array([state], dtype=np.float32), target_f, epochs=1, verbose=0)
            loss_values.append(history.history['loss'][0])  # Capture loss
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss_values

# Prepare dataset
def prepare_and_aggregate_data(urban_dataset, highway_dataset):
    urban_dataset = urban_dataset.copy()
    highway_dataset = highway_dataset.copy()
    for dataset in [urban_dataset, highway_dataset]:
        dataset['QoSScore'] = dataset.apply(lambda row: calculate_qos(row['EndToEndDelay'], row['PacketDeliveryRate']), axis=1)
        dataset['TaskSatisfactionRate'] = dataset['PacketDeliveryRate'].apply(lambda pdr: 1 if pdr > 0.8 else 0)
    aggregated_data = pd.concat([urban_dataset, highway_dataset], ignore_index=True)
    return urban_dataset, highway_dataset, aggregated_data

# Train DQN and collect loss values
def train_dqn(agent, dataset, batch_size, metrics):
    loss_values = []
    for index, row in dataset.iterrows():
        state = np.array(row[['TransmissionPower', 'CurrentChannelPowerGain', 'CrossChannelPowerGain', 'QoSScore']], dtype=np.float32)
        action = agent.act(state)
        reward = float(row['QoSScore'])
        next_state = state
        done = index == len(dataset) - 1
        agent.remember(state, action, reward, next_state, done)
        if len(agent.memory) > batch_size:
            batch_loss = agent.replay(batch_size)
            loss_values.extend(batch_loss)
        metrics['packet_delivery_rate'].append(float(row['PacketDeliveryRate']))
        metrics['end_to_end_delay'].append(float(row['EndToEndDelay']))
        metrics['task_satisfaction_rate'].append(int(row['TaskSatisfactionRate']))
        metrics['qos'].append(float(row['QoSScore']))
    return loss_values

# Load datasets
urban_dataset_df = load_and_preprocess_dataset('Urban.csv').head(100)
highway_dataset_df = load_and_preprocess_dataset('Highway.csv').head(100)

# Compute QoS
urban_dataset_df, highway_dataset_df, aggregated_dataset_df = prepare_and_aggregate_data(urban_dataset_df, highway_dataset_df)

# Initialize DQN Agent
state_size = 4
action_size = 4
agent = DQNAgent(state_size, action_size)

# Training parameters
batch_size = 1
num_episodes = 2
early_stopping_threshold = 0.01
patience = 3
best_reward = -np.inf  
no_improvement_count = 0

# Store metrics
urban_metrics = {'packet_delivery_rate': [], 'end_to_end_delay': [], 'task_satisfaction_rate': [], 'qos': []}
highway_metrics = {'packet_delivery_rate': [], 'end_to_end_delay': [], 'task_satisfaction_rate': [], 'qos': []}

# Training loop
for e in range(num_episodes):
    urban_loss_values = train_dqn(agent, urban_dataset_df.sample(frac=0.5, random_state=e), batch_size, urban_metrics)
    highway_loss_values = train_dqn(agent, highway_dataset_df.sample(frac=0.5, random_state=e), batch_size, highway_metrics)
    
    avg_reward = np.mean(urban_metrics['qos'][-5:])
    qos_value = np.mean(urban_metrics['qos'])

    # Early stopping
    if avg_reward > best_reward + early_stopping_threshold:
        best_reward = avg_reward
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= patience:
        print(f"Early stopping at episode {e + 1}")
        break

    print(f"Episode {e + 1}/{num_episodes} - Reward: {avg_reward:.4f}, QoS: {qos_value:.4f}")

# Fix: Ensure the loss values match the dataset length
def adjust_loss_values(dataset, loss_values):
    dataset_length = len(dataset)
    loss_length = len(loss_values)

    if loss_length < dataset_length:
        # Pad with NaN or repeat last value
        loss_values += [np.nan] * (dataset_length - loss_length)
    elif loss_length > dataset_length:
        # Truncate to match the dataset length
        loss_values = loss_values[:dataset_length]

    return loss_values

# Apply the fix
urban_dataset_df['Loss'] = adjust_loss_values(urban_dataset_df, urban_loss_values)
highway_dataset_df['Loss'] = adjust_loss_values(highway_dataset_df, highway_loss_values)

# Save results
urban_dataset_df.to_csv("Urban_Results.csv", index=False)
highway_dataset_df.to_csv("Highway_Results.csv", index=False)
print("Results saved successfully!")
