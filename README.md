# Lunar Lander - Deep Q-Network (DQN) Control

This project is implemented as part of the Reinforcement Learning course by DeepLearning.AI and Stanford Online. It solves the **Lunar Lander** environment using **Deep Q-Learning** and **Experience Replay**.

##  Key Features

- **DQN Architecture:** A neural network that maps states to action values ($Q$ values):
  - An Input layer (`state_size`)[cite: 3]
  - A Dense layer with 64 units and `relu` activation[cite: 3]
  - A second Dense layer with 64 units and `relu` activation[cite: 3]
  - An output Dense layer with `num_actions` units and `linear` activation[cite: 3]
- **Experience Replay:** A memory buffer with random mini-batch sampling to break strong temporal correlations and stabilize training[cite: 3].
- **Target Q-Network (`target_q_network`):** A separate target network used to maintain target stability during training updates[cite: 3].
- **$\epsilon$-greedy Policy:** Ensures exploration early in training and gradually decays toward a minimum threshold (0.01)[cite: 4].

##  File Structure & Components

1. **Q-Network & Target Network:** Two identical network architectures used for action-value estimation[cite: 3].
2. **`compute_loss`:** Calculates $y$ targets using the Bellman equation (handled cleanly via `done_vals`) and computes the Mean-Squared Error (MSE) loss[cite: 4].
3. **`agent_learn`:** Custom training loop utilizing `tf.GradientTape` for gradient descent and weight updates[cite: 4].
4. **`lunar_lander_model.h5`:** Saved weights of the fully trained model[cite: 4].

##  Training & Evaluation

The agent samples a random mini-batch from the experience memory buffer to perform learning updates every 4 steps[cite: 4]. The environment is considered solved when the agent achieves an average score of **200 points or higher over the last 100 consecutive episodes**, at which point the trained model is automatically saved[cite: 4].
