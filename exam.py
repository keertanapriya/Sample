# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 22:03:15 2024

@author: keertanapriya
"""

# H_table = {'A': 0.2, 'C': 0.3, 'G': 0.3, 'T': 0.2}
# L_table = {'A': 0.3, 'C': 0.2, 'G': 0.2, 'T': 0.3}
# start = {'H': 0.5, 'L': 0.5}
# H_H = 0.5
# H_L = 0.5
# L_H = 0.4
# L_L = 0.6
# seq = 'GGCACTGAA'
# H = []
# L = []
# state_sequence = []

# # Compute the probabilities for the first observation
# first_H = start['H'] * H_table[seq[0]]
# first_L = start['L'] * L_table[seq[0]]
# H.append(first_H)
# L.append(first_L)

# # Determine the initial state based on the highest probability
# if first_H > first_L:
#     state_sequence.append('H')
# else:
#     state_sequence.append('L')

# # Iterate over the remaining observations
# for i in range(1, len(seq)):
#     sum_H = sum_L = 0
    
#     # Calculate the probabilities for transitioning to H
#     sum_H += H[len(H) - 1] * H_H * H_table[seq[i]]
#     sum_H += L[len(L) - 1] * L_H * H_table[seq[i]]
#     #sum_H = max(prob_H, prob_L)

#     # Calculate the probabilities for transitioning to L
#     sum_L += L[len(L) - 1] * L_L * L_table[seq[i]]
#     sum_L += H[len(H) - 1] * H_L * L_table[seq[i]]
#     #sum_L = max(prob_H, prob_L)

#     # Store the probabilities for the current observation
#     H.append(sum_H)
#     L.append(sum_L)

#     # Determine the state with the highest probability
#     if sum_H > sum_L:
#         state_sequence.append('H')
#     else:
#         state_sequence.append('L')

# # Compute the probability of the entire sequence
# prob = H[len(H) - 1] + L[len(L) - 1]
# print("Probability of the sequence:", prob)

# # Print the expected state sequence
# print("Expected state sequence:", "".join(state_sequence))
# -------------------------------------------------------
# import numpy as np

# class HiddenMarkovModel:
#     def __init__(self, states, observations, initial_probabilities, transition_matrix, emission_matrix):
#         self.states = states
#         self.observations = observations
#         self.initial_probabilities = initial_probabilities
#         self.transition_matrix = transition_matrix
#         self.emission_matrix = emission_matrix

#     def _forward_algorithm(self, observation_sequence):
#         T = len(observation_sequence)
#         N = len(self.states)
#         alpha = np.zeros((T, N))

#         #Initialization
#         alpha[0] = self.initial_probabilities * self.emission_matrix[:, self.observations.index(observation_sequence[0])]

#         #recursion
#         for t in range(1, T):
#             for j in range(N):
#                 alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix[:, j]) * self.emission_matrix[j, self.observations.index(observation_sequence[t])]

#         return alpha

#     def _backward_algorithm(self, observation_sequence):
#         T = len(observation_sequence)
#         N = len(self.states)
#         beta = np.zeros((T, N))

#         beta[-1] = 1

#         for t in range(T-2, -1, -1):
#             for i in range(N):
#                 beta[t, i] = np.sum(beta[t+1] * self.transition_matrix[i, :] * self.emission_matrix[:, self.observations.index(observation_sequence[t+1])])

#         return beta

#     def predict_sequence_probability(self, observation_sequence):
#         alpha = self._forward_algorithm(observation_sequence)
#         return np.sum(alpha[-1])

#     def predict_state_sequence(self, observation_sequence):
#         T = len(observation_sequence)
#         N = len(self.states)
#         delta = np.zeros((T, N))
#         psi = np.zeros((T, N))

#         delta[0] = self.initial_probabilities * self.emission_matrix[:, self.observations.index(observation_sequence[0])]

#         for t in range(1, T):
#             for j in range(N):
#                 delta[t, j] = np.max(delta[t-1] * self.transition_matrix[:, j]) * self.emission_matrix[j, self.observations.index(observation_sequence[t])]
#                 psi[t, j] = np.argmax(delta[t-1] * self.transition_matrix[:, j])

#         #backtracking
#         state_sequence = [np.argmax(delta[-1])]
#         for t in range(T-2, -1, -1):
#             state_sequence.insert(0, int(psi[t+1, state_sequence[0]]))

#         return [self.states[i] for i in state_sequence]

# states = ['Sunny', 'Rainy']
# observations = ['Walk', 'Shop', 'Clean']
# initial_probabilities = np.array([0.6, 0.4])
# transition_matrix = np.array([[0.7, 0.3],
#                                [0.4, 0.6]])
# emission_matrix = np.array([[0.1, 0.4, 0.5],
#                              [0.6, 0.3, 0.1]])

# hmm = HiddenMarkovModel(states, observations, initial_probabilities, transition_matrix, emission_matrix)

# observation_sequence = ['Walk', 'Shop', 'Clean']
# probability = hmm.predict_sequence_probability(observation_sequence)
# state_sequence = hmm.predict_state_sequence(observation_sequence)
# print("Probability of observing sequence {} is {:.4f}".format(observation_sequence, probability))
# print("Most likely state sequence:", state_sequence)
#----------------------------------------------------
# def expected_sum_discounted_rewards(states, rewards, transitions, discount_factor, initial_state):


#   # Initialize a dictionary to store expected rewards for each state
#   value_of_states = {state: 0 for state in states}

#   # Iterate for multiple steps to achieve convergence (optional)
#   for _ in range(10):  # Adjust the number of iterations as needed
#     # Update value for each state using the Bellman equation
#     for state in states:
#       expected_future_reward = 0
#       for next_state, probability in transitions[state].items():
#         # Expected future reward considering all possible next states
#         expected_future_reward += probability * (rewards[next_state] + discount_factor * value_of_states[next_state])
#       value_of_states[state] = rewards[state] + discount_factor * expected_future_reward

#   return value_of_states

# # Example usage (replace placeholders with your actual values)
# states = ["Sunny", "Windy", "Hail"]
# rewards = {"Sunny": 4, "Windy": 0, "Hail": -8}
# transitions = {
#   "Sunny": {"Sunny": 0.5, "Windy": 0.5, "Hail": 0.5},
#   "Windy": {"Sunny": 0.5, "Windy": 0.5, "Hail": 0.5},
#   "Hail": {"Sunny": 0.5, "Windy": 0.5, "Hail": 0.5},
# }
# discount_factor = 0.9
# initial_state = "Sunny"

# expected_rewards = expected_sum_discounted_rewards(states, rewards, transitions, discount_factor, initial_state)

# print(f"Expected sum of discounted rewards for each state:")
# for state, reward in expected_rewards.items():
#   print(f"\t{state}: {reward:.2f}")
# ---------------------------------------------------
# import numpy as np

# class MarkovDecisionProcess:
#     def __init__(self, num_states, num_actions, transition_probabilities, rewards, discount_factor=0.9, tolerance=1e-6):
#         self.num_states = num_states
#         self.num_actions = num_actions
#         self.transition_probabilities = transition_probabilities
#         self.rewards = rewards
#         self.discount_factor = discount_factor
#         self.tolerance = tolerance

#     def value_iteration(self):
#         V = np.zeros(self.num_states)
#         while True:
#             V_new = np.zeros(self.num_states)
#             for s in range(self.num_states):
#                 Q = np.zeros(self.num_actions)
#                 for a in range(self.num_actions):
#                     for s_prime in range(self.num_states):
#                         Q[a] += self.transition_probabilities[s][a][s_prime] * (self.rewards[s][a][s_prime] + self.discount_factor * V[s_prime])
#                 V_new[s] = np.max(Q)
#             if np.max(np.abs(V - V_new)) < self.tolerance:
#                 break
#             V = V_new
#         policy = np.zeros(self.num_states, dtype=int)
#         for s in range(self.num_states):
#             Q = np.zeros(self.num_actions)
#             for a in range(self.num_actions):
#                 for s_prime in range(self.num_states):
#                     Q[a] += self.transition_probabilities[s][a][s_prime] * (self.rewards[s][a][s_prime] + self.discount_factor * V[s_prime])
#             policy[s] = np.argmax(Q)
#         return V, policy

# # Example usage
# num_states = 3
# num_actions = 2
# transition_probabilities = np.array([[[0.7, 0.3, 0.0],
#                                       [0.1, 0.9, 0.0]],
#                                      [[0.0, 0.8, 0.2],
#                                       [0.0, 0.0, 1.0]],
#                                      [[0.8, 0.1, 0.1],
#                                       [0.0, 0.0, 1.0]]])
# rewards = np.array([[[1, 0, 0],
#                      [2, 0, 0]],
#                     [[0, 0, 0],
#                      [0, 0, 1]],
#                     [[-1, 0, 0],
#                      [0, 0, -1]]])
# mdp = MarkovDecisionProcess(num_states, num_actions, transition_probabilities, rewards)
# optimal_values, optimal_policy = mdp.value_iteration()
# print("Optimal values:", optimal_values)
# print("Optimal policy:", optimal_policy)
