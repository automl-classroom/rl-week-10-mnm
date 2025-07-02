import numpy as np
import dqn

# Beispiel Seeds
seeds = [0, 42, 100, 1234, 2025]

# Beispiel Hyperparameter-Raum
learning_rates = [1e-4, 5e-4, 1e-3]
epsilon_decays = [0.99, 0.995]

results = []

for lr in learning_rates:
    for ed in epsilon_decays:
        rewards = []
        for seed in seeds:
            reward = dqn.train_dqn_programmatic(seed=seed, learning_rate=lr, epsilon_decay=ed)
            rewards.append(reward)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        result_line = f"lr={lr}, ed={ed}, mean_reward={mean_reward:.2f}, std={std_reward:.2f}"
        print(result_line)
        results.append(result_line)

print("______")
print(results)

# Speichere Ergebnisse
with open("results.txt", "w") as f:
    for line in results:
        f.write(line + "\n")