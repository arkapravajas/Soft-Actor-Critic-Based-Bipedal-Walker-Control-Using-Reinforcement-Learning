import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("BipedalWalker-v3")  # NO RENDERING

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=300_000,
    learning_rate=3e-4,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto"
)

model.learn(total_timesteps=500_000)
model.save("sac_bipedal")

env.close()