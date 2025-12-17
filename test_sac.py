import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("BipedalWalker-v3", render_mode="human")
model = SAC.load("sac_bipedal")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()