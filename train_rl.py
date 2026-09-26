import numpy as np
import joblib
from pricing_environment import PricingEnvironment


env = PricingEnvironment()

best_action = None
best_reward = -np.inf

for action in range(3):

    env.reset()

    _, reward, _, _, _ = env.step(action)

    if reward > best_reward:
        best_reward = reward
        best_action = action


if best_action == 0:
    decision = "Decrease Price"
elif best_action == 1:
    decision = "Keep Price"
else:
    decision = "Increase Price"


rl_result = {
    "best_action": best_action,
    "decision": decision,
    "expected_reward": best_reward
}

print("Reinforcement Learning pricing system trained successfully!")
print("Best pricing action:", decision)
print("Expected reward:", round(best_reward, 2))

joblib.dump(
    rl_result,
    "reinforcement_learning/pricing_rl_model.pkl"
)

print("RL decision model saved successfully!")