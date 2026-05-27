import gymnasium

from env import ApartmentEnv

T = 4
K = 4

gymnasium.register(
    id="CS5180/PS0_ApartmentEnv",
    entry_point=ApartmentEnv,
)

env = gymnasium.make("CS5180/PS0_ApartmentEnv", T=T, K=K)

obs, _ = env.reset()

print(f"Initial State: (t={obs["t"]}, U_t={obs["U_t"]})")


for _ in range(T + 1):
    action = env.action_space.sample()

    obs, rew, term, _, _ = env.step(action)

    print(
        f"(t={obs["t"]}, U_t={obs["U_t"]}, action={"accept" if action == 0 else "reject"}, reward={rew}, done={term})"
    )
