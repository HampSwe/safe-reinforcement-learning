import yaml
import environment
from rsl_rl.runners import OnPolicyRunner

device = "cuda:0" #"cpu"

def make_env():
    if device == "cpu":
        return environment.VecDrone2D(num_envs=64, device=device)
    else:
        return environment.VecDrone2D(num_envs=4096, device=device)

def main():
    # 1) Create your environment (usually provided by environment libraries such as Isaac Lab)
    env = make_env()

    # 2) Load the YAML configuration used during training
    with open("config/config1.yaml", "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    train_cfg = full_cfg["runner"]

    # 3) Build the runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        device="cpu",
    )

    # 4) Load the trained policy
    runner.load("logs/my_experiment/model_1499.pt")

    # 5) Get the inference policy
    policy = runner.get_inference_policy()

    # 6) Run the policy in the environment for 1000 steps
    obs = env.get_observations()
    for _ in range(1000):
        actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions)

if __name__ == "__main__":
    main()