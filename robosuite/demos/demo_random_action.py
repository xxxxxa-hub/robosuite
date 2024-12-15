from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers.gym_wrapper import GymWrapper, SingleArmGymWrapper
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
import copy



if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # Choose controller
    controller_name = choose_controller()

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
        reward_shaping=True,
        horizon=500,
    )

    eval_env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
        reward_shaping=True,
        horizon=500,
    )
    
    env = SingleArmGymWrapper(env)
    eval_env = SingleArmGymWrapper(eval_env)
    env.viewer.set_camera(camera_id=0)
    eval_env.viewer.set_camera(camera_id=0)
    env = Monitor(env,"train_monitor/")
    eval_env = Monitor(eval_env,"eval_monitor/")

    env.reset()
    eval_env.reset()
    

    # eval_callback = EvalCallback(eval_env, best_model_save_path="models/",
    #                           log_path="eval_logs/", eval_freq=5000,
    #                           n_eval_episodes=10, deterministic=True,
    #                           render=False)
    
    # Get action limits
    low = env.action_space.low
    high = env.action_space.high

    # Train model
    model = TD3("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    model.learn(total_timesteps=3e5, log_interval=1)
    model.save("td3_door")
    
    # Load model
    # model = TD3.load("td3_door", env=env)

    # Evaluate
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print("Mean reward: ", mean_reward)
    # print("Std reward: ", std_reward)
                  
    # Visualize
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(2000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     vec_env.envs[0].env.render()

    # do visualization
    # for i in range(10000):
    #     action = np.random.uniform(low, high)
    #     obs, reward, terminated, truncated, _ = env.step(action)
    #     env.render()
