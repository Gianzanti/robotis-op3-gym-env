from gymnasium.envs.registration import find_highest_version, register

from . import robotis

env_name = "Robotis"
env_version = 0
env = f"{env_name}-v{env_version}"

env_id = find_highest_version(ns=None, name=env_name)

if env_id is None:
    # Register this module as a gym environment. Once registered, the id is usable in gym.make().
    register(
        id=env,
        entry_point="robotis_op3.robotis:RobotisEnv",
    )
    print(f"Registered environment {env}")
else:
    print(f"Environment {env} already registered")

