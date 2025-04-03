import pickle

config_dict = {
    "observation": {
        "type": "Kinematics",  # Needed for continuous control
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "absolute": False,
    },
    "action": {
        "type": "ContinuousAction",  # Key change for Task 2
    },
    "lanes_count": 4,
    "vehicles_count": 15,
    "controlled_vehicles": 1,
    "initial_spacing": 0,
    "duration": 60,
    "collision_reward": -1,
    "right_lane_reward": 0.5,
    "high_speed_reward": 0.1,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": True,
}

with open("highway_continuous_config.pkl", "wb") as f:
    pickle.dump(config_dict, f)
