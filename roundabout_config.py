import pickle

config_dict = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "absolute": False,
    },
    "action": {
        "type": "ContinuousAction",
    },
    "lanes_count": 1,
    "vehicles_count": 8,
    "controlled_vehicles": 1,
    "initial_spacing": 2.0,
    "duration": 40,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 5.5,
    "render_agent": True,
    "show_trajectories": True,
    "offscreen_rendering": False,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "collision_reward": -5,
    "reward_speed_range": [0, 15],
    "high_speed_reward": 0.5,
    "lane_change_reward": -0.1,
    "right_lane_reward": 0.0
}

with open("roundabout_config.pkl", "wb") as f:
    pickle.dump(config_dict, f)
