import pickle

# Configuration for the Racetrack environment using continuous actions
racetrack_config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "absolute": False,
    },
    "action": {
        "type": "ContinuousAction",  # Enables continuous control (steering + acceleration)
    },
    "lanes_count": 4,
    "vehicles_count": 15,
    "controlled_vehicles": 1,
    "initial_spacing": 2,
    "duration": 50,
    "screen_width": 600,
    "screen_height": 600,
    "scaling": 5.5,
    "render_agent": True,
    "show_trajectories": True,
    "offscreen_rendering": False,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "collision_reward": -1,
    "reward_speed_range": [20, 30],
    "high_speed_reward": 1.0,
    "lane_change_reward": -0.05,
    "right_lane_reward": 0.0
}

with open("racetrack_config.pkl", "wb") as f:
    pickle.dump(racetrack_config, f)
