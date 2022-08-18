from gym.envs.registration import register

register(
    id="gym_uav_collision_avoidance/UAVWorld2D-v0",
    entry_point="gym_uav_collision_avoidance.envs:UAVWorld2D",
)
register(
    id="gym_uav_collision_avoidance/MultiUAVWorld2D-v0",
    entry_point="gym_uav_collision_avoidance.envs:MultiUAVWorld2D",
)
# register(
#     id="gym_uav_collision_avoidance/MultiUAVWorld2DEvaluate-v0",
#     entry_point="gym_uav_collision_avoidance.envs:MultiUAVWorld2DEvaluate",
# )