from setuptools import setup

setup(name='gym_uav_collision_avoidance',
    version='0.0.1',
    install_requires=['gym==0.24.1', 'pygame==2.1.2', 'stable-baselines3[extra]', 'tensorflow-gpu==2.5.0']
)