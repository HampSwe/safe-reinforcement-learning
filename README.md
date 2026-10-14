# Anytime safe reinforcement learning (RL-SGF) implementation

This repository contains an implementation of an anytime safe reinforcement learning algorithm ([RL-SGF](https://arxiv.org/abs/2504.16417) by Mestres et al.) with modifications (PPO-style estimates, learning rate scheduling) for a planar quadrotor task. We also provide a simulator/environment of the quadrotor.

We have also implemented this in a realistic 3D simulation in Isaac Lab (but in a different repository). Furthermore, we performed system identification on a Crazyflie, and managed to overcome the sim-to-real gap by deploying a trained policy on hardware.

In a separate repository, we also train RL-SGF for a G1 Humanoid in mjlab.