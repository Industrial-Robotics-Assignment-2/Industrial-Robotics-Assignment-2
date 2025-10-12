import swift
from spatialmath import SE3
import numpy as np
from math import pi
from roboticstoolbox import trapezoidal
from XArm6 import XArm6


env = swift.Swift()  # Start the simulator
env.launch(realtime=True)  # Launch the 3D visualizer in your browser

robot = XArm6()
env.add(robot)

t = 10
delta_t = 0.02
steps = int(t / delta_t)
delta = 2 * pi / steps

s = trapezoidal(0, 1, steps).q

x = np.zeros([3, steps])
theta = np.zeros([3, steps])

for i in range(steps):
    x[0, i] = 0.35                          # X fixed
    x[1, i] = (1 - s[i]) * -0.55 + s[i] * 0.55   # Y line
    x[2, i] = 0.5 + 0.2 * np.sin(i * delta)      # Z oscillation
    theta[0, i] = 0
    theta[1, i] = 5 * pi / 9
    theta[2, i] = 0

q_traj = np.zeros([steps, 6])
q_guess = np.zeros(6)

for i in range(steps):
    T = SE3(x[:, i]) * SE3.RPY(theta[:, i], order='xyz')
    sol = robot.ikine_LM(T, q_guess)
    q_traj[i, :] = sol.q
    q_guess = sol.q  # use last solution as initial guess

for i in range(steps):
    robot.q = q_traj[i, :]
    env.step(0.02)  # update every 20 ms