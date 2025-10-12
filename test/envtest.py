import swift
import os
from spatialmath import SE3
import numpy as np
from math import pi
from roboticstoolbox import trapezoidal
from XArm6 import XArm6
from spatialmath.base import transl
from roboticstoolbox import DHRobot
from swift import Swift
from spatialgeometry import Mesh

# --------------------------------------------------------------------------
# Launch Swift environment
env = Swift()
env.launch(realtime=True)

# Add robot
robot = XArm6()
robot.base = SE3(0, 0, 0.7)  # X=0, Y=0, Z=1
env.add(robot)

# --------------------------------------------------------------------------
# Add and scale STLs
current_dir = os.path.dirname(os.path.abspath(__file__))
scale_factor = 0.01  # 1/100th size

# Paths
bottombun_path = os.path.join(current_dir, "BottomBun.stl")
beef_path = os.path.join(current_dir, "Beef.stl")
topbun_path = os.path.join(current_dir, "TopBun.stl")
grill_path = os.path.join(current_dir, "grill.stl")
table_path = os.path.join(current_dir, "table.stl")

# --------------------------------------------------------------------------
# Grill
grill = Mesh(filename=grill_path, color=[0.4, 0.4, 0.4], scale=[0.05]*3)
grill.T = SE3(0.75, -0.7, 0.0)
env.add(grill)

# Table
table = Mesh(filename=table_path, color=[0.6, 0.3, 0.1], scale=[0.008]*3)
table.T = SE3(0, 0.0, 0.0)
env.add(table)

# Burger parts (stacked)
bscale_factor = 0.005  # 1/100th size

bottom_bun = Mesh(filename=bottombun_path, color=[1, 0.9, 0.5], scale=[bscale_factor]*3)
bottom_bun.T = SE3(0.0, 0.5, 0.7)  # slightly above grill
env.add(bottom_bun)

beef = Mesh(filename=beef_path, color=[0.5, 0.2, 0.2], scale=[bscale_factor]*3)
beef.T = SE3(-1, -1, 0.0)  # slightly above bottom bun
env.add(beef)

top_bun = Mesh(filename=topbun_path, color=[1, 0.9, 0.5], scale=[bscale_factor]*3)
top_bun.T = SE3(0.0, 0.5, 0.7)  # on top of beef
env.add(top_bun)

# --------------------------------------------------------------------------
# Robot motion (existing)
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
    env.step(0.02)
env.hold()