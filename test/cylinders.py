import swift
import numpy as np
from spatialmath import SE3
from roboticstoolbox import jtraj
from XArm6 import XArm6
from spatialgeometry import Cylinder, Cuboid
from math import pi

# -----------------------------
# Launch Swift
env = swift.Swift()
env.launch(realtime=True)

# -----------------------------
# Add robot at origin
robot = XArm6()
robot.base = SE3(0, 0, 0)
env.add(robot)

# -----------------------------
fence_center = SE3(0.5, 0, 0)
fence_height = 0.08
fence_thickness = 0.01
box_size = 0.2  # inner box width/length

# Front and back walls
front_wall = Cuboid([box_size, fence_thickness, fence_height], pose=fence_center * SE3(0, box_size/2, fence_height/2))
back_wall = Cuboid([box_size, fence_thickness, fence_height], pose=fence_center * SE3(0, -box_size/2, fence_height/2))
# Left and right walls
left_wall = Cuboid([fence_thickness, box_size, fence_height], pose=fence_center * SE3(-box_size/2, 0, fence_height/2))
right_wall = Cuboid([fence_thickness, box_size, fence_height], pose=fence_center * SE3(box_size/2, 0, fence_height/2))

for wall in [front_wall, back_wall, left_wall, right_wall]:
    env.add(wall)
# Burger cylinders scattered
radius = 0.05
height = 0.02
colors = [[1,0.9,0.5], [0.5,0.2,0.2], [1,0.9,0.5]]  # bottom bun, beef, top bun

scatter_positions = [
    SE3(0.3, 0.2, 0.0),
    SE3(0.25, -0.2, 0.0),
    SE3(0.2, 0.3, 0.0)
]

cylinders = []
for i in range(3):
    cyl = Cylinder(radius=radius, length=height, color=colors[i])
    cyl.T = scatter_positions[i]
    env.add(cyl)
    cylinders.append(cyl)

# -----------------------------
# Robot default joint position
q_default = np.zeros(6)
robot.q = q_default

# -----------------------------
# Burger stack target
target_pos = np.array([0, 0.5, 0])  # bottom bun target
stack_height = 0.0

for i, cyl in enumerate(cylinders):
    # Move above cylinder (EE facing down)
    above_cyl = SE3(cyl.T) * SE3(0,0,height + 0.01) * SE3.Rx(pi)
    sol = robot.ikine_LM(above_cyl, q0=robot.q)
    traj = jtraj(robot.q, sol.q, 50)
    for q in traj.q:
        robot.q = q
        env.step(0.02)

    # Move down to cylinder (EE facing down)
    pick_pose = SE3(cyl.T) * SE3.Rx(pi)
    sol = robot.ikine_LM(pick_pose, q0=robot.q)
    traj = jtraj(robot.q, sol.q, 30)
    for q in traj.q:
        robot.q = q
        env.step(0.02)

    # Pick cylinder (attach to EE)
    cyl.T = robot.fkine(robot.q) @ SE3(0,0,height/2)

    # Move above stack target (EE facing down)
    place_pose = SE3(target_pos[0], target_pos[1], target_pos[2] + stack_height + height/2) * SE3.Rx(pi)
    sol = robot.ikine_LM(place_pose, q0=robot.q)
    traj = jtraj(robot.q, sol.q, 50)
    for q in traj.q:
        robot.q = q
        cyl.T = robot.fkine(q) @ SE3(0,0,height/2)
        env.step(0.02)

    # Update stack height for next cylinder
    stack_height += height

# -----------------------------
# -----------------------------
# Move the entire burger to 0.5, 0, 0.5 (EE facing down)
# Compute the center of the stacked burger
burger_center = SE3(target_pos[0], target_pos[1], 0 + stack_height/2)  # current center
move_target = SE3(0.5, 0, 0.5) * SE3.Rx(pi)  # target center, EE facing down

sol = robot.ikine_LM(move_target, q0=robot.q)
traj = jtraj(robot.q, sol.q, 100)
for q in traj.q:
    robot.q = q
    # Update all cylinders relative to EE
    for i, cyl in enumerate(cylinders):
        cyl.T = robot.fkine(q) @ SE3(0,0,(i+0.5)*height)
    env.step(0.02)

# -----------------------------
# -----------------------------
# RMRC straight down to put burger in the box
import time

# Current end-effector pose
T_current = robot.fkine(robot.q)
z_start = T_current.t[2]
z_end = fence_height/2 + stack_height/2  # final Z inside the box

steps_rmrc = 50
for i in range(steps_rmrc):
    z = z_start + (z_end - z_start) * (i+1)/steps_rmrc
    T_des = SE3(T_current.t[0], T_current.t[1], z) * SE3.Rx(pi)
    sol = robot.ikine_LM(T_des, q0=robot.q)
    robot.q = sol.q
    # Move all cylinders with EE
    for j, cyl in enumerate(cylinders):
        cyl.T = robot.fkine(robot.q) @ SE3(0,0,(j+0.5)*height)
    env.step(0.02)

# Optional: "release" burger by not parenting to EE (they stay in box)
# In this simple simulation, you just stop updating their pose

# -----------------------------
# Return to default position
traj = jtraj(robot.q, q_default, 50)
for q in traj.q:
    robot.q = q
    env.step(0.02)

env.hold()

