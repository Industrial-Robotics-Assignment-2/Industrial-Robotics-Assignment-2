import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import time
import swift
import keyboard
from spatialmath import SE3
from spatialmath.base import *
from spatialgeometry import Cuboid, Mesh
from roboticstoolbox import jtraj, DHRobot, DHLink, models, trapezoidal
from ir_support import LinearUR3, CylindricalDHRobotPlot
import os
from math import pi


# Define robot
L1 = DHLink(d=0.112, a=-0.100, alpha=pi/2, offset=0, qlim=[np.deg2rad(-170), np.deg2rad(170)])
L2 = DHLink(d=0, a=0.250, alpha=0, offset=pi/2, qlim=[np.deg2rad(-65), np.deg2rad(160)])
L3 = DHLink(d=0, a=0.060, alpha=pi/2, offset=pi/2, qlim=[np.deg2rad(-225), np.deg2rad(51)])
L3_to_L4 = DHLink(d=0, a=0, alpha=pi/2, offset=pi/2, qlim = [0, 0])  # fixed reorientation link 
L4 = DHLink(d=0.190, a=0, alpha=pi/2, offset=pi/2, qlim=[np.deg2rad(-200), np.deg2rad(200)])
L5 = DHLink(d=0, a=0.065, alpha=0, offset=pi/2, qlim=[np.deg2rad(-135), np.deg2rad(135)])
L5_to_L6 = DHLink(d=0, a=0, alpha=-pi/2, offset=-pi/2, qlim = [0, 0])  # fixed reorientation link
L6 = DHLink(d=0.005, a=0, alpha=0, offset=-pi/2, qlim=[np.deg2rad(-360), np.deg2rad(360)])

robot = DHRobot([L1, L2, L3,L3_to_L4, L4, L5, L5_to_L6, L6], name='Frybot')


# Go through fries trajectory
T1 = SE3(-0.3, -0.4, 0) * SE3.Rx(np.pi)
T2 = SE3(0, -0.4, 0) * SE3.Rx(np.pi)
T3 = SE3(0.1, -0.4, 0.1) * SE3.Rx(np.pi) * SE3.Ry(-np.pi/2)

# Plot robot at zero configuration
robot.qz = np.zeros(robot.n)
q = robot.qz
# robot.plot(q, block=False)

fig = robot.plot(q, jointaxes=False)



x1 = np.array([-0.3, -0.4, 0])
x2 = np.array([0, -0.4, 0])
x3 = np.array([0.1, -0.4, 0.1])
delta_t1 = 0.05                                                                  # Discrete time step
min_manip_measure1 = 0.1                                                         # Threshold for Measure of Manipulability (MoM)
steps1 = 100                                                                     # No. of steps in trajectory
x_path1 = np.empty([3, steps1])                                                        # Assign memory for trajectory
m1 = np.zeros([1,steps1])                                                         # Assign memory to store Measure of Manipulability (MoM)
error1 = np.empty([6,steps1])
mask1 = [1,1,1,1,1,1]                                   # Discrete time step
s1 = trapezoidal(0,1,steps1).q

for i in range(steps1):
    x_path1[:,i] = x1 *(1-s1[i]) + s1[i]*x2                 # Create trajectory in x-y plane


q_matrix1 = np.zeros([steps1,8])
q_matrix1[0,:] = robot.ikine_LM(T1, q0=np.zeros(robot.n), mask = mask1).q    # Solve for joint angles


for i in range(steps1-1):
    # Linear x-y velocity, orientation fixed
    xdot = np.zeros(6)
    xdot[0:3] = (x_path1[:, i+1] - x_path1[:, i]) / delta_t1  # x, y
    xdot[3:6] = 0                                 # angular velocity

    J = robot.jacob0(q_matrix1[i])               # full 6x8 Jacobian
    m1[:, i] = np.sqrt(linalg.det(J[:3,:] @ J[:3,:].T))

    # Damped Least Squares
    if m1[:, i] < min_manip_measure1:
        qdot = linalg.pinv(J.T @ J + 0.01 * np.eye(J.shape[1])) @ J.T @ xdot
    else:
        qdot = linalg.pinv(J) @ xdot

    error1[:,i] = xdot - J@qdot
    # Update next joint angles
    q_matrix1[i+1, :] = q_matrix1[i, :] + delta_t1 * qdot

for q in q_matrix1:
    robot.q = q
    ee_position = robot.fkine(q).A[:3,3]
    fig.ax.plot(ee_position[0], ee_position[1], ee_position[2], 'b.')
    fig.step(0.05)

# Scooping move variables
delta_t2 = 0.05
min_manip_measure2 = 0.1
steps2 = 100
delta_theta = np.pi/2 / steps2
theta = np.zeros([3, steps2])  # EE orientation (roll, pitch, yaw)
m2 = np.zeros([1, steps2])
error2 = np.empty([6, steps2])
mask2 = [1,1,1,1,1,1]  # translation + rotation
x_path2 = np.zeros([3, steps2])

# 2.2: Create an arbitrary cartesian trajectory for the robot end-effector to follow
for i in range(steps2):
    theta_i = delta_theta * i
    x_path2[:, i] = [0.1 * np.sin(theta_i), -0.4, 0.1 * (1 - np.cos(theta_i))]  # keep Z constant for horizontal circle
    theta[:, i] = [0, -theta_i, 0]  # roll=0, pitch=0, yaw=angle


# Scoop Fries up trajectory
q_matrix2 = np.zeros([steps2, robot.n])
# q_matrix2[0,:] = robot.ikine_LM(SE3(x_path2[:,0]) @ SE3.Rx(np.pi) @ SE3.Ry(theta[1,0]), q0=np.zeros(robot.n), mask=mask2).q
q_matrix2[0,:] = q_matrix1[-1, :]  # start at end of previous trajectory

# 2.4: Use Resolved Motion Rate Control (RMRC) to solve joint velocities at each time step,
# such that end-effector follows the defined Cartesian trajectory
for i in range(steps2-1):
    xdot = np.zeros(6)
    xdot[0:3] = (x_path2[:, i+1] - x_path2[:, i]) / delta_t2
    xdot[3:6] = (theta[:, i+1] - theta[:, i]) / delta_t2
    J = robot.jacob0(q_matrix2[i])                                                  # Get the Jacobian at the current state
    m2[:,i] = np.sqrt(linalg.det(J @ J.T))                                       # Measure of Manipulability

    # Apply Damped Least Squares for configurations with low manipulability
    if m2[:,i] < min_manip_measure2:
        qdot = linalg.pinv(J.T @ J + 0.01 * np.eye(J.shape[1])) @ J.T @ xdot
    else:
        qdot = linalg.pinv(J) @ xdot                                             # Solve velocitities via RMRC

    error2[:,i] = xdot - J@qdot                                                  # Calculate and store velocity error
    q_matrix2[i+1, :] = q_matrix2[i, :] + delta_t2 * qdot                               # Update next joint state in trajectory

# 2.5: Plot trajectory and velocity error
# fig = robot.plot(q_matrix2[0], limits = [-2,2,-2,2,-2,2])       # Plot robot in first joint state
# Loop through joint matrix, for each step in trajectory, update robot joint states, get X,Y position of end-effector, plot red marker
for q in q_matrix2:
    robot.q = q
    pos = robot.fkine(q).A[:3,3]
    fig.ax.plot(pos[0], pos[1], pos[2], 'r.', markersize = 5)
    fig.step(0.05)

input("Press Enter to exit...") 
