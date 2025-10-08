# XArm6_simple.py
import swift
from roboticstoolbox import DHLink, DHRobot
from roboticstoolbox import jtraj

from ir_support import CylindricalDHRobotPlot
from math import pi
import time

# -----------------------------------------------------------------------------------#
class XArm6(DHRobot):
    def __init__(self):
        """
        Simplified UFactory XArm6 using DHRobot with cylinders.
        """
        # DH parameters (standard)
        a = [0, 0.324, 0.325, 0, 0, 0]
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
        d = [0.267, 0, 0, 0.111, 0.113, 0.097]
        qlim = [[-2*pi, 2*pi]] * 6

        # Create DH links
        links = [DHLink(a=a[i], alpha=alpha[i], d=d[i], qlim=qlim[i]) for i in range(6)]

        # Initialize DHRobot
        super().__init__(links, name='XArm6')
        # Add cylindrical visualization
        cyl_viz = CylindricalDHRobotPlot(self, cylinder_radius=0.03, color="#3478f6")
        cyl_viz.create_cylinders()

        # Initial joint configuration
        self.q = [0, 0, 0, 0, 0, 0]

# -----------------------------------------------------------------------------------#
if __name__ == "__main__":
    # Create robot
    robot = XArm6()

    # Launch Swift environment
    env = swift.Swift()
    env.launch(realtime=True)

    # Add robot to environment
    env.add(robot)

    env.step()


    # ----------------------------
    # Example joint-space trajectory
    q_start = robot.q
    q_goal = [0, -pi/4, pi/3, 0, pi/6, 0]
    q_traj = jtraj(q_start, q_goal, 50).q

    # Animate trajectory
    for q in q_traj:
        robot.q = q
        env.step(0.02)

    # Hold final position
    env.hold()
    time.sleep(2)
