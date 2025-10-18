import numpy as np
from math import pi
from roboticstoolbox import DHRobot, DHLink
from spatialmath import SE3
import swift
from ir_support import CylindricalDHRobotPlot

class CytonGamma300(DHRobot):
        
        def __init__(self):  
            
            DH = [
                # i,   a_{i-1} (m), alpha_{i-1} (rad), d_i (m),         theta_offset (rad)
                (1,    0.0,          +pi/2,            0.116150,     +pi/2),   # A1
                (2,    0.0,          -pi/2,            0.0,          0.0),     # A2
                (3,    0.0,          +pi/2,            0.141000,     0.0),     # A3
                (4,   -0.072000,     +pi/2,            0.0,         -pi/2),    # A4
                (5,   -0.071800,     -pi/2,            0.0,          0.0),     # A5
                (6,    0.0,          -pi/2,            0.0,         +pi/2),    # A6
                (7,    0.0,           0.0,             0.051425,    -pi/2),    # A7
            ]

            links = []
            for (i, a_prev, alpha_prev, d_i, thoff) in DH:
                links.append(
                    DHLink(d=d_i, a=a_prev, alpha=alpha_prev, offset=thoff, name=f'L{i}')
                )

            super().__init__(links, name='CytonGamma300')
            
            cyl_viz = CylindricalDHRobotPlot(self, cylinder_radius=0.03, color="#dff634")
            cyl_viz.create_cylinders()

            self.q = np.zeros(7)

if __name__ == "__main__":

    robot = CytonGamma300()

    env = swift.Swift()
    env.launch(realtime=True)

    env.add(robot)

    robot.q = np.array([pi/2] * 7)

    env.step()
    env.hold()