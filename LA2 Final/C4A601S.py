
import swift
import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os

# Useful variables
from math import pi

# -----------------------------------------------------------------------------------#
class C4A601S(DHRobot3D):
    def __init__(self):
        """
            UR3 Robot by DHRobot3D class

            Example usage:
            >>> from ir-support import UR3
            >>> import swift

            >>> r = UR3()
            >>> q = [0,-pi/2,pi/4,0,0,0]r
            >>> r.q = q
            >>> q_goal = [r.q[i]-pi/4 for i in range(r.n)]
            >>> env = swift.Swift()
            >>> env.launch(realtime= True)
            >>> r.add_to_env(env)
            >>> qtraj = rtb.jtraj(r.q, q_goal, 50).q
            >>> for q in qtraj:r
            >>>    r.q = q
            >>>    env.step(0.02)
        """
        # DH links
        links = self._create_DH()

        # Names of the robot link files in the directory
        link3D_names = dict(link0 = 'C4-A601S_Base',
                            link1 = 'C4-A601S_L1',
                            link2 = 'C4-A601S_L2',
                            link3 = 'C4-A601S_L3',
                            link4 = 'C4-A601S_L4',
                            link5 = 'C4-A601S_L5',
                            link6 = 'C4-A601S_L6'
                            )

        # A joint config and the 3D object transforms to match that config
        qtest = [0,0,0,0,0,0]
        qtest_transforms = [spb.transl(0,0,0),
                            spb.transl(0,0,0),
                            spb.transl(0,0,0),
                            spb.transl(0,0,0),
                            spb.transl(0,0,0),
                            spb.transl(0,0,0),
                            spb.transl(0,0,0)
                            ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(links, link3D_names, name = 'C4-A601S', link3d_dir = current_path, qtest = qtest, qtest_transforms = qtest_transforms)
        self.q = qtest





    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        a = [-0.1, 0.250, 0, 0, 0, 0]
        d = [0.320, 0, 0, 0.250, 0, 0.07]
        alpha = [pi/2, 0, pi/2, pi/2, -pi/2, 0]
        offset = [0, pi/2, pi, pi, 0, 0]
        qlim = [
            [np.deg2rad(-170), np.deg2rad(170)],   # L1
            [np.deg2rad(-65), np.deg2rad(160)],    # L2
            [np.deg2rad(-225), np.deg2rad(51)],    # L3
            [np.deg2rad(-200), np.deg2rad(200)],   # L4
            [np.deg2rad(-135), np.deg2rad(135)],   # L5
            [np.deg2rad(-360), np.deg2rad(360)]    # L6
        ]
        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i], qlim= qlim[i])
            links.append(link)
        return links
    

    # -----------------------------------------------------------------------------------#
    def test(self):
        """
        Test the class by adding 3d objects into a new Swift window and do a simple movement
        """
        env = swift.Swift()
        env.launch(realtime= True)
        self.q = self._qtest
        self.base = SE3(0.5,0.5,0)
        self.add_to_env(env)

        q_goal = [self.q[i]-pi/3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        #fig = self.plot(self.q)
        for q in qtraj:
            self.q = q
            env.step(0.02)
            #fig.step(0.01)
        time.sleep(3)
        env.hold()

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = C4A601S()
    r.test()

