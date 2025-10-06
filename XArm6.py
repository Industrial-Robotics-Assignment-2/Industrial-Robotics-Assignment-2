##  @file
#   @brief UFactory XArm6 Robot with DHRobot3D using cylinder STLs
#   @author Jack
#   @date October 6, 2025

import os
import time
from math import pi

import trimesh
import swift
from roboticstoolbox import DHLink
from ir_support.robots.DHRobot3D import DHRobot3D
from spatialmath import SE3
import spatialmath.base as spb

# -----------------------------------------------------------------------------------#
def create_cylinder_stl(folder, name, length, radius=0.03):
    """
    Create a simple cylinder STL file for a robot link.
    Cylinder points along Z, base at z=0
    """
    # Create cylinder mesh
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=32)
    # Move so base is at origin
    cylinder.apply_translation([0, 0, length / 2])
    stl_path = os.path.join(folder, f"{name}.stl")
    cylinder.export(stl_path)
    print(f"Saved {stl_path}")


# -----------------------------------------------------------------------------------#
class XArm6(DHRobot3D):
    def __init__(self):
        """
        UFactory XArm6 Robot using DHRobot3D with cylinders
        """
        # DH parameters (standard)
        a = [0, 0.324, 0.325, 0, 0, 0]
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
        d = [0.267, 0, 0, 0.111, 0.113, 0.097]
        qlim = [[-2*pi, 2*pi]] * 6

        # Create DH links
        links = [DHLink(a=a[i], alpha=alpha[i], d=d[i], qlim=qlim[i]) for i in range(6)]

        # Folder for STLs
        current_path = os.path.abspath(os.path.dirname(__file__))

        # Cylinder names and lengths
        link_lengths = {
            "link0": d[0],
            "link1": a[1],
            "link2": a[2],
            "link3": d[3],
            "link4": d[4],
            "link5": d[5],
            "link6": 0.05  # small end-effector
        }

        # Generate STL files for cylinders if they don't exist
        for name, length in link_lengths.items():
            stl_file = os.path.join(current_path, f"{name}.stl")
            if not os.path.exists(stl_file):
                create_cylinder_stl(current_path, name, length)

        # Names of the robot link files
        link3D_names = {name: name for name in link_lengths.keys()}

        # Home configuration
        qtest = [0, 0, 0, 0, 0, 0]

        # Approximate transforms for visualization
        qtest_transforms = [
            spb.transl(0, 0, 0),
            spb.transl(0, 0, 0.267),
            spb.transl(0.324, 0, 0.267),
            spb.transl(0.649, 0, 0.267),
            spb.transl(0.649, 0, 0.378),
            spb.transl(0.649, 0, 0.491),
            spb.transl(0.649, 0, 0.593)
        ]

        # Initialize DHRobot3D
        super().__init__(
            links,
            link3D_names,
            name='XArm6',
            link3d_dir=current_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )

        # Optional: rotate base so Z points down
        self.base = self.base * SE3.Rx(pi/2) * SE3.Ry(pi/2)
        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def test(self):
        """
        Launch Swift and run a simple joint-space trajectory
        """
        env = swift.Swift()
        env.launch(realtime=True)
        env.add(self)
        env.step()

        q_start = robot.q
        q_goal = [0, -pi/4, pi/3, 0, pi/6, 0]  # your desired joint state

        # Generate joint-space trajectory (50 steps)
        q_traj = robot.jtraj(q_start, q_goal, 50).q

        # Animate in Swift
        for q in q_traj:
            robot.q = q
            env.step(0.02)

        # Hold final position
        env.hold()
        time.sleep(2)


# -----------------------------------------------------------------------------------#
if __name__ == "__main__":
    robot = XArm6()
    input("Press Enter to test XArm6 in Swift...")
    robot.test()
