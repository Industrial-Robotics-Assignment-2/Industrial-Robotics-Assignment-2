import swift
import os
from spatialmath import SE3
from XArm6 import XArm6
from C4A601S import C4A601S
from CytonGamma300DH import CytonGamma300
from ir_support import LinearUR3


from spatialgeometry import Cuboid, Cylinder, Mesh
from math import pi

class EnvBuilder:
    def __init__(self):
        # ---------------- Launch environment ----------------
        self.env = swift.Swift()
        self.env.launch(realtime=True)

        # ---------------- Add robots ----------------
        self.robot = XArm6()
        self.robot.base = SE3(0, 0, 0.5)  # hardcoded height
        self.env.add(self.robot)
        self.robot.q = (0, pi, 0, 0, 0, 0)  # default pose


        self.frybot = C4A601S()
        self.frybot.base = SE3(0, 2, 0.5)
        self.frybot.add_to_env(self.env)   # use add_to_env, not env.add()
        self.frybot.q = (0,0,0,0,0,0)

        self.linear_ur3 = LinearUR3()
        self.linear_ur3.base = SE3(-1, 0, 0.5) * SE3.Rx(pi/2)  # rotated upright
        self.linear_ur3.add_to_env(self.env)
        self.linear_ur3.q = (0, 0, 0, 0, 0, 0, 0)

        self.CytonGamma300 = CytonGamma300()
        self.CytonGamma300.base = SE3(0, -2, 0.5)
        self.env.add(self.CytonGamma300)
        self.CytonGamma300.q = (0,0,0,0,0,0,0)



        # ---------------- Build table ----------------
        self._add_table()

        # ---------------- Build burger box ----------------
        self._add_box()

        # ---------------- Add burger patties ----------------
        self._add_burger()

        # ---------------- Add STL objects ----------------
        self._add_stl_objects()
        # ---------------- Add dae objects ----------------
        self._add_dae_objects()

    # ---------------- Table ----------------
    def _add_table(self):
        leg_height = 0.47
        leg_thickness = 0.05
        tabletop_size = [1.2, 1.2, 0.02]

        self.table_legs = []
        self.table_tops = []

        positions = [(0,0), (0,2), (0,-2), (-1.2,0), (-1.2, 1), (-1.2, -1), (-1.2, 2), (-1.2, -2)]
        for x, y in positions:
            leg = Cuboid([leg_thickness, leg_thickness, leg_height],
                         pose=SE3(x, y, leg_height/2),
                         color=[0.6,0.3,0.1])
            top = Cuboid([tabletop_size[0], tabletop_size[1], tabletop_size[2]],
                         pose=SE3(x, y, leg_height + tabletop_size[2]/2),
                         color=[0.6,0.3,0.1])
            self.env.add(leg)
            self.env.add(top)
            self.table_legs.append(leg)
            self.table_tops.append(top)

        # Floor under table
        self.floor = Cuboid([5,6,0.01], pose=SE3(-1.5,0,0), color=[0.7,0.7,0.7])
        self.env.add(self.floor)
        # Back wall behind robot
        wall_thickness = 0.05
        wall_height = 2.0
        wall_width = 6.0
        self.back_wall_robot = Cuboid([wall_thickness, wall_width, wall_height],
                                    pose=SE3(1.0 + wall_thickness/2, 0, wall_height/2),
                                    color=[0.7, 0.7, 0.7])
        self.env.add(self.back_wall_robot)

    # ---------------- Box ----------------
    def _add_box(self):
        self.fence_height = 0.08
        self.fence_thickness = 0.01
        self.box_size = 0.2
        self.fence_center = SE3(0.5, 0, 0.5)  # hardcoded height

        # Floor
        self.box_floor = Cuboid([self.box_size, self.box_size, self.fence_thickness],
                                pose=self.fence_center * SE3(0,0,self.fence_thickness/2),
                                color=[1,0,0])
        self.env.add(self.box_floor)

        # Walls
        self.front_wall = Cuboid([self.box_size, self.fence_thickness, self.fence_height],
                                 pose=self.fence_center * SE3(0, self.box_size/2, self.fence_height/2),
                                 color=[1,0,0])
        self.back_wall = Cuboid([self.box_size, self.fence_thickness, self.fence_height],
                                pose=self.fence_center * SE3(0, -self.box_size/2, self.fence_height/2),
                                color=[1,0,0])
        self.left_wall = Cuboid([self.fence_thickness, self.box_size, self.fence_height],
                                pose=self.fence_center * SE3(-self.box_size/2, 0, self.fence_height/2),
                                color=[1,0,0])
        self.right_wall = Cuboid([self.fence_thickness, self.box_size, self.fence_height],
                                 pose=self.fence_center * SE3(self.box_size/2, 0, self.fence_height/2),
                                 color=[1,0,0])
        for wall in [self.front_wall, self.back_wall, self.left_wall, self.right_wall]:
            self.env.add(wall)

        # Lid
        self.lid = Cuboid([self.box_size, self.box_size, self.fence_thickness],
                          color=[1,0,0])
        self.lid_home_pose = SE3(0.3, -0.4, 0.5)  # hardcoded
        self.lid.T = self.lid_home_pose
        self.env.add(self.lid)

    # ---------------- Burger ----------------
    def _add_burger(self):
        self.radius = 0.05
        self.height = 0.02
        self.colors = [[1,0.9,0.5], [0.5,0.2,0.2], [1,0.9,0.5]]
        self.scatter_positions = [SE3(0.35,0.4,0.5), SE3(0.25,0.4,0.5), SE3(0.15,0.4,0.5)]
        self.cylinders = []

        for i in range(3):
            cyl = Cylinder(radius=self.radius, length=self.height, color=self.colors[i])
            cyl.T = self.scatter_positions[i]
            self.env.add(cyl)
            self.cylinders.append(cyl)

    # ---------------- STL Objects ----------------
    def _add_stl_objects(self):
        clscale = 0.03
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stl_path = os.path.join(current_dir, "chainlink.stl")
        self.chainlinks = []
        for y in [1, -1, 3, -3]:
            cl = Mesh(stl_path, scale=[clscale]*3, color=[1,0,0])
            cl.T = SE3(2, y, -0.2)
            self.env.add(cl)
            self.chainlinks.append(cl)

        # Example worker STL
        worker_path = os.path.join(current_dir, "worker.stl")
        self.worker = Mesh(worker_path, scale=[clscale]*3, color=[1,1,0])
        self.worker.T = SE3(-2,2,0) * SE3.Rz(-pi/4)
        self.env.add(self.worker)

        # Example button STL
        bscale = 0.003
        button_base_path = os.path.join(current_dir, "buttonbase.stl")
        self.button_base = Mesh(button_base_path, scale=[bscale]*3, color=[1,1,0])
        self.button_base.T = SE3(-2,1,0.05) * SE3.Rx(-pi/2)
        self.env.add(self.button_base)

        button_path = os.path.join(current_dir, "button.stl")
        self.button = Mesh(button_path, scale=[bscale]*3, color=[1,0,0])
        self.button.T = SE3(-2,1,0.1) * SE3.Rx(-pi/2)
        self.env.add(self.button)
    
    # ---------------- DAE Objects ----------------
    def _add_dae_objects(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Fries pile
        fries_path = os.path.join(current_dir, "friespile.dae")
        self.fries_pile = Mesh(filename=fries_path, pose=SE3(-0.15, -0.4+2, 0.5))
        self.env.add(self.fries_pile)

        # Fry box construct
        frybox_path = os.path.join(current_dir, "FryboxConstruct.dae")
        self.fry_box = Mesh(filename=frybox_path, pose=SE3(-0.2, -0.2+2, 0.5))
        self.env.add(self.fry_box)
        self.fry_box_attached_offset = SE3(-0.075, 0, 0.02) * SE3.Ry(pi/2)

        # Full fry box
        frybox_full_path = os.path.join(current_dir, "FryboxFull.dae")
        self.fry_box_full = Mesh(filename=frybox_full_path, pose=SE3(100, 100, 100))
        self.env.add(self.fry_box_full)
