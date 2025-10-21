# LinearUR3Operator.py
import numpy as np
from spatialmath import SE3
from roboticstoolbox import jtraj
from math import pi
 
class LinearUR3Operator:
    def __init__(self, env_builder, estop_event):
        self.env = env_builder.env
        self.robot = env_builder.linear_ur3
        self.estop_event = estop_event
 
    def safe_step(self, dt=0.02):
        self.env.step(dt)
 
    def move_object_from_side(self, obj, target_xyzrpy, steps=100, side_offset=0.05):
        """
        Generic 3-step jtraj pick-and-place from the side.
        - obj: object with .T (Mesh/Cuboid). obj.T may be SE3 or 4x4 ndarray.
        - target_xyzrpy: (x,y,z, roll, pitch, yaw)
        - steps: number of steps for each jtraj phase
        - side_offset: approach distance from the side (meters)
        """
 
        # --- ensure object pose is SE3 ---
        obj_pose = SE3(obj.T)
 
        # --- default/home pose (start) ---
        q_home = np.zeros(self.robot.n)
        q_home[2] = -pi/2         # keep the 3rd joint as you used before
        # set robot to known start
        self.robot.q = q_home.copy()
 
        # --- fixed grasp orientation relative to end-effector so object stays upright ---
        # Adjust this if you want a different grasp rotation
        grasp_rotation = SE3.Ry(pi/2)   # gripper rotated for a side-grab
        attach_offset = grasp_rotation  # multiply FK by this when attaching
 
        # ---------- 1) default -> approach (side of object) ----------
        approach_pose = SE3(obj_pose.t[0], obj_pose.t[1], obj_pose.t[2]) * SE3.Ry(pi/2)
        sol_approach = self.robot.ikine_LM(approach_pose, q0=self.robot.q, mask=[1,1,1,1,1,1], joint_limits=False)
        traj_approach = jtraj(self.robot.q, sol_approach.q, steps)
        for q in traj_approach.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()
 
        # ---------- 2) approach -> object -> target (carry) ----------
        # # move from current approach to exact object side (small translation) and attach
        # pick_pose = SE3(obj_pose.t) * grasp_rotation
        # sol_pick = self.robot.ikine_LM(pick_pose, q0=self.robot.q, mask=[1,1,1,0,0,0], joint_limits=True)
        # traj_pick = jtraj(self.robot.q, sol_pick.q, steps)
        # for q in traj_pick.q:
        #     self.robot.q = q
        #     # attach object to end-effector with fixed rotation (so it doesn't flip)
        #     obj.T = self.robot.fkine(q) * attach_offset
        #     self.safe_step()
 
        # Now compute target SE3 (apply same grasp_rotation so robot matches orientation)
        x, y, z, roll, pitch, yaw = target_xyzrpy
        target_pose = SE3.Trans(x, y, z) * SE3.RPY([roll, pitch, yaw])
        target_with_grasp = target_pose * grasp_rotation  # orientation matched to grasp
        sol_target = self.robot.ikine_LM(target_with_grasp, q0=self.robot.q, mask=[1,1,1,1,1,1], joint_limits=False)
        traj_target = jtraj(self.robot.q, sol_target.q, steps)
        for q in traj_target.q:
            self.estop_event.wait()
            self.robot.q = q
            # keep object rigidly attached to end-effector with same orientation
            obj.T = self.robot.fkine(q) * SE3.Ry(-pi/2)
            self.safe_step()
 
        # ---------- 3) place and return home ----------
        # Place exactly at requested target_pose (dropping attachment)
        obj.T = target_pose
 
        # Return to home
        sol_home = self.robot.ikine_LM(SE3(self.robot.fkine(self.robot.q).t) , q0=self.robot.q)  # just to be safe
        traj_home = jtraj(self.robot.q, q_home, steps)
        for q in traj_home.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()
 
        # done