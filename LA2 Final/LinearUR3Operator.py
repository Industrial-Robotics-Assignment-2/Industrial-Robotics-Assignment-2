# LinearUR3Operator.py
import numpy as np
from spatialmath import SE3
from spatialmath.base import tr2angvec
from roboticstoolbox import jtraj, trapezoidal
from math import pi
from scipy import linalg
 
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
 


        x, y, z, roll, pitch, yaw = target_xyzrpy
        target_pose = SE3.Trans(x, y, z) * SE3.RPY([roll, pitch, yaw])
        target_with_grasp = target_pose * grasp_rotation  # includes fixed grasp rotation

        # RMRC parameters
        delta_t = 0.01
        min_manip_measure = 0.1
        steps_rmrc = steps if steps is not None else 75

        # --- Starting EE pose ---
        q_current = self.robot.q.copy()
        SE3_start = self.robot.fkine(q_current)
        SE3_target = target_with_grasp

        # Precompute trapezoidal scaling
        s = trapezoidal(0, 1, steps_rmrc).q

        # Storage for joint trajectory
        q_matrix = np.zeros([steps_rmrc, self.robot.n])
        q_matrix[0, :] = q_current
        m = np.zeros([1, steps_rmrc])

        # --- RMRC loop ---
        for i in range(steps_rmrc - 1):
            # Interpolate EE pose in Cartesian space (position + orientation)
            SE3_i = SE3_start.interp(SE3_target, s[i])
            SE3_next = SE3_start.interp(SE3_target, s[i+1])

            # Compute linear velocity
            xdot = np.zeros(6)
            xdot[:3] = (SE3_next.t - SE3_i.t) / delta_t

            # Compute angular velocity using stepwise rotation error
            R_err = SE3_next.R @ SE3_i.R.T
            theta, u = tr2angvec(R_err)
            # Limit angular step to avoid snapping
            max_theta_step = np.pi/18  # 10 degrees per step
            theta = np.clip(theta, -max_theta_step, max_theta_step)
            xdot[3:] = u * theta / delta_t

            # Jacobian and manipulability
            J = self.robot.jacob0(q_matrix[i, :])
            m[:, i] = np.sqrt(np.linalg.det(J[:3, :] @ J[:3, :].T))

            # Compute joint velocities (damped least squares if manipulability low)
            if m[:, i] < min_manip_measure:
                qdot = np.linalg.inv(J.T @ J + 0.01*np.eye(J.shape[1])) @ J.T @ xdot
            else:
                qdot = np.linalg.pinv(J) @ xdot

            # Integrate to get next joint configuration
            q_matrix[i+1, :] = q_matrix[i, :] + delta_t * qdot

        # --- Execute trajectory with safety/collision checks ---
        for q in q_matrix:
            self.estop_event.wait()
            self.robot.q = q

            # Keep object rigidly attached to EE with grasp offset
            obj.T = self.robot.fkine(q) * SE3.Ry(-pi/2)

            self.safe_step()


        # Now compute target SE3 (apply same grasp_rotation so robot matches orientation)
        # x, y, z, roll, pitch, yaw = target_xyzrpy
        # target_pose = SE3.Trans(x, y, z) * SE3.RPY([roll, pitch, yaw])
        # target_with_grasp = target_pose * grasp_rotation  # orientation matched to grasp
        # sol_target = self.robot.ikine_LM(target_with_grasp, q0=self.robot.q, mask=[1,1,1,1,1,1], joint_limits=False)
        # traj_target = jtraj(self.robot.q, sol_target.q, steps)
        # for q in traj_target.q:
        #     self.estop_event.wait()
        #     self.robot.q = q
        #     # keep object rigidly attached to end-effector with same orientation
        #     obj.T = self.robot.fkine(q) * SE3.Ry(-pi/2)
        #     self.safe_step()
 
        # # ---------- 3) place and return home ----------
        # # Place exactly at requested target_pose (dropping attachment)
        # obj.T = target_pose
 
        # Return to home
        sol_home = self.robot.ikine_LM(SE3(self.robot.fkine(self.robot.q).t) , q0=self.robot.q)  # just to be safe
        traj_home = jtraj(self.robot.q, q_home, steps)
        for q in traj_home.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()
 
        # done