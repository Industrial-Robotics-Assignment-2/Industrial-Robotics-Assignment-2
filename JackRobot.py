import numpy as np
from spatialmath import SE3
from roboticstoolbox import jtraj, trapezoidal
from math import pi
from spatialgeometry import Cuboid
from CollisionDetection import CollisionManager
import threading
from scipy import linalg

class JackRobot:
    def __init__(self, env_builder, estop_event, collision_detection):
        self.env = env_builder.env
        self.robot = env_builder.robot
        self.collision_event = threading.Event()
        self.collision_event.set()
        self.collision_detection = collision_detection
        self.estop_event = estop_event

        # Burger & box parts
        self.cylinders = env_builder.cylinders
        self.lid = env_builder.lid
        self.box_floor = env_builder.box_floor
        self.front_wall = env_builder.front_wall
        self.back_wall = env_builder.back_wall
        self.left_wall = env_builder.left_wall
        self.right_wall = env_builder.right_wall

        self.fence_center = env_builder.fence_center
        self.fence_thickness = env_builder.fence_thickness
        self.fence_height = env_builder.fence_height
        self.box_size = env_builder.box_size
        self.closed_box = None

    def safe_step(self, dt=0.02):
        self.env.step(dt)

    def stack_burger_into_box(self, steps=50, safe_height=0.3):
        total_height = 0.0
        for cyl in self.cylinders:
            pick_pose = SE3(cyl.T) * SE3.Rx(pi)
            sol_pick = self.robot.ikine_LM(pick_pose, q0=self.robot.q)
            for q in jtraj(self.robot.q, sol_pick.q, steps).q:
                self.estop_event.wait()
                self.robot.q = q
                self.env.step(0.02)

            # Move above box
            target_above_box = self.fence_center * SE3(0, 0, safe_height) * SE3.Rx(pi)
            sol_safe = self.robot.ikine_LM(target_above_box, q0=self.robot.q)
            for q in jtraj(self.robot.q, sol_safe.q, steps).q:
                self.estop_event.wait()
                self.robot.q = q
                cyl.T = self.robot.fkine(q)
                self.env.step(0.02)

            # Lower into box
            target_z = self.fence_thickness + total_height + 0.02
            target_in_box = self.fence_center * SE3(0, 0, target_z) * SE3.Rx(pi)
            sol_box = self.robot.ikine_LM(target_in_box, q0=self.robot.q)
            for q in jtraj(self.robot.q, sol_box.q, steps).q:
                self.estop_event.wait()
                self.robot.q = q
                cyl.T = self.robot.fkine(q)
                self.env.step(0.02)

            total_height += 0.02

    def animate_lid_closing(self, steps=50, safe_height=0.3):
        lid_pick_pose = self.lid.T * SE3.Rx(pi)
        sol_pick = self.robot.ikine_LM(lid_pick_pose, q0=self.robot.q)
        for q in jtraj(self.robot.q, sol_pick.q, steps).q:
            self.estop_event.wait()
            self.robot.q = q
            self.env.step(0.02)

        # Move above box
        target_above_box = self.fence_center * SE3(0, 0, safe_height) * SE3.Rx(pi)
        sol_safe = self.robot.ikine_LM(target_above_box, q0=self.robot.q)
        for q in jtraj(self.robot.q, sol_safe.q, steps).q:
            self.estop_event.wait()
            self.robot.q = q
            self.lid.T = self.robot.fkine(q)
            self.env.step(0.02)

        # Lower lid
        box_top_z = self.fence_height + self.fence_thickness / 2
        target_in_box = self.fence_center * SE3(0, 0, box_top_z) * SE3.Rx(pi)
        sol_box = self.robot.ikine_LM(target_in_box, q0=self.robot.q)
        for q in jtraj(self.robot.q, sol_box.q, steps).q:
            self.estop_event.wait()
            self.robot.q = q
            self.lid.T = self.robot.fkine(q)
            self.env.step(0.02)

        # Create closed box
        total_box_height = self.fence_height + 2*self.fence_thickness
        self.closed_box = Cuboid([self.box_size, self.box_size, total_box_height],
                                 pose=self.fence_center * SE3(0, 0, total_box_height/2),
                                 color=[1,0,0])
        self.move_old_parts()
        self.env.add(self.closed_box)

    def move_old_parts(self):
        old_parts = [
            self.box_floor,
            self.front_wall, self.back_wall,
            self.left_wall, self.right_wall,
            self.lid
        ] + self.cylinders

        for part in old_parts:
            part.T = SE3(100, 100, 100)

    def move_closed_box(self, target_x=-0.5, lift_height=0.8, steps1=70, steps2=150, steps3=70):
        if self.closed_box is None:
            print("❌ No closed box found.")
            return
    
        delta_t = 0.02
        min_manip_measure = 0.1
        grasp_offset = SE3(0, 0, 0)
        # --- INITIAL POSE ---
        current_pose = self.fence_center * SE3(0, 0, self.fence_height + self.fence_thickness/2) * SE3.Rx(pi)
        q_start = self.robot.ikine_LM(current_pose, q0=self.robot.q).q
    
        # --- JTRAJ 1: Lift up ---
        safe_pose = SE3(current_pose.t[0], current_pose.t[1], lift_height) * SE3.Rx(pi)
        sol_safe = self.robot.ikine_LM(safe_pose, q0=q_start)
        q_matrix1 = jtraj(q_start, sol_safe.q, steps1).q
    
        for q in q_matrix1:
            self.estop_event.wait()
            self.robot.q = q
            self.closed_box.T = self.robot.fkine(q) * grasp_offset
            self.safe_step()
    
        # --- RMRC 2: Semicircle motion to target ---
        center_x = (current_pose.t[0] + target_x) / 2
        center_y = current_pose.t[1]
        radius = abs(target_x - current_pose.t[0]) / 2
        theta = np.linspace(0, np.pi, steps2)
    
        x_path = np.zeros([3, steps2])
        theta_path = np.zeros([3, steps2])
        m = np.zeros([1, steps2])
        q_matrix2 = np.zeros([steps2, self.robot.n])
        q_matrix2[0, :] = q_matrix1[-1, :]
    
        # define semicircular path in X-Y plane
        for i in range(steps2):
            x_path[:, i] = [center_x + radius * np.cos(theta[i]),
                            center_y + radius * np.sin(theta[i]),
                            lift_height]
            theta_path[:, i] = [0, 0, 0]  # keep orientation constant
    
        for i in range(steps2 - 1):
            xdot = np.zeros(6)
            xdot[:3] = (x_path[:, i+1] - x_path[:, i]) / delta_t
            xdot[3:6] = (theta_path[:, i+1] - theta_path[:, i]) / delta_t
            J = self.robot.jacob0(q_matrix2[i])
            m[:, i] = np.sqrt(linalg.det(J @ J.T))
            
            if m[:, i] < min_manip_measure:
                qdot = linalg.inv(J.T @ J + 0.01 * np.eye(J.shape[1])) @ J.T @ xdot
            else:
                qdot = linalg.pinv(J) @ xdot
            q_matrix2[i+1, :] = q_matrix2[i, :] + delta_t * qdot
    
        for q in q_matrix2:
            self.estop_event.wait()
            self.collision_event.wait()
            self.robot.q = q
            q_list = [float(a) for a in q]  # convert to Python floats
            self.robot.q = q_list
            if self.collision_detection.is_collision(self.robot, [q_list]):
                print(f"Collision detected at joint {q_list}")
                self.collision_event.clear()
            self.closed_box.T = self.robot.fkine(q) * grasp_offset
            self.safe_step()
    
        # # --- JTRAJ 3: Place down ---
        # place_pose = SE3(target_x, current_pose.t[1], 0.5 + self.fence_height/2) * SE3.Rx(pi)
        # sol_place = self.robot.ikine_LM(place_pose, q0=self.robot.q)
        # q_matrix3 = jtraj(q_matrix2[-1, :], sol_place.q, steps3).q
    
        # for q in q_matrix3:
        #     self.estop_event.wait()
        #     self.robot.q = q
        #     self.closed_box.T = self.robot.fkine(q) * grasp_offset
        #     self.safe_step()
        # --- Move to final location --- JTRAJ 4 ---
        x_start = self.robot.fkine(q_matrix2[-1,:]).t
        x_final = np.array([-0.6, 0, 0.075 + 0.5])
        delta_t3 = 0.02
        min_manip_measure3 = 0.1
        steps3 = 75
        x_path3 = np.empty([3, steps3])
        m3 = np.zeros([1, steps3])
        s3 = trapezoidal(0,1,steps3).q

        for i in range(steps3):
            x_path3[:,i] = x_start*(1-s3[i]) + s3[i]*x_final

        q_matrix3 = np.zeros([steps3, self.robot.n])
        q_matrix3[0,:] = q_matrix2[-1,:]

        for i in range(steps3-1):
            xdot = np.zeros(6)
            xdot[:3] = (x_path3[:,i+1] - x_path3[:,i])/delta_t3
            J = self.robot.jacob0(q_matrix3[i])
            m3[:,i] = np.sqrt(linalg.det(J[:3,:] @ J[:3,:].T))
            if m3[:,i] < min_manip_measure3:
                qdot = linalg.inv(J.T@J + 0.01*np.eye(J.shape[1])) @ J.T @ xdot
            else:
                qdot = linalg.pinv(J) @ xdot
            q_matrix3[i+1,:] = q_matrix3[i,:] + delta_t3*qdot

        for q in q_matrix3:
            self.estop_event.wait()
            self.collision_event.wait()
            self.robot.q = q
            q_list = [float(a) for a in q]  # convert to Python floats
            self.robot.q = q_list
            if self.collision_detection.is_collision(self.robot, [q_list]):
                print(f"Collision detected at joint {q_list}")
                self.collision_event.clear()
            self.closed_box.T = self.robot.fkine(q) * grasp_offset
            self.safe_step()
    
        print("✅ Closed box moved successfully.")

    def return_robot_home(self, steps=50):
        q_home = np.zeros(6)
        for q in jtraj(self.robot.q, q_home, steps).q:
            self.estop_event.wait()
            self.robot.q = q
            self.env.step(0.02)