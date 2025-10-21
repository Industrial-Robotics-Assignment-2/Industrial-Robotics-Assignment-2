import numpy as np
from spatialmath import SE3
from roboticstoolbox import jtraj, trapezoidal
from scipy import linalg
from CollisionDetection import CollisionManager
import threading

class FrybotOperations:
    def __init__(self, frybot, env_builder, estop_event, collision_detection):
        """
        frybot: the C4A601S robot from EnvBuilder
        env_builder: your EnvBuilder instance with all objects loaded
        """
        self.robot = frybot
        self.env = env_builder.env
        self.collision_event = threading.Event()
        self.collision_event.set()  # start unblocked
        self.estop_event = estop_event
        self.collision_detection = collision_detection

        # Use objects already in your EnvBuilder
        self.Frypile = env_builder.fries_pile
        self.Frybox = env_builder.fry_box
        self.FryboxFull = env_builder.fry_box_full
        self.Frybox_attached_offset = env_builder.fry_box_attached_offset

        # Offsets
        self.y_offset = 2.0
        self.z_offset = 0.5

    def safe_step(self, dt=0.02):
        self.env.step(dt)

    def frybot_operation(self):
        print("🍟 Starting Frybot operation sequence...")

        # Reset robot
        self.robot.q = np.zeros(self.robot.n)

        # --- Move to pick Frybox --- JTRAJ 1 ---
        T_pick = SE3(-0.22, -0.2 + self.y_offset, 0.05 + self.z_offset) * SE3.RPY([0, np.pi/2, 0])
        sol = self.robot.ikine_LM(T_pick, q0=self.robot.q, mask=[1]*6, joint_limits=True)
        traj = jtraj(self.robot.q, sol.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()

        # --- Move down to scooping start --- JTRAJ 2 ---
        T1 = SE3(-0.3, -0.4 + self.y_offset, 0.052 + self.z_offset) * SE3.Rx(np.pi)
        q2 = self.robot.ikine_LM(T1, q0=self.robot.q, mask=[1]*6, joint_limits=True).q
        traj = jtraj(self.robot.q, q2, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.robot.q = q
            self.Frybox.T = self.robot.fkine(q) @ self.Frybox_attached_offset
            self.safe_step()

        # --- Linear scooping motion --- JTRAJ + Pseudoinverse ---
        x1 = np.array([-0.3, -0.4 + self.y_offset, 0.052 + self.z_offset])
        x2 = np.array([0, -0.4 + self.y_offset, 0.052 + self.z_offset])
        delta_t1 = 0.02
        min_manip_measure1 = 0.1
        steps1 = 75
        x_path1 = np.empty([3, steps1])
        m1 = np.zeros([1, steps1])
        mask1 = [1]*6
        s1 = trapezoidal(0,1,steps1).q

        for i in range(steps1):
            x_path1[:, i] = x1 * (1 - s1[i]) + s1[i] * x2

        q_matrix1 = np.zeros([steps1, self.robot.n])
        q_matrix1[0,:] = self.robot.ikine_LM(T1, q0=np.zeros(self.robot.n), mask=mask1).q

        for i in range(steps1-1):
            xdot = np.zeros(6)
            xdot[:3] = (x_path1[:, i+1] - x_path1[:, i]) / delta_t1
            J = self.robot.jacob0(q_matrix1[i])
            m1[:, i] = np.sqrt(linalg.det(J[:3,:] @ J[:3,:].T))
            if m1[:,i] < min_manip_measure1:
                qdot = linalg.inv(J.T@J + 0.01*np.eye(J.shape[1])) @ J.T @ xdot
            else:
                qdot = linalg.pinv(J) @ xdot
            q_matrix1[i+1,:] = q_matrix1[i,:] + delta_t1*qdot

        for q in q_matrix1:
            self.estop_event.wait()
            self.robot.q = q
            self.Frybox.T = self.robot.fkine(q) @ self.Frybox_attached_offset
            self.safe_step()

        # --- Scooping arc motion --- JTRAJ 3 ---
        delta_t2 = 0.02
        min_manip_measure2 = 0.1
        steps2 = 75
        delta_theta = np.pi/2 / steps2
        theta = np.zeros([3, steps2])
        m2 = np.zeros([1, steps2])
        x_path2 = np.zeros([3, steps2])

        for i in range(steps2):
            theta_i = delta_theta*i
            x_path2[:,i] = [0.1*np.sin(theta_i), -0.4 + self.y_offset, 0.1*(1 - np.cos(theta_i)) + self.z_offset]
            theta[:,i] = [0, -theta_i, 0]

        q_matrix2 = np.zeros([steps2, self.robot.n])
        q_matrix2[0,:] = q_matrix1[-1,:]

        # Hide Frybox and show full
        self.Frybox.T = SE3(0,0,-100)
        self.FryboxFull.T = self.robot.fkine(q_matrix2[0,:]) @ self.Frybox_attached_offset

        for i in range(steps2-1):
            xdot = np.zeros(6)
            xdot[:3] = (x_path2[:,i+1] - x_path2[:,i])/delta_t2
            xdot[3:6] = (theta[:,i+1] - theta[:,i])/delta_t2
            J = self.robot.jacob0(q_matrix2[i])
            m2[:,i] = np.sqrt(linalg.det(J @ J.T))
            if m2[:,i] < min_manip_measure2:
                qdot = linalg.inv(J.T@J + 0.01*np.eye(J.shape[1])) @ J.T @ xdot
            else:
                qdot = linalg.pinv(J) @ xdot
            q_matrix2[i+1,:] = q_matrix2[i,:] + delta_t2*qdot

        for q in q_matrix2:
            self.estop_event.wait()
            self.robot.q = q
            self.FryboxFull.T = self.robot.fkine(q) @ self.Frybox_attached_offset
            self.safe_step()

        # --- Move to final location --- JTRAJ 4 ---
        x_start = self.robot.fkine(q_matrix2[-1,:]).t
        x_final = np.array([-0.6, 0 + self.y_offset, 0.075 + self.z_offset])
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
            self.FryboxFull.T = self.robot.fkine(q) @ self.Frybox_attached_offset
            self.safe_step()

        # Return to default
        q_home = np.zeros(self.robot.n)
        traj_home = jtraj(self.robot.q, q_home, 100)
        for q in traj_home.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()

        print("✅ Frybot operation complete!")
