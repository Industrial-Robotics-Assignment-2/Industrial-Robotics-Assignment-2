import numpy as np
from spatialmath import SE3
from roboticstoolbox import jtraj, trapezoidal
from math import pi
from spatialgeometry import Cuboid
from CollisionDetection import CollisionManager
import threading
import FuncPoint as fp
from scipy import linalg
import time

class DrinkBotOperations:
    def __init__(self, drinkbot, env_builder, estop_event, collision_detection):

        self.robot = drinkbot
        self.env = env_builder.env
        self.collision_event = threading.Event()
        self.collision_event.set()  # start unblocked
        self.estop_event = estop_event
        self.collision_detection = collision_detection

        # Use objects already in the EnvBuilder
        self.cup = env_builder.cup
        self.cup_lid = env_builder.cup_lid
        self.straw = env_builder.straw
        self.machine = env_builder.machine
        self.whole_drink = env_builder.whole_drink

        # offsets

        self.cup_height = 0.125
        self.drink_machine_pos = -0.5

    def safe_step(self, dt=0.02):
        self.env.step(dt)

    def animate(self):

        print("Starting Drinkbot operation sequence...")

        # Reset robot
        self.robot.q = np.zeros(self.robot.n)

        # Pickup Cup        
        target_pos = self.cup.T * SE3(0,0,self.cup_height)
        ik_pose = self.robot.ikine_LM(target_pos, q0=self.robot.q, mask=[1,1,1,0,0,0])
        traj = jtraj(self.robot.q, ik_pose.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()

        # place cup on machine
        target_pos = self.machine.T * SE3(self.drink_machine_pos,0,self.cup_height + 0.125) * SE3.Ry(pi)
        ik_pose = self.robot.ikine_LM(target_pos, q0=self.robot.q, mask=[1,1,1,1,1,1])
        traj = jtraj(self.robot.q, ik_pose.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.cup.T = self.robot.fkine(self.robot.q) * SE3.Ry(pi) *SE3(0,0,-0.125)
            self.robot.q = q
            self.safe_step()

        time.sleep(2)

        # place cup
        target_pos = SE3(0, -1.75, 0.625) * SE3.Ry(pi)
        ik_pose = self.robot.ikine_LM(target_pos, q0=self.robot.q, mask=[1,1,1,1,1,1])
        traj = jtraj(self.robot.q, ik_pose.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.cup.T = self.robot.fkine(self.robot.q) * SE3.Ry(pi) *SE3(0,0,-0.125)
            self.robot.q = q
            self.safe_step()

        # pick up lid
        target_pos = self.cup_lid.T * SE3(0,0,0.125) * SE3.Ry(pi)
        ik_pose = self.robot.ikine_LM(target_pos, q0=self.robot.q, mask=[1,1,1,1,1,1])
        traj = jtraj(self.robot.q, ik_pose.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()

        # place lid
        target_pos = self.cup.T * SE3(0,0,self.cup_height) * SE3.Ry(pi)
        ik_pose = self.robot.ikine_LM(target_pos, q0=self.robot.q, mask=[1,1,1,1,1,1])
        traj = jtraj(self.robot.q, ik_pose.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.cup_lid.T = self.robot.fkine(self.robot.q) * SE3.Ry(pi) *SE3(0,0,-0.12)
            self.robot.q = q
            self.safe_step()

        self.cup_lid.T = self.cup.T

        # pick up straw
        target_pos = self.straw.T * SE3(0,0,0.15)
        ik_pose = self.robot.ikine_LM(target_pos, q0=self.robot.q, mask=[1,1,1,0,0,0])
        traj = jtraj(self.robot.q, ik_pose.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()

        # place straw above cup
        target_pos = self.cup.T * SE3(0,0,0.3) * SE3.Rx(-pi/2) * SE3.Rz(-pi/2)
        ik_pose = self.robot.ikine_LM(target_pos, q0=self.robot.q, mask=[1,1,1,1,1,1])
        traj = jtraj(self.robot.q, ik_pose.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.straw.T = self.robot.fkine(self.robot.q) * SE3.Ry(pi/2) *SE3(0,0,-0.15)
            self.robot.q = q
            self.safe_step()

        # put straw in cup
        # step = 120
        # p1 = self.robot.fkine(self.robot.q)
        # p2 = p1 * SE3(0,0,-0.3)

        # p1 = p1.t
        # p2 = p2.t

        # points = fp.points_between(p1,p2,step)

        # q_matrix = fp.RMRC(self.robot, self.robot.q, points, 2, step)

        # for q in q_matrix:
        #     self.estop_event.wait()
        #     self.straw.T = self.robot.fkine(self.robot.q) * SE3.Ry(pi/2) *SE3(0,0,-0.15)
        #     self.robot.q = q
        #     self.safe_step()

        target_pos = self.cup.T * SE3(0,0,0.15) * SE3.Rx(-pi/2) * SE3.Rz(-pi/2)
        ik_pose = self.robot.ikine_LM(target_pos, q0=self.robot.q, mask=[1,1,1,1,1,1])
        traj = jtraj(self.robot.q, ik_pose.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.straw.T = self.robot.fkine(self.robot.q) * SE3.Ry(pi/2) *SE3(0,0,-0.15)
            self.robot.q = q
            self.safe_step()

        self.straw.T = self.cup.T
        self.whole_drink.T = self.cup.T
        self.cup.T = SE3(-100, 100, 100)
        self.straw.T = self.cup.T
        self.cup_lid.T = self.cup.T

        # place for ur3
        # Pickup Cup        
        target_pos = self.whole_drink.T * SE3(0,0,self.cup_height)
        ik_pose = self.robot.ikine_LM(target_pos, q0=self.robot.q, mask=[1,1,1,0,0,0])
        traj = jtraj(self.robot.q, ik_pose.q, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()

        #RMRC move cup for ur3
        step = 120
        p1 = self.robot.fkine(self.robot.q)
        p2 = SE3(0, -2, 0.5) * SE3(-0.45,0,0.125)

        p1 = p1.t
        p2 = p2.t

        points = fp.points_between(p1,p2,step)

        q_matrix = fp.RMRC(rob=self.robot, q=self.robot.q, points=points, t=2, steps=step)

        for q in q_matrix:
            self.estop_event.wait()
            self.collision_event.wait()
            q_list = [float(a) for a in q]  # convert to Python floats
            if self.collision_detection.is_collision(self.robot, [q_list]):
                print(f"Collision detected at joint {q_list}")
                self.collision_event.clear()
            self.whole_drink.T = self.robot.fkine(self.robot.q) * SE3.Ry(pi/2) * SE3(0,0,-0.15)
            self.robot.q = q
            self.safe_step()

        # reset
        target_pos = np.zeros(self.robot.n)
        traj = jtraj(self.robot.q, target_pos, 100)
        for q in traj.q:
            self.estop_event.wait()
            self.robot.q = q
            self.safe_step()
            
        print("✅ Drinkbot operation complete!")

if __name__ == "__main__":
    pass