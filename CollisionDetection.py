from flask import Flask, render_template, request, jsonify
from swift import Swift
import roboticstoolbox as rtb
from roboticstoolbox import jtraj, DHRobot, DHLink, trapezoidal
import threading
import time
from spatialmath import SE3
from spatialmath.base import *
from spatialgeometry import Cuboid, Mesh, Sphere
import numpy as np
from ir_support import line_plane_intersection, RectangularPrism
from itertools import combinations


class CollisionManager:
    def __init__(self, env: Swift, cuboid_center, cuboid_scale, prism_scale=None):
        """
        cuboid_center: [x, y, z] location of the object in workspace
        cuboid_scale: (x, y, z) visual cuboid size
        prism_scale: (x, y, z) collision detection volume size (can be larger than cuboid)
        """
        self.env = env

        # 1️⃣ Create visible cuboid in the scene
        self.prism = Cuboid(scale=cuboid_scale, color=[0, 1, 0, 0.5])
        self.prism.T = transl(cuboid_center)
        self.env.add(self.prism)

        # 2️⃣ Create separate collision geometry (can be larger)
        if prism_scale is None:
            prism_scale = cuboid_scale  # default to same size if not specified
        
        self.prism_scale = prism_scale

        self.collision_prism = RectangularPrism(length=prism_scale[0],width=prism_scale[1],height=prism_scale[2],center=cuboid_center)
        self.vertices, self.faces, self.face_normals = self.collision_prism.get_data()

    def move_obstacle(self, new_center):
        """Move both visible and collision geometry to new location"""
        self.center = new_center
        self.prism.T = transl(new_center)
        self.collision_prism = RectangularPrism(length=self.prism_scale[0],width=self.prism_scale[1],height=self.prism_scale[2],center=new_center)
        self.vertices, self.faces, self.face_normals = self.collision_prism.get_data()
        self.env.step(0.02)
        print(f"📦 Obstacle moved to {new_center}")

    def move_far(self):
        """Move obstacle far out of the workspace"""
        far_pos = [100, 100, 0]
        self.move_obstacle(far_pos)

        # self.env = env
        # self.prism = Cuboid(scale=(0.1,0.1,0.1), color=[0,1,0,0.5]) #pose=SE3(1,1,0))
        # x, y, z = 0.3, -0.2, 0.0875
        # # x, y, z = 0.3, -2.2, 0.5875
        # self.prism.T = transl([x, y, z])
        # self.collision_prism = RectangularPrism(length=0.2, width=0.2, height=0.2, center=[x, y, z])
        # self.vertices, self.faces, self.face_normals = self.collision_prism.get_data()
        # self.env.add(self.prism)

    @staticmethod
    def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
        u = triangle_verts[1] - triangle_verts[0]
        v = triangle_verts[2] - triangle_verts[0]
        uu = np.dot(u, u)
        uv = np.dot(u, v)
        vv = np.dot(v, v)
        w = intersect_p - triangle_verts[0]
        wu = np.dot(w, u)
        wv = np.dot(w, v)
        D = uv * uv - uu * vv
        s = (uv * wv - vv * wu) / D
        t = (uv * wu - uu * wv) / D
        return (0 <= s <= 1) and (0 <= t <= 1) and (s + t <= 1)
    

    def is_collision(self, robot, q_matrix):
        """
        Check collision between robot's path and obstacle.
        Returns True if collision detected, False otherwise.
        """
        safety_distance = 0.0

        for q in q_matrix:
            q_list = [float(a) for a in q]
            T = robot.fkine(q_list).A
            ee_pos = T[:3, 3]

            # Check distance to faces
            for j, face in enumerate(self.faces):
                vert_on_plane = self.vertices[face][0]
                normal = self.face_normals[j]
                distance = np.dot(normal, (ee_pos - vert_on_plane))
                if 0 < distance < safety_distance:
                    print(f"⚠️ Proximity alert! EE {distance:.3f} m from face {j}.")
                    return True

            # Line-plane intersection
            tr = np.array([ee_pos, ee_pos + T[:3, 2] * 0.05])
            for j, face in enumerate(self.faces):
                vert_on_plane = self.vertices[face][0]
                intersect_p, check = line_plane_intersection(self.face_normals[j], vert_on_plane, tr[0], tr[1])
                if check:
                    triangle_list = np.array(list(combinations(face, 3)), dtype=int)
                    for triangle in triangle_list:
                        if self.is_intersection_point_inside_triangle(intersect_p, self.vertices[triangle]):
                            print("⚠️ Collision detected at EE position:", ee_pos)
                            return True

        return False
    