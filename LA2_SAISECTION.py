from flask import Flask, render_template, request, jsonify
import swift
import roboticstoolbox as rtb
from roboticstoolbox import jtraj, DHRobot, DHLink, trapezoidal
import threading
import time
from spatialmath import SE3
from spatialmath.base import *
from spatialgeometry import Cuboid, Mesh, Sphere
import numpy as np
from C4A601S import C4A601S
from ir_support import LinearUR3, CylindricalDHRobotPlot, line_plane_intersection, RectangularPrism
import os
from math import pi
from scipy import linalg
from C4A601S import C4A601S
from itertools import combinations

app = Flask(__name__)
env = swift.Swift()
robots = {}
controlled_robot = [None]
q = None
# Launch Swift
env.launch(realtime=True)

estop_active = False
lock = threading.Lock()
teach_active = False
env_running = True

# Use threading.Event for proper E-STOP/resume
estop_event = threading.Event()
estop_event.clear()  # Not active initially

collision_active = False
collision_event = threading.Event()
collision_event.set()  # no collision initially
prism = Cuboid(scale=(0.1,0.1,0.1), color=[0,1,0,0.5]) #pose=SE3(1,1,0))
x, y, z = 100, 100, 0
prism.T = transl([x, y, z])
collision_prism = RectangularPrism(length=0.2, width=0.2, height=0.2, center=[x, y, z])
vertices, faces, face_normals = collision_prism.get_data()
env.add(prism)

def frybot_operation(robot, env):
    """
    Complete Frybot pick–scoop–drop operation with GUI-linked E-STOP/RESUME.
    """

    print("🍟 Starting Frybot operation sequence...")

    global estop_active, lock

    # Helper functions to respect E-STOP
    def safe_step():
        estop_event.wait()  # Wait until resume if E-STOP active
        collision_event.wait()  # wait if collision detected
        env.step(0.02)

    def safe_sleep(duration):
        t0 = time.time()
        while time.time() - t0 < duration:
            estop_event.wait()
            time.sleep(0.05)

    # -------------------------------
    # Add meshes
    # -------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    Frypile_path = os.path.join(current_dir, "friespile.dae")
    Frypile = Mesh(filename=Frypile_path, pose=SE3(-0.15, -0.4, 0))
    env.add(Frypile)

    Frybox_path = os.path.join(current_dir, "FryboxConstruct.dae")
    Frybox = Mesh(filename=Frybox_path, pose=SE3(-0.2, -0.2, 0))
    env.add(Frybox)
    Frybox_attached_offset = SE3(-0.075, 0, 0.02) * SE3.Ry(np.pi / 2)

    FryboxFull_path = os.path.join(current_dir, "FryboxFull.dae")
    FryboxFull = Mesh(filename=FryboxFull_path)
    marker_radius = 0.005
    marker_mesh = Sphere(marker_radius)

    # -------------------------------
    # Define trajectories and motion
    # -------------------------------
    T1 = SE3(-0.3, -0.4, 0.052) * SE3.Rx(np.pi)
    Frybox_location = SE3(-0.22, -0.2, 0.05) * SE3.RPY([0, pi / 2, 0])
    T2 = SE3(0, -0.4, 0.052) * SE3.Rx(np.pi)
    T3 = SE3(0.1, -0.4, 0.152) * SE3.Rx(np.pi) * SE3.Ry(-np.pi / 2)

    robot.qz = np.zeros(robot.n)
    q = robot.qz

    # --- Move to pick Frybox
    q1 = robot.ikine_LM(Frybox_location, q0=q, mask=[1, 1, 1, 1, 1, 1], joint_limits=True).q
    traj = jtraj(robot.q, q1, 100)
    for t in traj.q:
        estop_event.wait()
        collision_event.wait()
        robot.q = t
        ee_position = robot.fkine(t).A[:3, 3]
        safe_step()

    # --- Move down to start scooping
    q_start = robot.q
    q2 = robot.ikine_LM(T1, q0=q_start, mask=[1, 1, 1, 1, 1, 1], joint_limits=True).q
    traj = jtraj(robot.q, q2, 100)
    for t in traj.q:
        estop_event.wait()
        collision_event.wait()
        robot.q = t
        ee_position = robot.fkine(t)
        Frybox.T = ee_position * Frybox_attached_offset
        safe_step()

    # -------------------------------
    # Linear X-Y Scooping Trajectory
    # -------------------------------
    x1 = np.array([-0.3, -0.4, 0.052])
    x2 = np.array([0, -0.4, 0.052])
    delta_t1 = 0.02
    min_manip_measure1 = 0.1
    steps1 = 75
    x_path1 = np.empty([3, steps1])
    m1 = np.zeros([1, steps1])
    error1 = np.empty([6, steps1])
    mask1 = [1, 1, 1, 1, 1, 1]
    s1 = trapezoidal(0, 1, steps1).q

    for i in range(steps1):
        x_path1[:, i] = x1 * (1 - s1[i]) + s1[i] * x2

    q_matrix1 = np.zeros([steps1, robot.n])
    q_matrix1[0, :] = robot.ikine_LM(T1, q0=np.zeros(robot.n), mask=mask1).q

    for i in range(steps1 - 1):
        xdot = np.zeros(6)
        xdot[0:3] = (x_path1[:, i + 1] - x_path1[:, i]) / delta_t1
        xdot[3:6] = 0
        J = robot.jacob0(q_matrix1[i])
        m1[:, i] = np.sqrt(linalg.det(J[:3, :] @ J[:3, :].T))

        if m1[:, i] < min_manip_measure1:
            qdot = linalg.inv(J.T @ J + 0.01 * np.eye(J.shape[1])) @ J.T @ xdot
        else:
            qdot = linalg.pinv(J) @ xdot

        q_matrix1[i + 1, :] = q_matrix1[i, :] + delta_t1 * qdot

    for q in q_matrix1:
        estop_event.wait()
        collision_event.wait()
        robot.q = q
        ee_position = robot.fkine(q)
        Frybox.T = ee_position * Frybox_attached_offset
        safe_step()

    # -------------------------------
    # Scooping motion arc
    # -------------------------------
    delta_t2 = 0.02
    min_manip_measure2 = 0.1
    steps2 = 75
    delta_theta = np.pi / 2 / steps2
    theta = np.zeros([3, steps2])
    m2 = np.zeros([1, steps2])
    mask2 = [1, 1, 1, 1, 1, 1]
    x_path2 = np.zeros([3, steps2])

    for i in range(steps2):
        theta_i = delta_theta * i
        x_path2[:, i] = [0.1 * np.sin(theta_i), -0.4, 0.1 * (1 - np.cos(theta_i))]
        theta[:, i] = [0, -theta_i, 0]

    q_matrix2 = np.zeros([steps2, robot.n])
    q_matrix2[0, :] = q_matrix1[-1, :]

    Frybox.T = SE3(0, 0, -100)
    FryboxFull.T = robot.fkine(q_matrix2[0, :]) * Frybox_attached_offset
    env.add(FryboxFull)

    for i in range(steps2 - 1):
        xdot = np.zeros(6)
        xdot[0:3] = (x_path2[:, i + 1] - x_path2[:, i]) / delta_t2
        xdot[3:6] = (theta[:, i + 1] - theta[:, i]) / delta_t2
        J = robot.jacob0(q_matrix2[i])
        m2[:, i] = np.sqrt(linalg.det(J @ J.T))

        if m2[:, i] < min_manip_measure2:
            qdot = linalg.inv(J.T @ J + 0.01 * np.eye(J.shape[1])) @ J.T @ xdot
        else:
            qdot = linalg.pinv(J) @ xdot

        q_matrix2[i + 1, :] = q_matrix2[i, :] + delta_t2 * qdot

    for q in q_matrix2:
        estop_event.wait()
        collision_event.wait()
        robot.q = q
        ee_position = robot.fkine(q)
        FryboxFull.T = ee_position * Frybox_attached_offset
        safe_step()

    # -------------------------------
    # Final placement trajectory
    # -------------------------------
    x_start = robot.fkine(q_matrix2[-1, :]).t
    x_final = np.array([0.5, 0, 0.075])
    delta_t3 = 0.02
    min_manip_measure3 = 0.1
    steps3 = 75
    x_path3 = np.empty([3, steps3])
    m3 = np.zeros([1, steps3])
    s3 = trapezoidal(0, 1, steps3).q

    for i in range(steps3):
        x_path3[:, i] = x_start * (1 - s3[i]) + s3[i] * x_final

    q_matrix3 = np.zeros([steps3, robot.n])
    q_matrix3[0, :] = q_matrix2[-1, :]

    for i in range(steps3 - 1):
        xdot = np.zeros(6)
        xdot[0:3] = (x_path3[:, i + 1] - x_path3[:, i]) / delta_t3
        xdot[3:6] = 0
        J = robot.jacob0(q_matrix3[i])
        m3[:, i] = np.sqrt(linalg.det(J[:3, :] @ J[:3, :].T))

        if m3[:, i] < min_manip_measure3:
            qdot = linalg.inv(J.T @ J + 0.01 * np.eye(J.shape[1])) @ J.T @ xdot
        else:
            qdot = linalg.pinv(J) @ xdot

        q_matrix3[i + 1, :] = q_matrix3[i, :] + delta_t3 * qdot

    for q in q_matrix3:
        estop_event.wait()
        collision_event.wait()
        robot.q = q
        q_list = [float(a) for a in q]  # convert to Python floats
        robot.q = q_list
        if is_collision(robot, [q_list], faces, vertices, face_normals, env=env):
            print(f"Collision detected at joint {q_list}")
            collision_event.clear()
        ee_position = robot.fkine(q)
        FryboxFull.T = ee_position * Frybox_attached_offset
        safe_step()

    print("Frybot operation completed successfully.")

@app.route('/')
def index():
    return render_template('index_swift_and_flask_advanced_multi_robot_teach.html')

@app.route('/add_robot', methods=['POST'])
def add_robot():
    data = request.get_json()
    robot_name = data.get('robot')
    tx, ty, tz = data.get('tx', 0), data.get('ty', 0), data.get('tz', 0)

    config = {
        'ur5': {'model': rtb.models.UR5, 'pillar_height': 1.0},
        'ur3': {'model': rtb.models.UR3, 'pillar_height': 0.6},
        'c4a601s': {'model': C4A601S, 'pillar_height': 0.8, 
                    'qlim': [np.deg2rad([-170, 170]),np.deg2rad([-65, 160]),
                             np.deg2rad([-225, 51]),np.deg2rad([-200, 200]),
                             np.deg2rad([-135, 135]),np.deg2rad([-360, 360])]
                    },
    }
    if robot_name not in config:
        return jsonify(success=False, error="Unknown robot")

    if robot_name not in robots:
        pillar_height = config[robot_name]['pillar_height']
        tz += pillar_height
        pillar = Cuboid(scale=(0.2, 0.2, pillar_height), pose=SE3(tx, ty, pillar_height / 2))
        robot = config[robot_name]['model']()
        robot.base = SE3(tx, ty, tz)
        env.add(pillar)
        env.add(robot)
        robot.add_to_env(env)
        robots[robot_name] = robot

    controlled_robot[0] = robots[robot_name]
    global q
    q = controlled_robot[0].q.copy()

    # Return the robot's joint limits to the frontend
    qlim = controlled_robot[0].qlim.tolist()
    return jsonify(success=True, q=controlled_robot[0].q.tolist(), qlim=qlim)

@app.route('/get_joint_values', methods=['GET'])
def get_joint_values():
    if controlled_robot[0] is not None:
        return jsonify(q=controlled_robot[0].q.tolist())
    return jsonify(q=None), 404

@app.route('/update_joint', methods=['POST'])
def update_joint():
    data = request.get_json()
    idx, value = data['index'], data['value']
    robot = controlled_robot[0]
    if robot is None:
        return jsonify(success=False, error="No robot selected")
    robot.q[idx] = value
    global q
    q = robot.q.copy()
    return jsonify(success=True)

@app.route('/flail', methods=['POST'])
def flail():
    steps = 100
    for _ in range(steps):
        for name, bot in robots.items():
            q = bot.q
            q += 0.1 * (np.random.rand(len(q)) - 0.5)
            q = np.clip(q, bot.qlim[0], bot.qlim[1])
            bot.q = q
            if controlled_robot[0] == bot:
                globals()['q'] = q.copy()
        time.sleep(0.05)
    return jsonify(success=True)


@app.route('/update_cartesian', methods=['POST'])
def update_cartesian():
    data = request.get_json()
    axis = data['axis']  # 0 = X, 1 = Y, 2 = Z
    value = data['value']

    robot = controlled_robot[0]
    if robot is None:
        return jsonify(success=False, error="No robot selected")

    # Persistent offset storage per robot
    if not hasattr(robot, 'cartesian_offset'):
        robot.cartesian_offset = np.array([0.0, 0.0, 0.0])

    robot.cartesian_offset[axis] = value

    # Compute target pose (home + offset)
    T_home = robot.fkine(np.zeros(robot.n))
    T_target = T_home * SE3(robot.cartesian_offset)

    sol = robot.ikine_LM(T_target, q0=robot.q)
    if sol.success:
        robot.q = sol.q
        global q
        q = sol.q.copy()
        env.step()
        return jsonify(success=True, q=q.tolist())
    else:
        return jsonify(success=False, error="IK failed")
    

@app.route("/emergency_stop", methods=["POST"])
def emergency_stop():
    estop_event.clear()
    print("🛑 E-STOP ACTIVATED")
    return jsonify({"status":"stopped"})

@app.route("/resume", methods=["POST"])
def resume():
    estop_event.set()
    print("▶ RESUMED")
    return jsonify({"status":"resumed"})


def run_swift_loop():
    global teach_active
    while teach_active:
        if controlled_robot[0] is not None and q is not None:
            controlled_robot[0].q = q
            env.step(0.02)
        time.sleep(0.05)

@app.route("/reset_env", methods=["POST"])
def reset_env():
    global env, robots, controlled_robot, q, teach_active

    with lock:
        env.close()
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose(position=[0.5, 3, 0.8], look_at=[0, 0, 0])
        robots.clear()
        controlled_robot[0] = None
        q = None
        teach_active = True
        threading.Thread(target=run_swift_loop, daemon=True).start()

    print("🔄 Environment reset and restarted successfully")
    return jsonify(success=True)

def get_link_poses(robot, q=None):
    """Return transforms for all robot links."""
    if q is None:
        return robot.fkine_all().A
    return robot.fkine_all(q).A

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


def is_collision(robot, q_matrix, faces, vertices, face_normals, env=None):
    """Check collision between the robot's end-effector path and obstacle."""
    safety_distance = 0.00
    
    for q in q_matrix:
        q_list = [float(a) for a in q]
        T = robot.fkine(q_list).A  # end-effector transform
        ee_pos = T[:3, 3]          # position of EE

        for j, face in enumerate(faces):
            vert_on_plane = vertices[face][0]
            normal = face_normals[j]

            # Signed distance from EE to plane
            distance = np.dot(normal, (ee_pos - vert_on_plane))

            # If EE is within proximity range on the outside of the face
            if 0 < distance < safety_distance:
                print(f"⚠️ Proximity alert! EE {distance:.3f} m from face {j}.")
                return True

        # We'll simulate a tiny forward step to get a motion line
        tr = np.array([ee_pos, ee_pos + T[:3, 2] * 0.05])  # 5cm forward line

        for j, face in enumerate(faces):
            vert_on_plane = vertices[face][0]
            intersect_p, check = line_plane_intersection(
                face_normals[j], vert_on_plane, tr[0], tr[1]
            )
            if check:
                triangle_list = np.array(list(combinations(face, 3)), dtype=int)
                for triangle in triangle_list:
                    if is_intersection_point_inside_triangle(intersect_p, vertices[triangle]):
                        # if env is not None:
                        #     s = Sphere(radius=0.01, color=[1, 0, 0, 1])
                        #     s.T = SE3(intersect_p)
                        #     env.add(s)
                        print("⚠️ Collision detected at EE position:", ee_pos)
                        return True
    return False


@app.route("/spawn_obstacle", methods=["POST"])
def spawn_obstacle():
    global prism, collision_prism, faces, vertices, face_normals

    # New position for the prism
    x, y, z = 0.3, -0.2, 0.0875
    prism.T = transl([x, y, z])

    # Update collision geometry
    collision_prism.center = [x, y, z]
    collision_prism = RectangularPrism(length=0.2, width=0.2, height=0.2, center=[x, y, z])
    vertices, faces, face_normals = collision_prism.get_data()  # recompute global collision arrays

    env.step(0.02)
    print("Prism spawned in workspace")
    return jsonify(success=True)


@app.route("/move_obstacle_far", methods=["POST"])
def move_obstacle_far():
    global prism, collision_prism, faces, vertices, face_normals

    # Move prism far away
    x, y, z = 100, 100, 0
    prism.T = transl([x, y, z])

    # Update collision geometry
    collision_prism.center = [x, y, z]
    collision_prism = RectangularPrism(length=0.2, width=0.2, height=0.2, center=[x, y, z])
    vertices, faces, face_normals = collision_prism.get_data()  # recompute global collision arrays

    env.step(0.02)
    collision_event.set()  # resume motion
    print("Prism moved far away")
    return jsonify(success=True)

if __name__ == '__main__':
    # Start background thread to keep Swift updated
    threading.Thread(target=run_swift_loop, daemon=True).start()

    frybot = C4A601S()
    frybot.add_to_env(env)
    controlled_robot[0] = frybot
    estop_event.set()  # allow motion
    threading.Thread(target=frybot_operation, args=(frybot, env), daemon=True).start()


    # Start Flask
    app.run(debug=False)
