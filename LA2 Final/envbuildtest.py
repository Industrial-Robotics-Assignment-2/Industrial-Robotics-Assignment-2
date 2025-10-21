from EnvBuilder import EnvBuilder
from SaiRobot import FrybotOperations
from JackRobot import JackRobot
from CollisionDetection import CollisionManager
from LinearUR3Operator import LinearUR3Operator
from serial_communication_func import SerialCom
import time
from flask import Flask, render_template, request, jsonify
import threading
import numpy as np
from spatialmath import SE3
from spatialmath.base import *
from spatialgeometry import Cuboid
from ir_support import RectangularPrism
from itertools import combinations
import roboticstoolbox as rtb
from C4A601S import C4A601S
from XArm6 import XArm6
from CytonGamma300DH import CytonGamma300
from swift import Swift
from DrinkBot import DrinkBotOperations

# ---------------------- 1️⃣ INITIALIZE ENVIRONMENT & ROBOTS ----------------------
env_builder = EnvBuilder()
env = env_builder.env

# Global state
estop_active = False
teach_active = True
estop_event = threading.Event()
estop_event.set()  # Initially running
lock = threading.Lock()

robots = {}
controlled_robot = [None]
q = None


frybot_collision = CollisionManager(env, cuboid_center=[100, 100, 0], cuboid_scale=(0.1, 0.1, 0.1), prism_scale=(0.25, 0.25, 0.25))

# Jack’s obstacle (different sizes)
jack_collision = CollisionManager(env,cuboid_center=[100, 100, 0],cuboid_scale=(0.1, 0.1, 0.1), prism_scale=(0.4, 0.4, 0.4))

drinkbot_collision = CollisionManager(env,cuboid_center=[100, 100, 0],cuboid_scale=(0.1, 0.1, 0.1), prism_scale=(0.4, 0.4, 0.4))


# Robot instances
frybot = FrybotOperations(env_builder.frybot, env_builder, estop_event, frybot_collision)
jack = JackRobot(env_builder, estop_event, jack_collision)
drinkbot = DrinkBotOperations(env_builder.CytonGamma300, env_builder, estop_event, drinkbot_collision)
LinearUR3 = LinearUR3Operator(env_builder, estop_event)

# ---------------------- 2️⃣ FLASK APP ----------------------
app = Flask(__name__)

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
        'XArm6': {'model': XArm6, 'pillar_height': 0.6, 
                    'qlim': [np.deg2rad([-170, 170]),np.deg2rad([-90, 90]),
                             np.deg2rad([-160, 160]),np.deg2rad([-300, 300]),
                             np.deg2rad([-120, 120]),np.deg2rad([-360, 360])]},
        'c4a601s': {'model': C4A601S, 'pillar_height': 0.8, 
                    'qlim': [np.deg2rad([-170, 170]),np.deg2rad([-65, 160]),
                             np.deg2rad([-225, 51]),np.deg2rad([-200, 200]),
                             np.deg2rad([-135, 135]),np.deg2rad([-360, 360])]},
        'CytonGamma300': {'model': CytonGamma300, 'pillar_height': 0.8, 
                    'qlim': [np.deg2rad([-360, 360]),np.deg2rad([-360, 360]),
                             np.deg2rad([-360, 360]),np.deg2rad([-360, 360]),
                             np.deg2rad([-360, 360]),np.deg2rad([-360, 360])]}
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
        if hasattr(robot, "add_to_env"):
            robot.add_to_env(env)
        else:
            env.add(robot)
        robots[robot_name] = robot

    controlled_robot[0] = robots[robot_name]
    global q
    q = controlled_robot[0].q.copy()
    qlim = controlled_robot[0].qlim.tolist()
    return jsonify(success=True, q=controlled_robot[0].q.tolist(), qlim=qlim)

@app.route('/get_joint_values', methods=['GET'])
def get_joint_values():
    if controlled_robot[0] is not None:
        return jsonify(q=controlled_robot[0].q.tolist())
    return jsonify(q=None), 404

@app.route('/update_joint', methods=['POST'])
def update_joint():
    global q
    data = request.get_json()
    idx, value = data['index'], data['value']
    robot = controlled_robot[0]
    if robot is None:
        return jsonify(success=False, error="No robot selected")
    robot.q[idx] = value
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

@app.route("/reset_env", methods=["POST"])
def reset_env():
    global env, robots, controlled_robot, q, teach_active

    with lock:
        env.close()
        env = Swift()
        env.launch(realtime=True)
        env.set_camera_pose(position=[0.5, 3, 0.8], look_at=[0, 0, 0])
        robots.clear()
        controlled_robot[0] = None
        q = None
        teach_active = True
        threading.Thread(target=run_swift_loop, daemon=True).start()

    print("🔄 Environment reset and restarted successfully")
    return jsonify(success=True)


@app.route("/spawn_obstacle", methods=["POST"])
def spawn_obstacle():

    frybot_collision.move_obstacle([0.3, 1.8, 0.5875])
    jack_collision.move_obstacle([0, 0.5, 0.8])
    drinkbot_collision.move_obstacle([-0.45,-2,0.625])

    print("Prism spawned in workspace")
    return jsonify(success=True)


@app.route("/move_obstacle_far", methods=["POST"])
def move_obstacle_far():
    
    frybot_collision.move_far()
    jack_collision.move_far()
    drinkbot_collision.move_far()

    frybot.collision_event.set()
    jack.collision_event.set()
    drinkbot.collision_event.set()

    print("Prism moved far away")
    return jsonify(success=True)

# ---------------------- 3️⃣ SWIFT LOOP ----------------------
def run_swift_loop():
    global teach_active
    while teach_active:
        if controlled_robot[0] is not None and q is not None:
            controlled_robot[0].q = q
            env.step(0.02)
        time.sleep(0.05)

# ---------------------- 4️⃣ ROBOT OPERATIONS ----------------------
def run_frybot_operations():
    frybot.frybot_operation()

def run_jack_operations():
    jack.stack_burger_into_box()
    jack.animate_lid_closing()
    jack.move_old_parts()
    jack.move_closed_box()
    jack.return_robot_home()

def run_drinkbot_operations():
    drinkbot.animate()

def linearUR3_Operations():
    # linear_operator = LinearUR3Operator(env_builder)
    target_frybox = (-1.4, 0.0, 0.48, 0, 0, np.pi)  # (x, y, z, roll, pitch, yaw)
    LinearUR3.move_object_from_side(env_builder.fry_box_full, target_frybox, steps=120)
    target_burgerbox = (-1.4, 0.4, 0.53, 0, 0, np.pi)
    LinearUR3.move_object_from_side(jack.closed_box, target_burgerbox, steps=120)
    target_drink = (-1.4, 0.8, 0.5, 0, 0, np.pi)
    LinearUR3.move_object_from_side(drinkbot.whole_drink, target_drink, steps=120)

def arduinoEstop():
    serial = SerialCom(9600)
    while True:
        recieve = serial.read()
        if recieve == "ESTOP" or "LIGHT CURTAIN":
            estop_event.clear()
            print("🛑 E-STOP from Arduino ACTIVATED")

def flaskStart():
    app.run(debug=False)

# ---------------------- 5️⃣ MAIN ----------------------
if __name__ == "__main__":
    # Start Swift stepping loop
    threading.Thread(target=run_swift_loop, daemon=True).start()

    flask_thread = threading.Thread(target=flaskStart, daemon=True).start()

    time.sleep(2)
    
    arduino_thread = threading.Thread(target=arduinoEstop, daemon=True).start()
    
    time.sleep(2)
    input("Press Enter to start operations...")
    # Create threads for robots
    frybot_thread = threading.Thread(target=run_frybot_operations)
    jack_thread = threading.Thread(target=run_jack_operations)
    drinkbot_thread = threading.Thread(target=run_drinkbot_operations)

    # Start all
    frybot_thread.start()
    jack_thread.start()
    drinkbot_thread.start()

    # Wait for all robots to finish before running LinearUR3
    frybot_thread.join()
    jack_thread.join()
    drinkbot_thread.join()

    # Now run Linear UR3 after all are done
    linearUR3_Operations()