import numpy as np
import roboticstoolbox as rtb
import swift
import spatialmath as sm
import os

def Cyton_Gamma_300():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "cyton_gamma_300_description\cyton_gamma_300.urdf")

    rob = rtb.ERobot.URDF(urdf_path)
    print(rob)
    return rob

if __name__ == "__main__":
    
    env = swift.Swift()
    env.launch(realtime=True)

    rob = Cyton_Gamma_300()

    env.add(rob)
    rob.q = np.zeros(rob.n)
    env.step(0.05)

    input("enter")