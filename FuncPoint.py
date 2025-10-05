import numpy as np
import sympy as sp
import roboticstoolbox as rtb

from spatialmath import SE3
from scipy import linalg
from math import pi

def funcPoint(func, var, start, end, step):

    # Takes an explicit Sympy function of x and returns a list of points between the start and end. 

    # func: Sympy Function. i.e. f(x)
    # var: Input variable. i.e. x. can only be one
    # Start: Start of sampled domain
    # End: End of sampled domain
    # Step: Interval between samples

    f = sp.lambdify(var, func, modules='numpy')     # Creates Sympy function
    x = np.linspace(start, end, step )              # Creates array of all x values
    y = f(x)                                        # Computes all y values

    p = np.column_stack((x,y))                      # Creates array of sampled points. [Rows: Points | Columns: x,y]

    return p                                        # returns array of points

def pointTransform(x=None,y=None,z=None):

    # Takes array of x, y and z values and returns a list of SE3 Transform of points. 
    # If no argument is given for a value it defaults to 0, 
    # If a scalar is given for a value all points will use that scalar
    # probably useless function

    x_arr = np.atleast_1d(0.0 if x is None else x)                  # Check if argument is given. If not then defaults to zero
    y_arr = np.atleast_1d(0.0 if y is None else y)
    z_arr = np.atleast_1d(0.0 if z is None else z)

    x_b, y_b, z_b = np.broadcast_arrays(x_arr, y_arr, z_arr)        # Broadcast arrays to ensure equal size

    T = []                                                          # List for SE3 Transforms
    for i in range(len(x_b)):
        T.append(SE3(x_b[i],y_b[i],z_b[i]))                         # Append SE3 Transforms of x, y and z to T
    return T                                                        # Return list fo SE3 transforms

def RMRC(rob,q,points,t=5,steps=100,):

    delta_t = t/steps
    min_manip_measure = 0.1
    q_matrix = [q]

    for i in range(steps-1):
        xdot = (points[i+1,:] - points[i,:]) / delta_t
        J = rob.jacob0(q_matrix[i])
        J = J[:2,:]
        m = np.sqrt(linalg.det(J @ J.T))

        ep = 0.015
        
        if m < min_manip_measure:
            lmda = (1 - (m/ep)**2)*0.015
            J_lds = J.T @ linalg.inv((J @ J.T + 0.01 * np.eye(2)))
            qdot = J_lds @ xdot
        else: qdot = linalg.inv(J) @ xdot
        
        q_matrix.append(q_matrix[i] + delta_t * qdot)
    return q_matrix

def funcPoint_Demo():

    x = sp.symbols('x')                             # Create Sympy Symbol
    f = sp.cos(x)                                   # Create Sympy Function

    points = funcPoint(f, x, 0, 2*pi, 100)          # Compute points using funcPoint function

    for i in range(len(points)):
        print(points[i,:])                          # Print

def pointTransform_Demo():

    x = sp.symbols('x')                             # Create Sympy Symbol
    f = sp.cos(x)                                   # Create Sympy Function

    p = funcPoint(f, x, 0, 2*pi, 100)               # Compute points using funcPoint function

    T = pointTransform(x=p[:,0],y=p[:,1])           # Create list of SE3 Transforms using pointTransform Function
    
    for i in range(len(T)):
        print(T[i])                                 # Print

def RMRC_Demo():
    pass


if __name__ == "__main__":

    funcPoint_Demo()
    input("Press Enter for next demo")

    pointTransform_Demo()
    input("Press Enter for next demo")

    RMRC_Demo()
    input("Press Enter for next demo")