#############################################################################################################################################
#############################################################################################################################################
###################################################  QUATERNION HELPER FUNCTIONS  ###########################################################
#############################################################################################################################################
#############################################################################################################################################
"""quaternion_operations module."""
import numpy as np


def quaternion_mult(q1,q2):
    """Documentation for the function.
:param q1: Value for q1.
:type q1: Any
:param q2: Value for q2.
:type q2: Any
:return: Return value.
:rtype: Any"""
    return np.array([q1[3]*q2[0]+q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1],
            q1[3]*q2[1]+q1[1]*q2[3]-q1[0]*q2[2]+q1[2]*q2[0],
            q1[3]*q2[2]+q1[2]*q2[3]+q1[0]*q2[1]-q1[1]*q2[0],
            q1[3]*q2[3]-q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]])

def quaternion_conjugate(q): 
    """Documentation for the function.
:param q: Value for q.
:type q: Any
:return: Return value.
:rtype: Any"""

    return np.array([-q[0],-q[1],-q[2],q[3]])

def quaternion_normalize(q):
    """Documentation for the function.
:param q: Value for q.
:type q: Any
:return: Return value.
:rtype: Any"""
    norm = np.linalg.norm(q)
    return q/norm

def quaternion_to_matrix(q):
    """Documentation for the function.
:param q: Value for q.
:type q: Any
:return: Return value.
:rtype: Any"""
    return np.array([
        [1 - 2 * (q[1]**2 + q[2]**2),     2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[0] * q[2] + q[3] * q[1]), 0],
        [2 * (q[0] * q[1] + q[3] * q[2]), 1 - 2 * (q[0]**2 + q[2]**2),     2 * (q[1] * q[2] - q[3] * q[0]), 0],
        [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * (q[0]**2 + q[1]**2),     0],
        [0,                               0,                               0,                               1]
    ])
