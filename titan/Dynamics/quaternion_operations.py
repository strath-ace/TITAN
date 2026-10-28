#############################################################################################################################################
#############################################################################################################################################
###################################################  QUATERNION HELPER FUNCTIONS  ###########################################################
#############################################################################################################################################
#############################################################################################################################################
"""quaternion_operations module."""
import numpy as np


def quaternion_mult(q1,q2):
    """Multiply two quaternions in vector-scalar component order.

    :param q1: Left quaternion represented as ``[x, y, z, w]``.
    :type q1: Any
    :param q2: Right quaternion represented as ``[x, y, z, w]``.
    :type q2: Any
    :return: Quaternion product in ``[x, y, z, w]`` order.
    :rtype: numpy.ndarray
"""
    return np.array([q1[3]*q2[0]+q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1],
            q1[3]*q2[1]+q1[1]*q2[3]-q1[0]*q2[2]+q1[2]*q2[0],
            q1[3]*q2[2]+q1[2]*q2[3]+q1[0]*q2[1]-q1[1]*q2[0],
            q1[3]*q2[3]-q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]])

def quaternion_conjugate(q): 
    """Return the conjugate of a quaternion in ``[x, y, z, w]`` order.

    :param q: Quaternion to conjugate.
    :type q: Any
    :return: Conjugated quaternion.
    :rtype: numpy.ndarray
"""

    return np.array([-q[0],-q[1],-q[2],q[3]])

def quaternion_normalize(q):
    """Scale a quaternion to unit norm.

    :param q: Quaternion to normalize.
    :type q: Any
    :return: Unit quaternion with the same orientation.
    :rtype: numpy.ndarray
"""
    norm = np.linalg.norm(q)
    return q/norm

def quaternion_to_matrix(q):
    """Convert a quaternion to a homogeneous rotation matrix.

    :param q: Quaternion in ``[x, y, z, w]`` component order.
    :type q: Any
    :return: A 4-by-4 homogeneous rotation matrix.
    :rtype: numpy.ndarray
"""
    return np.array([
        [1 - 2 * (q[1]**2 + q[2]**2),     2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[0] * q[2] + q[3] * q[1]), 0],
        [2 * (q[0] * q[1] + q[3] * q[2]), 1 - 2 * (q[0]**2 + q[2]**2),     2 * (q[1] * q[2] - q[3] * q[0]), 0],
        [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * (q[0]**2 + q[1]**2),     0],
        [0,                               0,                               0,                               1]
    ])
