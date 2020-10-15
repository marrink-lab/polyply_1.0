# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from numpy import sqrt, pi, cos, sin, dot, cross, arccos, degrees
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def vector_angle_degrees(v1, v2):
    """
    Compute the angle between two vectors
    in degrees and between 0 180 degrees.

    Parameters
    ----------
    v1: np.array
    v2: np.array

    Returns
    ---------
    float
      angle in degrees
    """
    angle = degrees(arccos(dot(u_vect(v1), u_vect(v2))))
    return angle

@jit(nopython=True, cache=True, fastmath=True)
def u_vect(vect):
    """
    Compute unit vector of vector.

    Parameters
    ----------
    vect: np.array

    Returns
    -----------
    np.array
    """
    u_vect = vect/norm(vect)
    return u_vect

@jit(nopython=True, cache=True, fastmath=True)
def angle(A, B, C):
    """
    Compute angle between three points,
    where `B` is the center of the angle.

    Paramters
    ---------
    A, B, C:  numpy.array

    Returns
    ---------
    float
         angle in degrees
    """
    v1 = B - A
    v2 = B - C
    angle = vector_angle_degrees(v1, v2)
    return angle

@jit(nopython=True, cache=True, fastmath=True)
def dih(A, B, C, D):
    """
    Compute dihedral angle between four points,
    where `B` and `C` are in the center.

    Paramters
    ---------
    A, B, C, D:  numpy.array

    Returns
    ---------
    float
         angle in degrees
    """
    r1 = A - B
    r2 = B - C
    r3 = C - D
    n1 = u_vect(cross(r1, r2))
    n2 = u_vect(cross(r2, r3))
    dih = vector_angle_degrees(n1, n2)
    return dih

def center_of_geometry(points):
    """
    Compute center of geometry.

    Paramters
    ---------
    points:  numpy.array

    Returns
    ---------
    float
    """
    COG = np.average(points, axis=0)
    return COG

def norm_sphere(values=50):
    """
    Generate unit vectors on a
    sphere. Note the unit vectors
    cover the sphere with equal probablility
    around the sphere.

    For more information see:
    https://stackoverflow.com/questions/33976911/generate-a-
    random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere

    https://mathworld.wolfram.com/SpherePointPicking.html

    Paramters
    ---------
    values: int
        number of vectors to generate

    Returns
    ---------
    np.array shape (values, 3)
    """
    v_sphere = np.random.normal(0.0, 1, (values, 3))
    return np.array([u_vect(vect) for vect in v_sphere])

@jit(nopython=True, cache=True, fastmath=True)
def radius_of_gyration(points):
    """
    Compute radius of gyration of points.

    Parameters
    ---------
    points: np.array

    Returns
    ---------
    float
         radius of gyration
    """
    N = len(points)
    diff=np.zeros((N**2))
    count=0
    for i in points:
        for j in points:
            diff[count]=np.dot((i - j),(i-j))
            count = count + 1
    RG = np.sqrt(1/(2*N**2.0) * np.sum(diff))
    return RG

@jit(nopython=True, cache=True, fastmath=True)
def multiply_matrix(A, B):
    m, n = A.shape
    p = B.shape[1]

    C = np.zeros((m,p))

    for i in range(0,m):
        for j in range(0,p):
            for k in range(0,n):
                C[i,j] += A[i,k]*B[k,j]
    return C

def rotate_xyz(object_xyz, theta_x, theta_y, theta_z):
    """
    Rotate `object_xyz` around angles `theta_x`, `theta_y`
    and `theta_z` around the x,y,z axis respectively and
    return coordinates of the rotated object. Note
    object_xyz needs to be in column vector format i.e.
    of numpy shape (3, N).

    Parameters:
    ----------
    object_xyz: numpy.ndarray
        coordinates of the object
    thata_x: float
    thata_y: float
    thata_z: float
        angles in degrees
    """
    rotation = Rotation.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=True)
    rotated_object = multiply_matrix(rotation.as_matrix(), object_xyz)
    #rotated_object = np.matmul(rotation.as_matrix(), object_xyz)
    return rotated_object
