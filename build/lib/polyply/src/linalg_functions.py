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
from polyply import jit

def _vector_angle_degrees(v1, v2):
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

# this is the numba implementation
vector_angle_degrees = jit(_vector_angle_degrees)

def _u_vect(vect):
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

# this is the numba implementation
u_vect = jit(_u_vect)

def _angle(A, B, C):
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
angle = jit(_angle)

def _dih(A, B, C, D):
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
    r2 = C - B
    r3 = C - D
    cross1 = cross(r1, r2)
    cross2 = cross(r2, r3)
    n1 = u_vect(cross1)
    n2 = u_vect(cross2)
    dih = vector_angle_degrees(n1, n2)
    # GROMACS specific definition of the sign of the
    # dihedral.
    if dot(r1, cross2) < 0:
        dih = - dih
    return dih
dih = jit(_dih)

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

def _radius_of_gyration(points):
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
radius_of_gyration = jit(_radius_of_gyration)

def _matrix_multiplication(*args):
   """
   Implements matrix multiplicaiton of numpy ndarrays.
   Basically the same as np.matmul but it can handle 1-D
   column vectors. In addition together with numba a
   modest speed-up is expected. See the following for
   some testing https://stackoverflow.com/questions/36526708/\
   comparing-python-numpy-numba-and-c-for-matrix-multiplication
   Also note that the data type of the matrices needs to be
   explicitly set and is assumed to be np.float64.

   Parameters
   ----------
   args: list[np.array]
       a list of numpy ndarrays
   """
   matrix_a = args[0]
   for matrix_b in args[1:]:
       rows, other_dim = matrix_a.shape
       columns = matrix_b.shape[1]
       new_matrix = np.zeros((rows, columns))
       for i in range(0, rows):
           for j in range(0, columns):
               for k in range(0, other_dim):
                   new_matrix[i,j] += matrix_a[i,k] * matrix_b[k,j]

       matrix_a = new_matrix
   return new_matrix

matrix_multiplication = jit(_matrix_multiplication)

def _rotate_xyz(object_xyz, theta_x, theta_y, theta_z):
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
    # convert all angles to degree
    #theta_z, theta_y, theta_x = np.deg2rad(theta_z), np.deg2rad(theta_y), np.deg2rad(theta_x)
    # compute rotatin matrix around z
    sin_z = np.sin(theta_z)
    cos_z = np.cos(theta_z)
    rot_z = np.array([[cos_z, -1.0*sin_z, 0.0], [sin_z, cos_z , 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # compute rotation matrix around y
    cos_y = np.cos(theta_y)
    sin_y = np.sin(theta_y)
    rot_y = np.array([[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]], dtype=np.float64)
    # compute rotation matrix around x
    cos_x = np.cos(theta_x)
    sin_x = np.sin(theta_x)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]], dtype=np.float64)
    # consecutively multiply all matrices and the object matrix
    rotated_object = matrix_multiplication(rot_z, rot_y, rot_x, object_xyz)
    return rotated_object

rotate_xyz = jit(_rotate_xyz)
