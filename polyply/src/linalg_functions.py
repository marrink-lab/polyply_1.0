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
    return degrees(arccos(dot(u_vect(v1), u_vect(v2))))


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
    return vect/norm(vect)


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
    return vector_angle_degrees(v1, v2)


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
    return vector_angle_degrees(n1, n2)


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
    return np.average(points, axis=0)


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
    diff = np.zeros((N**2))
    count = 0
    for i in points:
        for j in points:
            diff[count] = np.dot((i - j), (i-j))
            count = count + 1
    return np.sqrt(1/(2*N**2.0) * sum(diff))


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
    rotation = Rotation.from_euler(
        'xyz', [theta_x, theta_y, theta_z], degrees=True)
    rotated_object = np.matmul(rotation.as_matrix(), object_xyz)
    return rotated_object


def plane_dist(normal, plane_point, point):
    v = point - plane_point
    dist = np.dot(v, normal)
    return dist


def projection(center, atoms):
    """
    project points B-D,center on a plane normal to the vector center-A
    and centered at point 'center', this works for 3D

    Parameters:
    -------------------------------
    center: numpy ndarray with 3 coords
    A,B,C,D: numpy ndarrays with 3 coords

    Procedure:
    -------------------------------
    1. compute normal as center A
    2. compute vector of point and center
    3. compute norm of that vector
    4. substract point vector from distance multupiled by unit normal vector

    Returns:
    ---------------------------------
    projected points list of numpy ndarrays
    """
    A, B, C, D = atoms
    normal = u_vect(center - A)
    proj_points = []

    for point in [B, C, D, center]:
        v = point - center
        dist = np.dot(v, normal)
        new_point = point - dist * normal
        proj_points.append(new_point)

    proj_points.append(normal)

    return proj_points


def signed_angle(v1, v2, n):
    """
    Copmute signed angle between vectors v1 and v2 in 3D along
    a view direction taken from:
    https://math.stackexchange.com/questions/2140504/how-to-calculate-signed-angle-between-two-vectors-in-3d 

    Parameters
    ---------------------------
    v1,v2,n: numpy ndarrays
    Returns:
    -----------------------------------
    singed angle in radians
    """

    sin_theta = np.dot(np.cross(n, v1), v2) / (norm(v1)*norm(v2))
    cos_theta = np.dot(v1, v2) / (norm(v1)*norm(v2))

    if sin_theta >= 0:
        theta = np.arccos(cos_theta)
    else:
        theta = 2 * np.pi - np.arccos(cos_theta)
    return theta


def which_chirality(atoms):
    """
    determine the chirality of center with four
    substituents A,B,C,D.

    Paramteres:
    -----------------------------------
    A,B,C,D: numpy ndarrays of 3 coords
    Note that the molecular weight must be A<B<C<D

    Returns:
    -----------------------------------
    singed angle
    """
    center = atoms[0]
    a, b, c, origin_proj, normal = projection(center, atoms[1:])
    aO = a - origin_proj
    bc = b - c
    ang = signed_angle(aO, bc, normal)

    return np.sin(ang)

def mirror_coordinates(plane_point, normal, coords):
    """
    Mirror coordinates across plane defined by normal
    and plane point.
    """
    dists = plane_dist(normal, plane_point, coords)
    new_coords = coords - 2 * dists.reshape(len(dists), 1) * normal
    return new_coords

def reconstruct_H(center,j, k, l, ang=109.5):
    """
    reconstruct tetrahedral hydrogens for UA FFs.
    Parameters:
    -----------------------------------
    center, j,k,l : numpy ndarray with 3 coords
    ang: tetrahedral angle in degree, default 109.5

    Returns:
    -----------------------------------
    position H: numpy ndarray
    """

    jc = j-center
    kc = k-center
    lc = l-center

    A = np.array([[jc[0],jc[1],jc[2]],[kc[0],kc[1],kc[2]],[lc[0],lc[1],lc[2]]])
    B = np.array([np.sin(ang),np.sin(ang),np.sin(ang)])
    v = np.dot(np.transpose(A),B)
    return v + center
