import numpy as np
from numpy import sqrt, pi, cos, sin, dot, cross, arccos, degrees
from numpy.linalg import norm

def u_vect(vect):
    return(vect/norm(vect))

def angle(A, B, C):
    v1 = B - A
    v2 = B - C
    return(degrees(arccos(np.clip(dot(u_vect(v1), u_vect(v2)), -1.0, 1.0))))

def dih(A, B, C, D):
    r1 = A - B
    r2 = B - C
    r3 = C - D
    n1 = cross(r1, r2)
    n2 = cross(r2, r3)
    return(degrees(arccos(np.clip(dot(u_vect(n1), u_vect(n2)), -1.0, 1.0))))


def geometrical_center(coord):
    return(sum(coord)/float(len(coord)))

def norm_sphere(values=50):
    v_sphere = np.random.normal(0.0, 1, (values,3))
    return(np.array([ u_vect(vect) for vect in v_sphere]))

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def _align(object3d, ref_vector, target_vector):
     mat = rotation_matrix_from_vectors(ref_vector, target_vector)
     rotated_object = []

     for vector in object3d:
         rotated_object.append(mat.dot(vector))

     return rotated_object


def _find_tangent(point_1, point_2, point_3):

    v1 = u_vect(point_2 - point_1)
    v2 = u_vect(point_2 - point_3)

    normal = np.cross(v1,v2)
    ang = angle(v1, v2)
    ang1 = ang + (180 - ang)/2.
    ang2 = (180 - ang)/2.

    A = np.array([[N[0],N[1],N[2]],
              [v1[0],v2[1],v1[2]],
              [v2[0],v2[1],v2[2]]])

    B = np.array([0,np.sin(ang1),np.sin(ang2)])

    v =  u_vect(np.dot(np.transpose(A),B))

    def tangent(length):
        point = length * (v - point_2) + point_2
        return point

    return tangent

