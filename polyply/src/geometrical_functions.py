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
