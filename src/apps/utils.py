import math

def length(v):
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

def normalize(v):
    l = length(v)
    return [v[0]/l, v[1]/l, v[2]/l]

def dot(v0, v1):
    return v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2]

def cross(v0, v1):
    return [v0[1]*v1[2]-v1[1]*v0[2],
            v0[2]*v1[0]-v1[2]*v0[0],
            v0[0]*v1[1]-v1[0]*v0[1]]

def uvToPoint(uv, radius, theta_offset):
    '''
    converts UV point to 3D spherical coordinate
    :param uv: A tuple of (u, v) 2D position
    :param radius: Spherical radius (rho)
    :param theta_offset: Latitude rotation offset in radians
    :return: A tuple (x, y, z) coordinate
    '''
    theta = 2*math.pi * uv[0] + theta_offset;
    phi = math.pi * uv[1];

    x = math.cos(theta) * math.sin(phi) * radius;
    y = math.cos(phi) * radius;
    z = math.sin(theta) * math.sin(phi) * radius;

    return x, y, z

def lookAtMatrix(eye, target, up):
    '''
    Lookat Matrix (aim/target)
    :param eye: Source point
    :param target: Target point
    :param up: Up vector
    :return: Transformation matrix
    '''
    mz = normalize( (eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]) ) # inverse line of sight
    mx = normalize( cross( up, mz ) )
    my = normalize( cross( mz, mx ) )
    tx =  dot( mx, eye )
    ty =  dot( my, eye )
    tz = -dot( mz, eye )

    return [[mx[0], my[0], mz[0], 0],
            [mx[1], my[1], mz[1], 0],
            [mx[2], my[2], mz[2], 0],
            [tx, ty, tz, 1]]

def matrixToRotation(R) :
    '''
    Converts transformation matrix to rotation
    :param R: Matrix
    :return: Rotation in degrees as a tuple (x, y, z)
    '''
    sy = math.sqrt(R[0][0] * R[0][0] +  R[1][0] * R[1][0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2][1] , R[2][2])
        y = math.atan2(-R[2][0], sy)
        z = math.atan2(R[1][0], R[0][0])
    else :
        x = math.atan2(-R[1][2], R[1][1])
        y = math.atan2(-R[2][0], sy)
        z = 0

    return math.degrees(x), math.degrees(y), math.degrees(z)
