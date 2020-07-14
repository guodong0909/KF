import numpy as np
import math

def isRotationMatrix(R_x) :
    Rt = np.transpose(R_x)
    shouldBeIdentity = np.dot(Rt, R_x)
    I = np.identity(3, dtype = R_x.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-4
 
 
# Calculates rotation matrix to euler angles 
# The result is the same as MATLAB except the order of the euler angles ( x and z are swapped ).

def rotationMatrixToEulerAngles(R_x) :

    assert(isRotationMatrix(R_x))
     
    sy = math.sqrt(R_x[0,0] * R_x[0,0] +  R_x[1,0] * R_x[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R_x[2,1] , R_x[2,2])
        y = math.atan2(-R_x[2,0], sy)
        z = math.atan2(R_x[1,0], R_x[0,0])
    else :
        x = math.atan2(-R_x[1,2], R_x[1,1])
        y = math.atan2(-R_x[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def EulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R
