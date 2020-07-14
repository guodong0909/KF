#!/usr/bin/python
# coding:utf8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.rotation2euler import rotationMatrixToEulerAngles, EulerAnglesToRotationMatrix
from lib.error_cal import add, re, te

# This code is used on YCB-dataset, if want to change to other dataset, please change the path and the details.
# written by Yu Su from EPFL(2020.5.20) if you have any problem, please contact suyuathit@gmail.com
# This code use KF funtion from opencv


# Before using, please define two funtion to read real R&T and give xyz points of the module
# parameters name with 'measurement' is the result before using KF 
# parameters name with 'result' is the result after using KF
# parameters name with 'real' is the groundtruth
# final result is saved in ./data/result


'''
xyz = xyz(Nx3)
R = read_R(groundtruth,i)
T = read_R(groundtruth,i)

'''

def read_measurement(k,i):
    file = open('./data/measurement result/00{:02d}_00{:04d}-color.txt'.format(k,i),'r') # pose estimation result to use KF (3x4)
    cont = file.readlines()
    RaT = np.zeros([3,4])
    RaT[0] = cont[0].split()
    RaT[1] = cont[1].split()
    RaT[2] = cont[2].split()
    R = RaT[:,0:3]
    T = RaT[:,3]
    Eu = rotationMatrixToEulerAngles(R)
    x=np.float(T[0])
    y=np.float(T[1])
    z=np.float(T[2])
    a=np.float(Eu[0])
    b=np.float(Eu[1])
    c=np.float(Eu[2])
    f = np.array([x,y,z,0,0,0,0,0,0,a,b,c,0,0,0,0,0,0],dtype=np.float32)
    return f


global last_measurement, current_measurement, last_predicition, current_prediction

add_error_list = []
add_error_measurement_list = []
R_error_list = []
R_error_measurement_list = []
Eular_error_list = []
Eular_error_measurement_list = []
T_resultlist = []
R_resultlist = []

test_number = 50 #imput test_file number(48--59) of YCB-dataset
model_No = 1  #which model in the picture

obj_p = np.array(xyz,dtype=np.float32)

R = read_cam_R(1)
T = read_cam_t(1) 

kalman = cv2.KalmanFilter(18, 18)


last_measurement = current_measurement  = last_predicition = current_prediction =  np.array(read_measurement(test_number,1)).reshape((18,1))


measureM = np.eye(18)
transitionM = np.eye(18)
mnoiseCov = np.eye(18)
processnoise = np.eye(18)

dt = np.array(1/30,dtype=np.float32)

measureM[0,3] = dt
measureM[1,4] = dt
measureM[2,5] = dt
measureM[3,6] = dt
measureM[4,7] = dt
measureM[5,8] = dt
measureM[9,12] = dt
measureM[10,13] = dt
measureM[11,14] = dt
measureM[12,15] = dt
measureM[13,16] = dt
measureM[14,17] = dt
measureM[0,6] = (1/2)*(np.square(dt))
measureM[1,7] = (1/2)*(np.square(dt))
measureM[2,8] = (1/2)*(np.square(dt))
measureM[9,15] = (1/2)*(np.square(dt))
measureM[10,16] = (1/2)*(np.square(dt))
measureM[11,17] = (1/2)*(np.square(dt))

#transitionM[0,0] = transitionM[1,1] = transitionM[2,2] = transitionM[3,9] = transitionM[4,10] = transitionM[5,11] = 1

kalman.measurementMatrix = np.array(transitionM, dtype = np.float32) 

kalman.transitionMatrix = np.array(measureM, dtype = np.float32)

kalman.processNoiseCov = np.array(processnoise, dtype = np.float32)  * 0.0001

kalman.measurementNoiseCov = np.array(mnoiseCov,dtype = np.float32)  * 0.01

for i in range(1,800):
    current_measurement = np.array(read_measurement(test_number,j)).reshape((18,1))
    kalman.correct(current_measurement)
    current_prediction = kalman.predict()
    result = np.array(current_prediction).reshape((18,1))

    T_measurement = np.array((current_measurement[0],current_measurement[1],current_measurement[2]),dtype=np.float32)*1000
    Eular_measurement = np.array((current_measurement[9],current_measurement[10],current_measurement[11]),dtype=np.float32)
    R_measurement =  np.array(EulerAnglesToRotationMatrix(theta_yinlin),dtype=np.float32)

    T_result = np.array((current_prediction[0],current_prediction[1],current_prediction[2]),dtype=np.float32).reshape((1,3))*1000
    Eular_result = np.array((current_prediction[9],current_prediction[10],current_prediction[11]),dtype=np.float32)
    R_result = np.array(EulerAnglesToRotationMatrix(theta_result),dtype=np.float32)

    T_resultlist.append(T_result)
    R_resultlist.append(R_result)


    T_real = np.array(read_cam_t(test_number,model_No,i),dtype=np.float32).reshape(-1,3)
    R_real = np.array(read_cam_R(test_number,model_No,i),dtype=np.float32).reshape(3,3)


    # caculate error

    add_error = add(R_result,T_result,R_real,T_real,obj_p)
    add_error_measurement = add(R_measurement,T_measurement,R_real,T_real,obj_p)
    add_error_list.append(add)
    add_error_measurement.append(add_measurement)

    R_error = re(R_result,R_real)
    R_error_measurement = re(R_measurement,R_real)
    R_error_list.append(R_error)
    R_error_measurement_list.append(R_error_measurement)

    Eular_real = np.array(rotationMatrixToEulerAngles(R_real),dtype=np.float32).reshape((3,1))
    Eular_error = (theta_result - theta_real)*180/(np.pi)
    Eular_error_measurement = (theta_measurement - theta_real)*180/(np.pi)
    Eular_error_list.append(theta_error)
    Eular_error_measurement_list.append(theta_error_measurement)


np.savetxt('./data/result/add.txt',add_error_list)
np.savetxt('./data/result/R_result.txt',T_resultlist)
np.savetxt('./data/result/T_result.txt',R_resultlist)


# then you can plot the result after using KF 


