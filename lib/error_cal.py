import numpy as np 
import matplotlib.pyplot as plt 
import math
import os
from scipy.linalg import logm
import numpy.linalg as LA

def re(R_est, R_gt):
    assert (R_est.shape == R_gt.shape == (3, 3))
    temp = logm(np.dot(np.transpose(R_est), R_gt))
    rd_rad = LA.norm(temp, 'fro') / np.sqrt(2)
    rd_deg = rd_rad / np.pi * 180
    return rd_deg

def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert (t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt.reshape(3) - t_est.reshape(3))
    return error

def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert (pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def transform_pts_Rt_2d(pts, R, t, K):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :param K: 3x3 intrinsic matrix
    :return: nx2 ndarray with transformed 2D points.
    """
    assert (pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))  # 3xn
    pts_c_t = K.dot(pts_t)
    n = pts.shape[0]
    pts_2d = np.zeros((n, 2))
    pts_2d[:, 0] = pts_c_t[0, :] / pts_c_t[2, :]
    pts_2d[:, 1] = pts_c_t[1, :] / pts_c_t[2, :]

    return pts_2d

def add(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e