import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from numpy.random import randn


def kalman_main(mem):class Kalman:
    def __init__(self, dt, obs, mod):
        self.dt = dt

        self.KF = KalmanFilter(dim_x=4, dim_z=2)
        self.KF.F = np.array([[1, dt, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, dt],
                              [0, 0, 0, 1]], dtype=float)
        self.KF.H = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0]])
        self.KF.P *= 100  # initial uncertainty
        self.z_std = obs
        self.KF.R = np.diag([self.z_std ** 2, self.z_std ** 2])  # 1 standard
        self.KF.Q = np.matrix(np.eye(4)) * mod

    def predict(self):
        return self.KF.predict()

    def update(self, z):
        return self.KF.update(z)

    def read(self):
        return np.array(self.KF.x).flatten()

if __name__ == '__main__':
    kf = Kalman(0.1, 10)
    zs = np.array([[i + randn(), i + randn()] for i in range(50)])  # measurements
    zs_result = []
    for z in zs:
        kf.predict()
        kf.update(z)
        zs_result.append(kf.read())
    zs_result = np.array(zs_result)
    plt.plot(zs[:, 0], zs[:, 1], 'ro')
    plt.plot(zs_result[:, 0], zs_result[:, 2], 'g-')
    plt.show()
