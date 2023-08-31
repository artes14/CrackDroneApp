import csv
import numpy as np
from numpy.linalg import inv

import os
from pathlib import Path
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from datetime import time

filename = '../location/20230508/log_291_2023-3-2-19-21-40_ekf_jh_0.csv'

class logData:
    def __init__(self, fileName: str = None):
        self.z_meas=[]
        self.timestamp = []
        self.d_forward, self.d_uwb1, self.d_uwb2, self.d_down, self.vel_y = 0,0,0,0,0

        self.time_end = 10
        self.dt=0.01
        self.time = np.arange(0, self.time_end, self.dt)
        self.n_samples = len(self.time)
        self.A=np.eye(4) + self.dt * np.array([[0, 0, 0, 0],
                                               [0, 0, 0, 1],
                                               [0, 0, 0, 0],
                                               [0, 0, 0, 0]])
        self.H=np.zeros((4,5))
        self.Q = np.array([[0.01, 0, 0, 0],
                           [0, 0.0001, 0, 0],
                           [0, 0, 0.0005, 0],
                           [0, 0, 0, 0.1]])
        self.R = np.array([[0.0002, 0, 0, 0, 0],
                           [0, 0.001, 0, 0, 0],
                           [0, 0, 0.002, 0, 0],
                           [0, 0, 0, 0.012, 0],
                           [0, 0, 0, 0, 10]])
        self.P = 10*np.eye(4)

        # Initialization for estimation.
        self.x_0 = np.array([-0.8, 1.3, -1.0, 0])

        self.p_uwb_1 = np.array([-5,0,-1.27])
        self.p_uwb_2 = np.array([-5,3,-1.27])

        self.z_offset=-0.2
        self.y_offset=0

        self.pos_x_saved=np.zeros(self.n_samples)
        self.pos_y_saved=np.zeros(self.n_samples)
        self.pos_z_saved=np.zeros(self.n_samples)
        self.pos_v_saved=np.zeros(self.n_samples)

        if fileName is not None:
            with open(fileName) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count != 0:
                        self.z_meas.append([(-1)*float(row[5]), float(row[27]), float(row[28]), (-1) * float(row[26]), float(row[13])])
                    line_count += 1

    def GetData(self, t):
        if len(self.z_meas)==0:
            return None
        else:
            return self.z_meas[t]

    # Jacobian calculation
    def Ajacob_at(self, x_esti):
        return self.A

    def Hjacob_at(self, x_pred):
        x_uwb_1 = self.p_uwb_1[0]
        y_uwb_1 = self.p_uwb_1[1]
        z_uwb_1 = self.p_uwb_1[2]

        x_uwb_2 = self.p_uwb_2[0]
        y_uwb_2 = self.p_uwb_2[1]
        z_uwb_2 = self.p_uwb_2[2]

        norm1 = np.sqrt((x_pred[0]-x_uwb_1)**2+(x_pred[1]-y_uwb_1)**2+(x_pred[2]-z_uwb_1)**2)
        norm2 = np.sqrt((x_pred[0]-x_uwb_2)**2+(x_pred[1]-y_uwb_2)**2+(x_pred[2]-z_uwb_2)**2)

        H=np.array([[1, 0, 0, 0],
                    [(x_pred[0]-x_uwb_1)/norm1, (x_pred[1]-y_uwb_1)/norm1, (x_pred[2]-z_uwb_1)/norm1, 0],
                    [(x_pred[0]-x_uwb_2)/norm2, (x_pred[1]-y_uwb_2)/norm2, (x_pred[2]-z_uwb_2)/norm2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        return H

    # non-linear system calculation (측정모델)
    def fx(self, x_esti):
        return self.A @ x_esti

    def hx(self, x_pred):
        x_uwb_1 = self.p_uwb_1[0]
        y_uwb_1 = self.p_uwb_1[1]
        z_uwb_1 = self.p_uwb_1[2]

        x_uwb_2 = self.p_uwb_2[0]
        y_uwb_2 = self.p_uwb_2[1]
        z_uwb_2 = self.p_uwb_2[2]

        norm1 = np.sqrt((x_pred[0]-x_uwb_1)**2+(x_pred[1]-y_uwb_1)**2+(x_pred[2]-z_uwb_1)**2)
        norm2 = np.sqrt((x_pred[0]-x_uwb_2)**2+(x_pred[1]-y_uwb_2)**2+(x_pred[2]-z_uwb_2)**2)
        z_pred = np.array([x_pred[0], norm1, norm2, x_pred[2], x_pred[3]])
        return z_pred

    def EKF(self, z, x_esti):
        """Extended Kalman Filter Algorithm"""
        # (1) Prediction
        self.A=self.Ajacob_at(x_esti)
        x_pred = self.fx(x_esti)
        P_pred = self.A @ self.P @ self.A.T +self.Q

        # (2) Kalman Gain.
        H = self.Hjacob_at(x_pred)
        K = P_pred @ H.T @ inv(H @ P_pred @ H.T + self.R)

        # (3) Estimation.
        x_esti = x_pred + K @ (z - self.hx(x_pred))

        # (4) Error Covariance.
        self.P = P_pred - K @ H @ P_pred

        return x_esti

    def calculate(self, plot=False):
        for i in range(self.n_samples):
            z_meas = self.GetData(i)
            if i == 0:
                x_esti = self.x_0
            else:
                x_esti = self.EKF(z_meas, x_esti)

            self.pos_x_saved[i] = x_esti[0]
            self.pos_y_saved[i] = x_esti[1]
            self.pos_z_saved[i] = x_esti[2]
            self.pos_v_saved[i] = x_esti[3]

        if plot==True:
            fig, axes = plt.subplots(nrows=2, ncols=2)

            axes[0, 0].plot(self.time, self.pos_x_saved, 'b-', label='Estimation (EKF)')
            axes[0, 0].legend(loc='upper left')
            axes[0, 0].set_title('X Axis Position: Esti. (EKF)')
            axes[0, 0].set_xlabel('Time [sec]')
            axes[0, 0].set_ylabel('X Axis Position [m]')

            axes[1, 0].plot(self.time, self.pos_y_saved, 'b-', label='Estimation (EKF)')
            axes[1, 0].legend(loc='upper left')
            axes[1, 0].set_title('Y Axis Position: Esti. (EKF)')
            axes[1, 0].set_xlabel('Time [sec]')
            axes[1, 0].set_ylabel('Y Axis Position [m]')

            axes[0, 1].plot(self.time, self.pos_z_saved, 'b-', label='Estimation (EKF)')
            axes[0, 1].legend(loc='upper left')
            axes[0, 1].set_title('Z Axis Position: Esti. (EKF)')
            axes[0, 1].set_xlabel('Time [sec]')
            axes[0, 1].set_ylabel('Z Axis Position [m]')

            axes[1, 1].plot(self.pos_y_saved[10:], self.pos_z_saved[10:], 'b-', label='Estimation (EKF)')
            axes[1, 1].legend(loc='upper left')
            axes[1, 1].set_title('YZ Axis Position: Esti. (EKF)')
            axes[1, 1].set_xlabel('Y Axis Position [m]')
            axes[1, 1].set_ylabel('Z Axis Position [m]')

            plt.show()


ld = logData(filename)
ld.calculate(plot=True)

