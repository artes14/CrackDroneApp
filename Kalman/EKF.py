import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

np.random.seed(0)


def get_radar(xpos_pred):
    """레이다로 측정된 고도와 직선거리를 반환해줌"""
    xvel_w = np.random.normal(0, 5)  # xvel_w: 이동거리의 시스템 잡음 [m/s].
    xvel_true = 100 + xvel_w  # xvel_true: 이동거리의 참값 [m/s].

    ypos_w = np.random.normal(0, 10)  # ypos_w: 고도의 시스템 잡음 [m].
    ypos_true = 1000 + ypos_w  # ypos_true: 고도의 참값 [m].

    xpos_pred = xpos_pred + xvel_true * dt  # xpos_pred: 이동거리 예상치 [m].

    rpos_v = xpos_pred * np.random.normal(0, 0.05)  # rpos_v: 레이다의 측정잡음.
    rpos_meas = np.sqrt(xpos_pred ** 2 + ypos_true ** 2) + rpos_v  # r: 측정 거리 [m] (observable).

    return rpos_meas, xpos_pred


# 야코비안 계산
def Ajacob_at(x_esti):
    return A


def Hjacob_at(x_pred):
    H[0][0] = x_pred[0] / np.sqrt(x_pred[0] ** 2 + x_pred[2] ** 2)
    H[0][1] = 0
    H[0][2] = x_pred[2] / np.sqrt(x_pred[0] ** 2 + x_pred[2] ** 2)
    return H


# 비선형시스템 계산 (측정모델)
def fx(x_esti):
    return A @ x_esti


def hx(x_pred):
    z_pred = np.sqrt(x_pred[0] ** 2 + x_pred[2] ** 2)
    return np.array([z_pred])


def extended_kalman_filter(z_meas, x_esti, P):
    """Extended Kalman Filter Algorithm."""
    # (1) Prediction.
    A = Ajacob_at(x_esti)
    x_pred = fx(x_esti)
    P_pred = A @ P @ A.T + Q

    # (2) Kalman Gain.
    H = Hjacob_at(x_pred)
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    # (3) Estimation.
    x_esti = x_pred + K @ (z_meas - hx(x_pred))

    # (4) Error Covariance.
    P = P_pred - K @ H @ P_pred

    return x_esti, P


# Input parameters.
time_end = 20
dt = 0.05

# Initialization for system model.
# Matrix: A, H, Q, R, P_0
# Vector: x_0
A = np.eye(3) + dt * np.array([[0, 1, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
H = np.zeros((1, 3))
Q = np.array([[0, 0, 0],
              [0, 0.001, 0],
              [0, 0, 0.001]])
R = np.array([[10]])

# Initialization for estimation.
x_0 = np.array([0, 90, 1100])  # [horizontal position, horizontal velocity, vertical position].
P_0 = 10 * np.eye(3)

time = np.arange(0, time_end, dt)
n_samples = len(time)
xpos_esti_save = np.zeros(n_samples)
ypos_esti_save = np.zeros(n_samples)
rpos_esti_save = np.zeros(n_samples)
xvel_esti_save = np.zeros(n_samples)
rpos_meas_save = np.zeros(n_samples)
xpos_pred = 0
x_esti, P = None, None
for i in range(n_samples):
    z_meas, xpos_pred = get_radar(xpos_pred)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = extended_kalman_filter(z_meas, x_esti, P)

    xpos_esti_save[i] = x_esti[0]
    ypos_esti_save[i] = x_esti[2]
    rpos_esti_save[i] = np.sqrt(x_esti[0] ** 2 + x_esti[2] ** 2)
    xvel_esti_save[i] = x_esti[1]
    rpos_meas_save[i] = z_meas
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

axes[0, 0].plot(time, xpos_esti_save, 'bo-', label='Estimation (EKF)')
axes[0, 0].legend(loc='upper left')
axes[0, 0].set_title('Horizontal Distance: Esti. (EKF)')
axes[0, 0].set_xlabel('Time [sec]')
axes[0, 0].set_ylabel('Horizontal Distance [m]')

axes[0, 1].plot(time, ypos_esti_save, 'bo-', label='Estimation (EKF)')
axes[0, 1].legend(loc='upper left')
axes[0, 1].set_title('Vertical Distance: Esti. (EKF)')
axes[0, 1].set_xlabel('Time [sec]')
axes[0, 1].set_ylabel('Vertical Distance [m]')

axes[1, 0].plot(time, rpos_meas_save, 'r*--', label='Measurements', markersize=10)
axes[1, 0].plot(time, rpos_esti_save, 'bo-', label='Estimation (EKF)')
axes[1, 0].legend(loc='upper left')
axes[1, 0].set_title('Radar Distance: Meas. v.s. Esti. (EKF)')
axes[1, 0].set_xlabel('Time [sec]')
axes[1, 0].set_ylabel('Radar Distance [m]')

axes[1, 1].plot(time, xvel_esti_save, 'bo-', label='Estimation (EKF)')
axes[1, 1].legend(loc='upper left')
axes[1, 1].set_title('Horizontal Velocity: Esti. (EKF)')
axes[1, 1].set_xlabel('Time [sec]')
axes[1, 1].set_ylabel('Horizontal Velocity [m/s]')

plt.show()