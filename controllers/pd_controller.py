import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_d, q_d_dot, q_d_ddot):
        ### TODO: Please implement me
        e = q_d - q
        e_dot = q_d_dot - q_dot
        Kd = np.array([[25, 0], [0, 15]])
        Kp = np.array([[25, 0], [0, 60]])
        u = Kd @ e_dot + Kp @ e
        return u
