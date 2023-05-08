import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = []
        self.models = [ManiuplatorModel(Tp, m3 = 0.1, r3 = 0.05),
                       ManiuplatorModel(Tp, m3 = 0.01, r3 = 0.01),
                       ManiuplatorModel(Tp, m3 = 1.0, r3 = 0.3)]
        self.i = 0
        self.x_prev = np.zeros(4)
        self.u_prev = np.zeros(2)
        self.Tp = Tp
        self.u = np.zeros((2, 1))

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        q1, q2, q1_dot, q2_dot = x
        x_mi = []

        for model in self.models:
            x_mi.append((model.x_dot(self.x_prev, self.u_prev) - self.x_prev.reshape(4, 1)) / self.Tp)
        pass

        x_reshaped = x.reshape(4, 1)
        errors = list(map( lambda x: np.sum(abs(x_reshaped - x)), x_mi))
        self.i = np.argmin(errors)
        # print('current model: ', self.i)


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]

        K_d = [[25, 0], [0, 25]]
        K_p = [[60, 0], [0, 60]]

        v = q_r_ddot + K_d @ (q_r_dot - q_dot) + K_p @ (q_r - q)

        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.u = u

        self.u_prev = u
        self.x_prev = x
        return u