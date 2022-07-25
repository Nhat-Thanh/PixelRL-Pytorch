import cv2
import numpy as np
from utils.common import to_numpy


class State:
    def __init__(self):
        self.image = None
        self.tensor = None
        self.move_range = 3

    def reset(self, bgr_image):
        self.image = bgr_image
        b, _, h, w = self.image.shape
        previous_state = np.zeros(shape=(b, 64, h, w), dtype=self.image.dtype)
        self.tensor = np.concatenate([self.image, previous_state], axis=1)

    def set(self, CIELab_image):
        lab = np.copy(CIELab_image)
        lab[:, 0, :, :] /= 100
        lab[:, 1, :, :] /= 127
        lab[:, 2, :, :] /= 127
        self.tensor[:, 0:3, :, :] = lab

    def step(self, act, inner_state):
        actions = to_numpy(act)
        inner_state = to_numpy(inner_state)
        bgr_1 = self.image.copy() * 0.95 + 0.5 * 0.05
        bgr_2 = self.image.copy() * 1.05 - 0.5 * 0.05

        bgr_t = np.transpose(self.image, (0, 2, 3, 1))
        bgr_3 = np.zeros(bgr_t.shape, bgr_t.dtype)
        b = self.image.shape[0]
        for i in range(0, b):
            if np.sum(actions[i] == 3) > 0:
                hsv = cv2.cvtColor(bgr_t[i], cv2.COLOR_BGR2HSV)
                hsv[1] *= 0.95
                bgr_3[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr_3 = np.transpose(bgr_3, (0, 3, 1, 2))

        bgr_4 = np.zeros(bgr_t.shape, bgr_t.dtype)
        for i in range(0, b):
            if np.sum(actions[i] == 4) > 0:
                hsv = cv2.cvtColor(bgr_t[i], cv2.COLOR_BGR2HSV)
                hsv[1] *= 1.05
                bgr_4[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr_4 = np.transpose(bgr_4, (0, 3, 1, 2))

        bgr_5 = self.image.copy() - 0.5 * 0.05

        bgr_6 = self.image.copy() + 0.5 * 0.05

        bgr_7 = np.copy(self.image)
        bgr_7[:, 1:, :, :] *= 0.95
        
        bgr_8 = np.copy(self.image)
        bgr_8[:, 1:, :, :] *= 1.05

        bgr_9 = np.copy(self.image)
        bgr_9[:, :2, :, :] *= 0.95

        bgr_10 = np.copy(self.image)
        bgr_10[:, :2, :, :] *= 1.05

        bgr_11 = np.copy(self.image)
        bgr_11[:, ::2, :, :] *= 0.95

        bgr_12 = np.copy(self.image)
        bgr_12[:, ::2, :, :] *= 1.05

        actions = np.stack([actions, actions, actions], axis=1)
        self.image = np.where(actions == 1, bgr_1, self.image)
        self.image = np.where(actions == 2, bgr_2, self.image)
        self.image = np.where(actions == 3, bgr_3, self.image)
        self.image = np.where(actions == 4, bgr_4, self.image)
        self.image = np.where(actions == 5, bgr_5, self.image)
        self.image = np.where(actions == 6, bgr_6, self.image)
        self.image = np.where(actions == 7, bgr_7, self.image)
        self.image = np.where(actions == 8, bgr_8, self.image)
        self.image = np.where(actions == 9, bgr_9, self.image)
        self.image = np.where(actions == 10, bgr_10, self.image)
        self.image = np.where(actions == 11, bgr_11, self.image)
        self.image = np.where(actions == 12, bgr_12, self.image)

        self.tensor[:, 0:3, :, :] = self.image
        self.tensor[:, -64:, :, :] = inner_state
