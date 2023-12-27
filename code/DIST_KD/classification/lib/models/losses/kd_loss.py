import math
import torch
import torch.nn as nn
from functools import partial

from .kl_div import KLDivergence
from .dist_kd import DIST


class KDLoss():
    '''
    kd loss wrapper.
    '''

    def __init__(self, student, teacher, ori_loss, kd_method='kdt4', student_module='', teacher_module='', ori_loss_weight=1.0, kd_loss_weight=1.0):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight

        self._teacher_out = None
        self._student_out = None

        # init kd loss
        if kd_method == 'kd':
            self.kd_loss = KLDivergence(tau=4)
        elif kd_method == 'dist':
            self.kd_loss = DIST(beta=1, gamma=1, tau=1)
        elif kd_method.startswith('dist_t'):
            tau = float(kd_method[6:])
            self.kd_loss = DIST(beta=1, gamma=1, tau=tau)
        elif kd_method.startswith('kdt'):
            tau = float(kd_method[3:])
            self.kd_loss = KLDivergence(tau)
        else:
            raise RuntimeError(f'KD method {kd_method} not found.')

        teacher.eval()

    def __call__(self, x, targets):
        with torch.no_grad():
            t_logits, _ = self.teacher(x)

        # compute ori loss of student
        logits, _ = self.student(x)
        ori_loss = self.ori_loss(logits, targets)

        # compute kd loss
        kd_loss = self.kd_loss(logits, t_logits)

        return ori_loss * self.ori_loss_weight + kd_loss * self.kd_loss_weight
