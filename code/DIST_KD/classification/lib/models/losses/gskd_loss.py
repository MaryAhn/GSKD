import torch
import torch.nn as nn
from functools import partial

from .kl_div import KLDivergence
from .dist_kd import DIST
from .gskd_loss_B import GSKDB

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class KDLoss():
    '''
    kd loss wrapper.
    '''

    def __init__(self, student, teacher, ori_loss, kd_method='kd', student_model='resnet18', teacher_model='resnet34', ori_loss_weight=1.0, kd_loss_weight=1.0,
                 gskd_loss_weight=0.0, gskd_temperature=1.0, N=16, S=8, distill_layer=-1):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight
        self.teacher_model = teacher_model
        self.student_model = student_model

        self.gskd_loss_weight = gskd_loss_weight
        self.gskd_temperature = gskd_temperature
        self.distill_layer = distill_layer
        self.N = N
        self.S = S

        # teacher channel
        if 'cifar' in self.teacher_model:
            if 'resnet' in self.teacher_model:
                self.t_channels = [16, 16, 32, 64]
            else:
                self.t_channels = [16, 32, 64, 128]
        else:
            if 'resnet' in self.teacher_model:
                if self.teacher_model.endswith('18') or self.teacher_model.endswith('34'):
                    self.t_channels = [64, 64, 128, 256, 512]  # resnet18, 34
                else:
                    self.t_channels = [64, 256, 512, 1024, 2048]  # resnet50, 101, 152
            else:
                self.t_channels = [64, 64, 128, 256, 512]
        self.t_channel = self.t_channels[self.distill_layer]

        # student channel
        if 'cifar' in self.student_model:
            if 'resnet' in self.student_model:
                self.s_channels = [16, 16, 32, 64]
            else:
                self.s_channels = [16, 16, 32, 64]
        elif 'mobile' in self.student_model:
            self.s_channels = [64, 256, 512, 512, 1024]
        else:
            self.s_channels = [64, 64, 128, 256, 512]
        self.s_channel = self.s_channels[self.distill_layer]

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
        self.gskd_loss = GSKDB(temperature=self.gskd_temperature, N=self.N, S=self.S, t_channel=self.t_channel, s_channel=self.s_channel).to(device='cuda')

        teacher.eval()

    def __call__(self, x, targets):
        if 'cifar' in self.teacher_model:
            with torch.no_grad():
                t_output = self.teacher(x, is_feat=True)
                t_logits, t_feats = t_output[1], t_output[0][self.distill_layer]

            s_output = self.student(x, is_feat=True)
            logits, feats = s_output[1], s_output[0][self.distill_layer]
        else:
            with torch.no_grad():
                t_output = self.teacher.extract_feature(x)
                t_logits, t_feats = t_output[0], t_output[1][self.distill_layer]

            s_output = self.student(x)
            logits, feats = s_output[0], s_output[1][self.distill_layer]

        # compute ori loss of student
        ori_loss = self.ori_loss(logits, targets)

        # compute kd loss
        kd_loss = self.kd_loss(logits, t_logits)
        gskd_loss = self.gskd_loss(s_feats=feats, t_feats=t_feats)
        return ori_loss * self.ori_loss_weight + kd_loss * self.kd_loss_weight + gskd_loss * self.gskd_loss_weight

