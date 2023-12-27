import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GSKDB']

# GSKD for classification
class GSKDB(nn.Module):
    def __init__(self, temperature, N, S, t_channel, s_channel):
        super(GSKDB, self).__init__()
        self.temperature = temperature
        self.t_channel = t_channel
        self.s_channel = s_channel

        self.project_head = nn.Sequential(
            nn.Conv2d(self.s_channel, self.t_channel, 1, bias=False),
            nn.SyncBatchNorm(self.t_channel),
            nn.ReLU(True),
            nn.Conv2d(self.t_channel, self.t_channel, 1, bias=False)
        )

        self.N = N
        self.S = S

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    def image_sub_desc(self, student_feat, teacher_feat, temperature):
        B, C, sample = student_feat.shape
        if self.S < 0:
            stride = int(-1 * self.S * C)
        else:
            stride = self.S
        if stride == 0:
            col = 0
        else:
            col = sample // stride
        desc = B * sample * stride

        student_feat_T, teacher_feat_T = student_feat.transpose(1, 2), teacher_feat.transpose(1, 2)
        image_desc_cand_tt, image_desc_cand_ss = torch.matmul(teacher_feat_T, teacher_feat), torch.matmul(student_feat_T, student_feat)
        image_desc_total_err = 0

        shuffle_map = torch.cuda.LongTensor(torch.argsort(torch.rand((B, sample, sample), device='cuda'), dim=2))

        if col > 0:
            for i in range(col):
                desc_tt = image_desc_cand_tt.gather(dim=2, index=shuffle_map[:, i * stride: (i + 1) * stride]).reshape(desc)
                desc_ss = image_desc_cand_ss.gather(dim=2, index=shuffle_map[:, i * stride: (i + 1) * stride]).reshape(desc)
                image_desc_tt, image_desc_ss = F.softmax(desc_tt / temperature, dim=0), F.log_softmax(desc_ss / temperature, dim=0)
                image_desc_err = F.kl_div(input=image_desc_ss, target=image_desc_tt, reduction='batchmean') * (temperature ** 2)
                image_desc_total_err += image_desc_err
            image_desc_total_err = image_desc_total_err / col
            if sample % stride != 0:
                desc = sample * (sample - col * stride)
                desc_tt = image_desc_cand_tt.gather(dim=2, index=shuffle_map[:, col * stride:]).reshape(desc)
                desc_ss = image_desc_cand_ss.gather(dim=2, index=shuffle_map[:, col * stride:]).reshape(desc)
                image_desc_tt, image_desc_ss = F.softmax(desc_tt / temperature, dim=0), F.log_softmax(desc_ss / temperature, dim=0)
                image_desc_total_err += F.kl_div(input=image_desc_ss, target=image_desc_tt, reduction='batchmean') * (temperature ** 2)
        else:
            desc_tt = image_desc_cand_tt.gather(dim=2, index=shuffle_map)
            desc_ss = image_desc_cand_ss.gather(dim=2, index=shuffle_map)
            image_desc_tt, image_desc_ss = F.softmax(desc_tt / temperature, dim=0), F.log_softmax(desc_ss / temperature,
                                                                                                  dim=0)
            image_desc_err = F.kl_div(input=image_desc_ss, target=image_desc_tt, reduction='batchmean') * (temperature ** 2)
            image_desc_total_err += image_desc_err

        return image_desc_total_err

    def image_desc_kd(self, keys, students):
        keys = self.concat_all_gather(keys)
        students = self.concat_all_gather(students)

        batch_size, feat_dim, H, W = keys.size()
        _, student_feat_dim, H_s, W_s = students.size()
        keys = keys.reshape(batch_size, H, W, feat_dim)
        students = students.reshape(batch_size, H_s, W_s, student_feat_dim)

        this_key = keys
        this_feat = this_key.contiguous().view(batch_size, feat_dim, -1)

        this_student = students
        this_student_feat = this_student.contiguous().view(batch_size, feat_dim, -1)

        num_pixel = H_s * W_s
        K = self.N * self.S
        perm = torch.randperm(num_pixel)
        feat = this_feat[:, :, perm[:K]]
        student_feat = this_student_feat[:, :, perm[:K]]

        image_desc_error = self.image_sub_desc(student_feat=student_feat, teacher_feat=feat,
                                               temperature=self.temperature)
        return image_desc_error


    def forward(self, s_feats, t_feats):
        t_feats = F.normalize(t_feats, p=2, dim=1)
        s_feats = self.project_head(s_feats)
        s_feats = F.normalize(s_feats, p=2, dim=1)

        ori_s_fea = s_feats
        ori_t_fea = t_feats

        image_desc_sim_dis = self.image_desc_kd(ori_t_fea.detach().clone(), ori_s_fea)

        return image_desc_sim_dis
