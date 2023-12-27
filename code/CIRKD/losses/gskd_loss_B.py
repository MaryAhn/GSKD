import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GSKDB']


class GSKDB(nn.Module):
    def __init__(self, num_classes, ignore_label, temperature, N, S, t_channel, s_channel):
        super(GSKDB, self).__init__()
        self.temperature = temperature
        self.t_channel = t_channel
        self.s_channel = s_channel
        self.ignore_label = ignore_label

        self.project_head = nn.Sequential(
            nn.Conv2d(self.s_channel, self.t_channel, 1, bias=False),
            nn.SyncBatchNorm(self.t_channel),
            nn.ReLU(True),
            nn.Conv2d(self.t_channel, self.t_channel, 1, bias=False)
        )
        self.num_classes = num_classes
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
        C, N, Ch = student_feat.shape
        sample = N * C
        if self.S < 0:
            stride = int(-1 * self.S * C)
        else:
            stride = self.S
        if stride == 0:
            col = 0
        else:
            col = sample // stride
        desc = N * C * stride

        student_feat = student_feat.reshape(N * C, Ch)
        teacher_feat = teacher_feat.reshape(N * C, Ch)
        student_feat_T, teacher_feat_T = student_feat.transpose(0, 1), teacher_feat.transpose(0, 1)
        image_desc_cand_tt, image_desc_cand_ss = torch.matmul(teacher_feat, teacher_feat_T), torch.matmul(student_feat, student_feat_T)
        image_desc_total_err = 0

        shuffle_map = torch.cuda.LongTensor(torch.argsort(torch.rand((sample, sample), device='cuda'), dim=1))

        if col > 0:
            for i in range(col):
                desc_tt = image_desc_cand_tt.gather(dim=1, index=shuffle_map[:, i * stride: (i + 1) * stride]).reshape(desc)
                desc_ss = image_desc_cand_ss.gather(dim=1, index=shuffle_map[:, i * stride: (i + 1) * stride]).reshape(desc)
                image_desc_tt, image_desc_ss = F.softmax(desc_tt / temperature, dim=0), F.log_softmax(desc_ss / temperature, dim=0)
                image_desc_err = F.kl_div(input=image_desc_ss, target=image_desc_tt, reduction='batchmean') * (temperature ** 2)
                image_desc_total_err += image_desc_err
            image_desc_total_err = image_desc_total_err / col
            if sample % stride != 0:
                desc = sample * (sample - col * stride)
                desc_tt = image_desc_cand_tt.gather(dim=1, index=shuffle_map[:, col * stride:]).reshape(desc)
                desc_ss = image_desc_cand_ss.gather(dim=1, index=shuffle_map[:, col * stride:]).reshape(desc)
                image_desc_tt, image_desc_ss = F.softmax(desc_tt / temperature, dim=0), F.log_softmax(desc_ss / temperature, dim=0)
                image_desc_total_err += F.kl_div(input=image_desc_ss, target=image_desc_tt, reduction='batchmean') * (temperature ** 2)
        else:
            desc_tt = image_desc_cand_tt.gather(dim=1, index=shuffle_map)
            desc_ss = image_desc_cand_ss.gather(dim=1, index=shuffle_map)
            image_desc_tt, image_desc_ss = F.softmax(desc_tt / temperature, dim=0), F.log_softmax(desc_ss / temperature,
                                                                                                  dim=0)
            image_desc_err = F.kl_div(input=image_desc_ss, target=image_desc_tt, reduction='batchmean') * (temperature ** 2)
            image_desc_total_err += image_desc_err

        return image_desc_total_err

    def image_desc_kd(self, keys, students, labels, gt_idx):
        keys = self.concat_all_gather(keys)
        students = self.concat_all_gather(students)
        labels = self.concat_all_gather(labels)
        gt_idx = self.concat_all_gather(gt_idx)

        batch_size, feat_dim, H, W = keys.size()
        keys = keys.reshape(batch_size, H, W, feat_dim)
        students = students.reshape(batch_size, H, W, feat_dim)
        gt_idx = gt_idx.type(torch.cuda.BoolTensor)

        image_desc_error = 0

        for bs in range(batch_size):
            this_key = keys[bs]
            this_key = this_key[gt_idx[bs].nonzero(as_tuple=True)]
            this_feat = this_key.contiguous().view(feat_dim, -1)

            this_student = students[bs]
            this_student = this_student[gt_idx[bs].nonzero(as_tuple=True)]
            this_student_feat = this_student.contiguous().view(feat_dim, -1)

            this_labels = labels[bs]
            this_labels = this_labels[gt_idx[bs].nonzero(as_tuple=True)]
            this_labels = this_labels.contiguous().view(-1)
            this_label_ids = torch.unique(this_labels)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_label]
            b_t_sampled, b_s_sampled = [], []
            image_desc_error, wrong = 0, 0

            for lb in this_label_ids:
                idxs = (this_labels == lb).nonzero().squeeze()
                if torch.tensor(0, ).shape == idxs.shape:
                    wrong += 1
                else:
                    num_pixel = idxs.shape[0]
                    perm = torch.randperm(num_pixel)
                    K = min(num_pixel, self.N)
                    if K == self.N:
                        feat = this_feat[:, idxs[perm[:K]]]
                        student_feat = this_student_feat[:, idxs[perm[:K]]]
                    else:
                        rand_idxs = torch.ones(this_feat.shape[1]).type(torch.cuda.LongTensor)
                        rand_idxs[idxs] = 0
                        rand_idxs = rand_idxs.nonzero().squeeze()
                        if torch.tensor(0, ).shape == rand_idxs.shape:
                            return 0. * (students ** 2).mean()
                        rand_perm = torch.randperm(rand_idxs.shape[0])
                        sampled_idxs = rand_idxs[rand_perm[:self.N - K]]
                        idxs = torch.cat((idxs, sampled_idxs))
                        feat = this_feat[:, idxs]
                        student_feat = this_student_feat[:, idxs]
                    feat = torch.transpose(feat, 0, 1)
                    student_feat = torch.transpose(student_feat, 0, 1)

                    b_t_sampled.append(feat)
                    b_s_sampled.append(student_feat)
            if wrong == len(this_label_ids) or not b_t_sampled:
                image_desc_error = 0. * (students ** 2).mean()
            else:
                i_t_sampled, i_s_sampled = torch.stack(b_t_sampled, dim=0), torch.stack(b_s_sampled, dim=0)
                image_desc_error += self.image_sub_desc(student_feat=i_s_sampled, teacher_feat=i_t_sampled,
                                                        temperature=self.temperature)

        return image_desc_error

    def forward(self, t_predicts, s_feats, t_feats, labels = None):
        B, H, W = labels.size()
        t_feats = F.normalize(t_feats, p=2, dim=1)
        s_feats = self.project_head(s_feats)
        s_feats = F.normalize(s_feats, p=2, dim=1)

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (s_feats.shape[2], s_feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == s_feats.shape[-1], '{} {}'.format(labels.shape, s_feats.shape)

        gt_idx = torch.ones_like(labels)
        gt_idx[t_predicts != labels] = 0

        ori_s_fea = s_feats
        ori_t_fea = t_feats
        ori_labels = labels

        image_desc_sim_dis = self.image_desc_kd(ori_t_fea.detach().clone(), ori_s_fea, ori_labels.detach().clone(),gt_idx=gt_idx)

        return image_desc_sim_dis
