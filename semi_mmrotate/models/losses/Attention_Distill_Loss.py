import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_LOSSES

@ROTATED_LOSSES.register_module()
class AttentionDistillLoss(nn.Module):
    def __init__(self, channels_dict={'cls': 15, 'bbox': 4, 'angle': 3, 'center': 1}, branches=('cls', 'bbox', 'angle', 'center'), weights=None):
        """
        channels_dict: dict like {'cls': 15, 'bbox': 4, 'angle': 3, 'center': 1}
        weights: per-branch weight for loss, same order as branches
        """
        super(AttentionDistillLoss, self).__init__()
        self.branches = branches
        self.weights = weights if weights is not None else [1.0] * len(branches)

        # Create Q/K/V projections for each branch
        # self.q_projs = nn.ModuleDict()
        # self.k_projs = nn.ModuleDict()
        # self.v_projs = nn.ModuleDict()

        # for b in branches:
        #     c = channels_dict[b]
        #     self.q_projs[b] = nn.Linear(c, c)
        #     self.k_projs[b] = nn.Linear(c, c)
        #     self.v_projs[b] = nn.Linear(c, c)
            # for param in self.k_projs[b].parameters():
            #     param.requires_grad = False
            # for param in self.v_projs[b].parameters():
            #     param.requires_grad = False

    def forward(self, student_logits, teacher_logits):
        """
        student_logits/teacher_logits: tuple of (cls, bbox, angle, center), each is a list of 5 tensors [B, C, H, W]
        """
        total_loss = 0
        for idx, b in enumerate(self.branches):
            if idx>0:
                break
            s_feats = student_logits[idx]
            t_feats = teacher_logits[idx]

            # for i in range(len(s_feats)):  # FPN层数，通常是5
            i=0
            s = s_feats[i]  # [B, C, H, W]
            t = t_feats[i]
            # print(f"s.requires_grad={s.requires_grad}, t.requires_grad={t.requires_grad}")

            B, C, H, W = s.shape
            s_flat = s.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
            t_flat = t.view(B, C, -1).permute(0, 2, 1)

            # Project Q, K, V
            # q = self.q_projs[b](s_flat)
            # k = self.k_projs[b](t_flat)
            # v = self.v_projs[b](t_flat)

            q =  s_flat
            k =  t_flat
            v =  t_flat

            # Compute attention and output
            attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5), dim=-1)
            t_proj = torch.bmm(attn, v)

            # Alignment loss
            loss = F.mse_loss(q, t_proj)
            total_loss += self.weights[idx] * loss

        return total_loss
