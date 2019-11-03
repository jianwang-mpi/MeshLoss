import torch

class MeshLoss(torch.nn.Module):
    def __init__(self, point_sample_num):
        super(MeshLoss, self).__init__()
        self.point_sample_num = point_sample_num
