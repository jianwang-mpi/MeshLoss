import torch
from sample_points.sample_points_python import PointSampler
from chamfer_distance.chamfer_distance import ChamferDistance

class MeshLoss(torch.nn.Module):
    def __init__(self, point_sample_num, chamfer_weight=1.0, norm_weight=0.1, edge_weight=0.5):
        super(MeshLoss, self).__init__()
        self.point_sample_num = point_sample_num
        self.chamfer_weight = chamfer_weight
        self.norm_weight = norm_weight
        self.edge_weight = edge_weight

        self.points_sampler = PointSampler(self.point_sample_num)
        self.chamfer_distance = ChamferDistance()

    def forward(self, predicted_vertices, predicted_faces, gt_vertices, gt_faces):
        predicted_point_cloud = self.points_sampler(predicted_vertices, predicted_faces)
        
