import torch
from sample_points.sample_points_python import PointSampler
from chamfer_distance.chamfer_distance import ChamferDistance
from normal_loss.normal_loss_python import NormalLoss
from edge_loss.edge_loss_python import EdgeLoss



class MeshLoss(torch.nn.Module):
    def __init__(self, point_sample_num, chamfer_weight=1.0, norm_weight=0.1, edge_weight=0.5):
        super(MeshLoss, self).__init__()
        self.point_sample_num = point_sample_num
        self.chamfer_weight = chamfer_weight
        self.norm_weight = norm_weight
        self.edge_weight = edge_weight

        self.points_sampler = PointSampler(self.point_sample_num)
        self.chamfer_distance = ChamferDistance()
        self.normal_loss = NormalLoss()
        self.edge_loss = EdgeLoss()

    def forward(self, predicted_vertices, predicted_faces, gt_vertices, gt_faces):
        # chamfer loss
        predicted_point_cloud, predicted_normals = self.points_sampler(predicted_vertices, predicted_faces)
        gt_point_cloud, gt_normals = self.points_sampler(predicted_vertices, predicted_faces)
        dist1, dist2 = self.chamfer_distance(predicted_point_cloud, gt_point_cloud)
        chamfer_loss = torch.mean(dist1) + torch.mean(dist2)

        norm_loss = self.normal_loss(predicted_point_cloud, predicted_normals, gt_point_cloud, gt_normals)
        edge_loss = self.edge_loss(predicted_vertices, predicted_faces)

        loss = self.chamfer_weight * chamfer_loss + self.norm_weight * norm_loss + self.edge_weight * edge_loss

        return loss
