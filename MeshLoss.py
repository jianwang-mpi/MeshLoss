from sample_points.sample_points_python import PointSampler
# from chamfer_distance import ChamferDistance
from .ChamferDistancePytorch.dist_chamfer import chamferDist
from normal_loss.normal_loss_python import NormalLoss
from edge_loss.edge_loss_python import EdgeLoss
import torch


class MeshLoss(torch.nn.Module):
    def __init__(self, point_sample_num=1000, chamfer_weight=1.0, norm_weight=0.1, edge_weight=0.5):
        super(MeshLoss, self).__init__()
        self.point_sample_num = point_sample_num
        self.chamfer_weight = chamfer_weight
        self.norm_weight = norm_weight
        self.edge_weight = edge_weight

        self.points_sampler = PointSampler(self.point_sample_num)
        self.chamfer_distance = chamferDist()
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

    def test(self):
        point1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]).unsqueeze(0).cuda()
        point2 = torch.FloatTensor([[3, 2, 1], [1, 2, 3]]).unsqueeze(0).cuda()
        print(self.chamfer_distance(point1, point2))


if __name__ == '__main__':
    meshLoss = MeshLoss(point_sample_num=1000)
    meshLoss.test()