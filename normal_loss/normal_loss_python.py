
import torch

class NormalLoss(torch.nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def distance_matrix(self, points1, points2):
        return torch.cdist(points1, points2)

    def normal_distance(self, normal1, normal2):
        return 0

    def forward(self, predicted_points_batch, predicted_normals_batch, gt_points_batch, gt_normals_batch):
        for predicted_points, predicted_normals, gt_points, gt_normals in zip(predicted_points_batch, predicted_normals_batch,
                                                                              gt_points_batch, gt_normals_batch):
            distance_matrix = self.distance_matrix(predicted_points, gt_points)

            _, gt_points_indices = torch.min(distance_matrix, dim=1)
            _, predicted_points_indices = torch.min(distance_matrix, dim=0)
            nearest_gt_normals = gt_normals[gt_points_indices]
            nearest_predicted_normals = predicted_normals[predicted_points_indices]

            dist1 = self.normal_distance(predicted_normals, nearest_gt_normals)
            dist2 = self.normal_distance(gt_normals, nearest_predicted_normals)

            return torch.mean(dist1) + torch.mean(dist2)



if __name__ == '__main__':
    a = torch.FloatTensor([[3, 2, 1], [4, 5, 6], [6, 7, 8]])
    b = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [3, 2, 1]])

    distance_matrix = torch.cdist(a, b, 2)

    print(distance_matrix)

    print(torch.min(distance_matrix, dim=0))
    print(torch.min(distance_matrix, dim=1))