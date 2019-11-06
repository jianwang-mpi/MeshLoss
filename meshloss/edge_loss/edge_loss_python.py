import open3d
import torch

import numpy as np


class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, vertices_batch, faces_batch):
        result_batch = []
        for vertices, faces in zip(vertices_batch, faces_batch):
            triangles = vertices[faces]
            edge1 = triangles[:, 1, :] - triangles[:, 0, :]
            edge2 = triangles[:, 2, :] - triangles[:, 0, :]
            edge3 = triangles[:, 2, :] - triangles[:, 1, :]

            length1 = torch.norm(edge1, p=2, dim=1)
            length2 = torch.norm(edge2, p=2, dim=1)
            length3 = torch.norm(edge3, p=2, dim=1)
            result_batch.append(torch.mean(length1 + length2 + length3))
        return torch.stack(result_batch)


if __name__ == '__main__':
    mesh: open3d.geometry.TriangleMesh = open3d.io.read_triangle_mesh(filename='../tmp/model_normalized.obj')
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    faces = torch.from_numpy(faces).long()
    vertices = torch.from_numpy(vertices).float()
    vertices.requires_grad = True

    triangles = vertices[faces]
    edge1 = triangles[:, 1, :] - triangles[:, 0, :]
    edge2 = triangles[:, 2, :] - triangles[:, 0, :]
    edge3 = triangles[:, 2, :] - triangles[:, 1, :]

    length1 = torch.norm(edge1, p=2, dim=1)
    length2 = torch.norm(edge2, p=2, dim=1)
    length3 = torch.norm(edge3, p=2, dim=1)

    print(torch.mean(length1 + length2 + length3))
