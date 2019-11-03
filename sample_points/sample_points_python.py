import open3d
# ATTENTION! open3d have to be imported before pytorch
import torch
import numpy as np
from torch.distributions.categorical import Categorical


class PointSampler(torch.nn.Module):
    def __init__(self, point_num):
        super(PointSampler, self).__init__()
        self.point_num = point_num

    def sample_surfaces(self, vertices, faces, point_num):
        triangles = vertices[faces]
        vec1 = triangles[:, 1, :] - triangles[:, 0, :]
        vec2 = triangles[:, 2, :] - triangles[:, 0, :]
        areas = np.linalg.norm(np.cross(vec1, vec2), axis=1) / 2

        areas = torch.from_numpy(areas)
        m = Categorical(areas)

        return m.sample(sample_shape=(point_num, ))



    def forward(self, vertices_batch: torch.Tensor, faces_batch: torch.Tensor):
        # randomly select surface
        for vertices, faces in zip(vertices_batch, faces_batch):
            indices = self.sample_surfaces(vertices.numpy(), faces.numpy(), self.point_num)


if __name__ == '__main__':
    mesh: open3d.geometry.TriangleMesh = open3d.io.read_triangle_mesh(filename='../tmp/model_normalized.obj')
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)



    triangles = vertices[faces]
    vec1 = triangles[:, 1, :] - triangles[:, 0, :]
    vec2 = triangles[:, 2, :] - triangles[:, 0, :]
    areas = np.linalg.norm(np.cross(vec1, vec2), axis=1) / 2

    areas = torch.from_numpy(areas)



    m = Categorical(areas)

    print(m.sample(sample_shape=(3,)))



