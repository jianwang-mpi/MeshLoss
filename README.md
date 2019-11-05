# MeshLoss

### Install
- Compile the chamfer distance package
```bash
cd ChamferDistancePytorch && python setup.py install
```
- Install the mesh loss
```bash
python setup.py install
```

### Usage:
```python
from mesh_loss import MeshLoss

# get predicted vertices and faces

meshLoss = MeshLoss(point_sample_num=1000)
loss = meshLoss(predicted_vertices, predicted_faces, gt_vertices, gt_faces)
```