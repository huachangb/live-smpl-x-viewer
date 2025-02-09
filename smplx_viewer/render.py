import pyrender
import trimesh
import torch
import smplx
import numpy as np


def update_viewer(mesh: pyrender.Mesh, viewer: pyrender.Viewer, scene: pyrender.Scene, clear: bool = False, *args, **kwargs) -> None:
    """ Adds mesh to scene """
    viewer.render_lock.acquire()

    if clear:
        for node in list(scene.mesh_nodes):
            scene.remove_node(node)

    scene.add(mesh, *args, **kwargs)
    viewer.render_lock.release()


def update_smpl_model(
        model: smplx.SMPLX,
        viewer: pyrender.Viewer,
        scene: pyrender.Scene,
        betas: torch.Tensor,
        expression: torch.Tensor,
        global_orient: torch.Tensor,
        plot_joints: bool = False,
    ):
    """ Updates scene """
    print(betas.shape, expression.shape)
    output = model(
        betas=betas,
        expression=expression,
        global_orient=global_orient,
        return_verts=True
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, model.faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    update_viewer(mesh, viewer, scene, clear=True)

    if plot_joints:
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        update_viewer(joints_pcl, viewer, scene)