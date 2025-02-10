from typing import Callable

import pyrender
import trimesh
import torch
import smplx
import numpy as np


class SMPLViewer(pyrender.Viewer):
    """ Wrapper for Pyrender Viewer to dynamically update SMPL-X renders """
    def __init__(self,
                 model: smplx.SMPLX, scene: pyrender.Scene,
                 initial_betas: torch.Tensor, initial_pose: torch.Tensor,
                 initial_global_orient: torch.Tensor = None,
                 show_joints: bool = False,
                 viewport_size=None,
                 render_flags=None, viewer_flags=None, registered_keys=None,
                 run_in_thread=False, **kwargs):
        super().__init__(scene, viewport_size, render_flags, viewer_flags, registered_keys, run_in_thread, **kwargs)
        self.show_joints = show_joints
        self.model = model
        self.model_params = {}
        self.model_params["betas"] = initial_betas.clone().detach()
        self.model_params["body_pose"] = initial_pose.clone().detach()

        if initial_global_orient is not None:
            self.model_params["global_orient"] = initial_global_orient.clone().detach()
        else:
            self.model_params["global_orient"] = torch.tensor([[0, 0, 0]], dtype=torch.float32)

        self.__update()

    def __update(self) -> None:
        """ Updates view with current values """
        self.update_smpl_model(**self.model_params)


    def __update_viewer(self,
                        mesh: pyrender.Mesh,
                        clear: bool = False,
                        *args, **kwargs) -> None:
        """ Adds mesh to scene """
        self.render_lock.acquire()

        if clear:
            for node in list(self.scene.mesh_nodes):
                self.scene.remove_node(node)

        self.scene.add(mesh, *args, **kwargs)
        self.render_lock.release()

    def update_smpl_model(
            self,
            *args, **kwargs
        ):
        """ Updates scene """
        output = self.model(
            **kwargs,
            return_verts=True
        )
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, self.model.faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        self.__update_viewer(mesh, clear=True)

        if self.show_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            self.__update_viewer(joints_pcl)

    def update_param_factory(self, index: int, param: str) -> Callable:
        """ Returns function that updates model parameters and view """
        def __update_nth_orient_param(value: str) -> None:
            self.model_params[param][0, index] = float(value)
            self.__update()

        return __update_nth_orient_param
