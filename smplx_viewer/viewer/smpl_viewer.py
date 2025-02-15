from typing import Callable, Dict, Tuple

import pyrender
import trimesh
import torch
import smplx
import numpy as np


class SMPLViewer(pyrender.Viewer):
    """ Wrapper for Pyrender Viewer to dynamically update SMPL-X renders """
    def __init__(self,
                 model: smplx.SMPLX,
                 show_joints: bool = False,
                 viewport_size=None,
                 render_flags=None, viewer_flags=None, registered_keys=None,
                 run_in_thread=False, **kwargs):
        # initialize Pyrender viewer
        scene = pyrender.Scene()
        super().__init__(scene, viewport_size, render_flags, viewer_flags, registered_keys, run_in_thread, **kwargs)

        self.show_joints = show_joints
        self.model = model

        self.model_params = {}
        self.model_params["betas"] = torch.zeros([1, model.num_betas], dtype=torch.float32)
        self.model_params["expression"] = torch.zeros([1, model.num_expression_coeffs], dtype=torch.float32)

        # initialize pose
        mean_pose = torch.from_numpy(model.create_mean_pose(None)).unsqueeze(0)
        body_pose_end = 3 + model.NUM_BODY_JOINTS * 3
        self.model_params["global_orient"] = mean_pose[:, :3]
        self.model_params["body_pose"] = mean_pose[:, 3:body_pose_end]
        self.model_params["jaw_pose"] = mean_pose[:, body_pose_end: body_pose_end + 3]
        self.model_params["leye_pose"] = mean_pose[:, body_pose_end + 3: body_pose_end + 6]
        self.model_params["reye_pose"] = mean_pose[:, body_pose_end + 6: body_pose_end + 9]
        self.model_params["left_hand_pose"] = model.left_hand_pose.clone()
        self.model_params["right_hand_pose"] = model.right_hand_pose.clone()

        self.__update()

    @property
    def n_parameters(self) -> Tuple[Dict[str, int], int]:
        param_count = {}
        param_count["body (incl. body, eyes, jaw)"] = (
           self.model_params["global_orient"].shape[1] +
           self.model_params["body_pose"].shape[1] +
           self.model_params["jaw_pose"].shape[1] +
           self.model_params["leye_pose"].shape[1] +
           self.model_params["reye_pose"].shape[1]
        )
        param_count["shape"] = self.model_params["betas"].shape[1]
        param_count["facial expression"] = self.model_params["expression"].shape[1]
        param_count["hands"] = self.model_params["left_hand_pose"].shape[1] + self.model_params["right_hand_pose"].shape[1]
        total = sum(param_count.values())
        return param_count, total

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
        def __update_nth_param(value: str) -> None:
            self.model_params[param][0, index] = float(value)
            self.__update()

        return __update_nth_param
