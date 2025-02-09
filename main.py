import pyrender
import trimesh
import torch
import smplx
import numpy as np
import argparse
import os.path as osp
from typing import Any
import tkinter as tk


from smplx_viewer.render import update_smpl_model



GLOBAL_ORIENT = torch.tensor([[0, 0, 0]], dtype=torch.float32)


def update_orient_factory(index: str, *args, **kwargs) -> callable:
    def update_nth_orient_param(value: str) -> None:
        GLOBAL_ORIENT[0, index] = float(value)
        kwargs["global_orient"] = GLOBAL_ORIENT
        update_smpl_model(*args, **kwargs)

    return update_nth_orient_param



def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         num_expression_coeffs=10,
         use_face_contour=False):

    # create Pyrender viewer
    scene = pyrender.Scene()
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)


    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)

    torch.manual_seed(42)
    betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)

    # create Tkinter stuff
    root = tk.Tk()
    root.title("Global orientation")

    config = dict(
        model=model,
        viewer=viewer,
        scene=scene,
        betas=betas,
        expression=expression,
        plot_joints=plot_joints
    )

    update_smpl_model(global_orient=GLOBAL_ORIENT, **config)

    slider1 = tk.Scale(root, from_=0, to=2 * torch.pi, resolution=2 * torch.pi / 25, orient="horizontal", command=update_orient_factory(0, **config))
    slider2 = tk.Scale(root, from_=0, to=2 * torch.pi, resolution=2 * torch.pi / 25, orient="horizontal", command=update_orient_factory(1, **config))
    slider3 = tk.Scale(root, from_=0, to=2 * torch.pi, resolution=2 * torch.pi / 25, orient="horizontal", command=update_orient_factory(2, **config))

    slider1.pack()
    slider2.pack()
    slider3.pack()

    root.mainloop()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         use_face_contour=use_face_contour)