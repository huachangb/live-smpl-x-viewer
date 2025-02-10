from typing import Any

import os.path as osp
import argparse

import tkinter as tk

import pyrender
import torch
import smplx

from smplx_viewer.viewer import SMPLViewer
from smplx_viewer.tkinter import create_parameter_frame


def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         num_expression_coeffs=10,
         use_face_contour=False):

    # create model
    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)

    # create Pyrender viewer
    scene = pyrender.Scene()
    viewer = SMPLViewer(
        model=model,
        scene=scene,
        use_raymond_lighting=True,
        run_in_thread=True,
        show_joints=True
    )

    # create Tkinter stuff
    root = tk.Tk()
    root.title("Parameters")

    print(viewer.model_params.keys())

    for i, param in enumerate(viewer.model_params):
        orient_frame = create_parameter_frame(
            parent=root,
            param=param,
            viewer=viewer,
            from_=0,
            to_=2 * torch.pi,
            resolution=2 * torch.pi / 100,
        )
        orient_frame.grid(row=0, column=i)



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