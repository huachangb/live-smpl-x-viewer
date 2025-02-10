import tkinter as tk

from smplx.joint_names import SMPLH_JOINT_NAMES

from .scrollable_frame import ScrollFrame
from ..render import SMPLViewer


def computeparam_dim_name(index: int) -> str:
    """ Computes joint name """
    k = index // 3
    return SMPLH_JOINT_NAMES[1 + k]


def create_parameter_frame(
        parent: tk.Misc,
        param: str,
        viewer: SMPLViewer,
        from_: float,
        to_: float,
        resolution: float,
        enable_scroll: bool = True
    ) -> tk.Frame:
    """ Adds frame with a slider for each param dimension. """
    if enable_scroll:
        frame = ScrollFrame(parent)
        frame_viewport = frame.viewPort
    else:
        frame = tk.Frame(parent)
        frame_viewport = frame

    label = tk.Label(frame, text=param)
    label.pack()

    param_dim = viewer.model_params[param].shape[1]

    for i in range(param_dim):
        if param == "body_pose":
            slider_label = computeparam_dim_name(i)
        elif param == "global_orient":
            slider_label = "pelvis"
        else:
            slider_label = f"{param} {i}"

        update_param_dim_fn = viewer.update_param_factory(index=i, param=param)
        slider = tk.Scale(
            frame_viewport,
            from_=from_,
            to=to_,
            resolution=resolution,
            orient="horizontal",
            command=update_param_dim_fn,
            label=slider_label
        )
        slider.pack()

    return frame

