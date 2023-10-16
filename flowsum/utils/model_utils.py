from functools import partial

import torch.nn as nn

from pyro.distributions.torch_transform import TransformModule
import pyro.distributions.transforms as T

from flowsum.nf.transforms import (
    PyroPlanar,
    PyroRadial,
    pyrospline_coupling,
    pyrospline,
)


def get_activation(act):
    if act == "tanh":
        act = nn.Tanh()
    elif act == "relu":
        act = nn.ReLU()
    elif act == "softplus":
        act = nn.Softplus()
    elif act == "rrelu":
        act = nn.RReLU()
    elif act == "leakyrelu":
        act = nn.LeakyReLU()
    elif act == "elu":
        act = nn.ELU()
    elif act == "selu":
        act = nn.SELU()
    elif act == "glu":
        act = nn.GLU()
    else:
        print("Defaulting to tanh activations...")
        act = nn.Tanh()
    return act


SUPPORTED_TRANSFORMS = [
    "planar",
    "radial",
    "sylvester",
    "linear_spline",
    "quadratic_spline",
    "rlnsf",  # rational-linear neural spline flows
    "rqnsf",  # rational-quadratic neural spline flows
    "affine_coupling",
    "iaf",  # inverse autoregressive flow
]


def get_pyro_transform_cls_from_name(transform_name: str) -> TransformModule:
    """Return a Pyro Transform class or a helper function for the class."""
    assert (
        transform_name in SUPPORTED_TRANSFORMS
    ), f"We currently only support {SUPPORTED_TRANSFORMS}, whereas your input is '{transform_name}'."

    if transform_name == "planar":  # have passed debugging test
        return PyroPlanar
    elif transform_name == "radial":  # have passed debugging test
        return PyroRadial
    elif transform_name == "sylvester":  # have passed debugging test
        return T.Sylvester
    elif transform_name == "linear_spline":  # have passed debugging test
        return partial(pyrospline, order="linear")
    elif transform_name == "quadratic_spline":  # have passed debugging test
        return partial(pyrospline, order="quadratic")
    elif transform_name == "rlnsf":  # have passed debugging test
        return partial(pyrospline_coupling, order="linear")
    elif transform_name == "rqnsf":  # have passed debugging test
        return partial(pyrospline_coupling, order="quadratic")
    elif transform_name == "affine_coupling":  # have passed debugging test
        return T.affine_coupling
    elif transform_name == "iaf":  # have passed debugging test
        return T.affine_autoregressive
