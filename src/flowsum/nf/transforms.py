from functools import partial

import torch
import torch.nn.functional as F
from torch.distributions import Transform

from pyro.distributions.transforms import (
    Planar,
    Radial,
    Spline,
    ConditionalSpline,
    SplineCoupling,
    AffineAutoregressive,
)
from pyro.distributions.transforms.spline import (
    _calculate_knots,
    _searchsorted,
    _select_bins,
    ConditionedSpline,
)
from pyro.nn import DenseNN, AutoRegressiveNN
from pyro.distributions import constraints
from pyro.distributions.torch_transform import TransformModule


class PyroPlanar(Planar):
    """Modified version of Planar from Pyro.

    The reason for the modification is that the existence of self._params() will make
        the parameters on different devices from the inputs, and hence not suitable
        for DataParallel on multiple GPUs. Therefore, we don't call self._params()
        in the self._call() function.

    Note:
        (1) Sylvester and AffineCoupling don't have similar issues, so we make no changes.
    """

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x => y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        bias, u, w = self.bias, self.u, self.w

        # x ~ (batch_size, dim_size, 1)
        # w ~ (batch_size, 1, dim_size)
        # bias ~ (batch_size, 1)
        act = torch.tanh(
            torch.matmul(w.unsqueeze(-2), x.unsqueeze(-1)).squeeze(-1) + bias
        )
        u_hat = self.u_hat(u, w)
        y = x + u_hat * act

        psi_z = (1.0 - act.pow(2)) * w
        self._cached_logDetJ = torch.log(
            torch.abs(
                1
                + torch.matmul(psi_z.unsqueeze(-2), u_hat.unsqueeze(-1))
                .squeeze(-1)
                .squeeze(-1)
            )
        )

        return y


class PyroRadial(Radial):
    """Modified version of Radial from Pyro.

    The reason for the modification is that the existence of self._params() will make
        the parameters on different devices from the inputs, and hence not suitable
        for DataParallel on multiple GPUs. Therefore, we don't call self._params()
        in the self._call() function.
    """

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from the base distribution (or the output
        of a previous transform)
        """
        x0, alpha_prime, beta_prime = self.x0, self.alpha_prime, self.beta_prime

        # Ensure invertibility using approach in appendix A.2
        alpha = F.softplus(alpha_prime)
        beta = -alpha + F.softplus(beta_prime)

        # Compute y and logDet using Equation 14.
        diff = x - x0
        r = diff.norm(dim=-1, keepdim=True)
        h = (alpha + r).reciprocal()
        h_prime = -(h**2)
        beta_h = beta * h

        self._cached_logDetJ = (
            (x0.size(-1) - 1) * torch.log1p(beta_h)
            + torch.log1p(beta_h + beta * h_prime * r)
        ).sum(-1)
        return x + beta_h * diff


class PyroSpline(Spline):
    """Modified version of Pyro's Spline transformation.

    (1) Bug 1 -- RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Half for the source.

        Solution: Modified _monotonic_rational_spline() at the very end: force the outputs'
            dtype to be the same as inputs' dtype. Many operations inside this function
            generates torch.float32, including _calculate_knots() and .float(). So
            for convenience, I directly modified the final outputs' dtype. Also, note
            that self._params() also gives torch.float32.
            - [TODO] need to check the effect of mixed precision training.

        Note: the code for self.spline_op() is the same as in Spline, but now it
            is calling the _monotonic_rational_spline() function defined in this
            file.

    (2) Bug 2 -- The result given by `log_q = flow_dist.log_prob(z)` has size
        (batch_size, latent_size) when it is supposed to be (batch_size, ) when
        transformations are applied. Also, most of the values are positive, when
        log probabilities should all be negative.

        Solution: This is actually not a bug, but by design. In the SplineCoupling._call(),
            we need the log det in the form of tensors for concatenation with the conditional
            log det. This is for coding convenience. As we can see, in SplineCoupling.log_abs_det_jacobian(),
            sum(-1) is applied to get the log det in the size of (batch_size, ). Therefore,
            we don't modify the class here, but add a condition checking in nf/model.py such
            that if the transform is PyroSpline, then we do a sum(-1) operation. As for the
            positivity, this is not an issue as log_prob is actually calculating log density,
            and density can be greater than 1.

    (3) Bug 3 -- Get the same error as Planar, on different devices. The root
        cause, again, is calling self._params(). Calling it will move the outputs
        to cuda:0 when the inputs are in cuda:1 or other devices.

        Solution: don't call self._params(), and put the code there directly
            inside self.spline_op().
    """

    def spline_op(self, x, **kwargs):
        # widths, unnormalized_widths ~ (input_dim, num_bins)
        w = F.softmax(self.unnormalized_widths, dim=-1)
        h = F.softmax(self.unnormalized_heights, dim=-1)
        d = F.softplus(self.unnormalized_derivatives)
        if self.order == "linear":
            l = torch.sigmoid(self.unnormalized_lambdas)
        else:
            l = None
        y, log_detJ = _monotonic_rational_spline(
            x, w, h, d, l, bound=self.bound, **kwargs
        )
        return y, log_detJ


class PyroConditionedSpline(ConditionedSpline):
    """
    When used in ConditionalSpline, self._params is no longer callable,
        so we don't have similar issues as PyroSpline Bug 3.
    """

    def spline_op(self, x, **kwargs):
        w, h, d, l = self._params() if callable(self._params) else self._params
        y, log_detJ = _monotonic_rational_spline(
            x, w, h, d, l, bound=self.bound, **kwargs
        )
        return y, log_detJ


class PyroConditionalSpline(ConditionalSpline):
    """Modified version of Pyro's ConditionalSpline transformation."""

    def condition(self, context):
        params = partial(self._params, context)
        return PyroConditionedSpline(params, bound=self.bound, order=self.order)


class PyroSplineCoupling(SplineCoupling):
    """Modified version of Pyro's SplineCoupling.

    (1) Bug 1 -- RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Half for the source.

        Solution: Modified the __init__() to call PyroSpline and PyroConditionalSpline to
            initize self.lower_spline.

    (2) Bug 2 -- RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!

        Solution: Modify ConditionalSpline to be PyroConditionalSpline.

    (3) Bug 3 -- Unlike the bug in Spline, the shape of log_q is (batch_size, ),
        as expected, but the values are positive.

        Solution: It is okay to have positive log_q, since q is density instead of probability.
    """

    def __init__(
        self,
        input_dim,
        split_dim,
        hypernet,
        count_bins=8,
        bound=3.0,
        order="linear",
        identity=False,
    ):
        super(SplineCoupling, self).__init__(cache_size=1)

        # One part of the input is (optionally) put through an element-wise spline and the other part through a
        # conditional one that inputs the first part.
        self.lower_spline = PyroSpline(split_dim, count_bins, bound, order)
        self.upper_spline = PyroConditionalSpline(
            hypernet, input_dim - split_dim, count_bins, bound, order
        )
        self.split_dim = split_dim
        self.identity = identity


def pyrospline(input_dim, **kwargs):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.Spline` object for consistency with
    other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int

    """

    return PyroSpline(input_dim, **kwargs)


def pyrospline_coupling(
    input_dim, split_dim=None, hidden_dims=None, count_bins=4, bound=3.0, order="linear"
):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.SplineCoupling` object for consistency
    with other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int

    """

    if split_dim is None:
        split_dim = input_dim // 2

    if hidden_dims is None:
        hidden_dims = [input_dim, input_dim]

    if order == "linear":
        nn = DenseNN(
            split_dim,
            hidden_dims,
            param_dims=[
                (input_dim - split_dim) * count_bins,  # w
                (input_dim - split_dim) * count_bins,  # h
                (input_dim - split_dim) * (count_bins - 1),  # d
                (input_dim - split_dim) * count_bins,  # l
            ],
        )
    elif order == "quadratic":
        nn = DenseNN(
            split_dim,
            hidden_dims,
            param_dims=[
                (input_dim - split_dim) * count_bins,  # w
                (input_dim - split_dim) * count_bins,  # h
                (input_dim - split_dim) * (count_bins - 1),  # d
            ],
        )
    else:
        raise ValueError(
            f"Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{order}' was found!"
        )

    return PyroSplineCoupling(input_dim, split_dim, nn, count_bins, bound, order)


def _monotonic_rational_spline(
    inputs,
    widths,
    heights,
    derivatives,
    lambdas=None,
    inverse=False,
    bound=3.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    min_lambda=0.025,
    eps=1e-6,
):
    """
    Calculating a monotonic rational spline (linear or quadratic) or its inverse,
    plus the log(abs(detJ)) required for normalizing flows.
    NOTE: I omit the docstring with parameter descriptions for this method since it
    is not considered "public" yet!
    """

    # Ensure bound is positive
    # NOTE: For simplicity, we apply the identity function outside [-B, B] X [-B, B] rather than allowing arbitrary
    # corners to the bounding box. If you want a different bounding box you can apply an affine transform before and
    # after the input
    assert bound > 0.0

    num_bins = widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    # inputs, inside_interval_mask, outside_interval_mask ~ (batch_dim, input_dim)
    left, right = -bound, bound
    bottom, top = -bound, bound
    inside_interval_mask = (inputs >= left) & (inputs <= right)
    outside_interval_mask = ~inside_interval_mask

    # outputs, logabsdet ~ (batch_dim, input_dim)
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # For numerical stability, put lower/upper limits on parameters. E.g .give every bin min_bin_width,
    # then add width fraction of remaining length
    # NOTE: Do this here rather than higher up because we want everything to ensure numerical
    # stability within this function
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    derivatives = min_derivative + derivatives

    # Cumulative widths are x (y for inverse) position of knots
    # Similarly, cumulative heights are y (x for inverse) position of knots
    widths, cumwidths = _calculate_knots(widths, left, right)
    heights, cumheights = _calculate_knots(heights, bottom, top)

    # Pad left and right derivatives with fixed values at first and last knots
    # These are 1 since the function is the identity outside the bounding box and the derivative is continuous
    # NOTE: Not sure why this is 1.0 - min_derivative rather than 1.0. I've copied this from original implementation
    derivatives = F.pad(
        derivatives, pad=(1, 1), mode="constant", value=1.0 - min_derivative
    )

    # Get the index of the bin that each input is in
    # bin_idx ~ (batch_dim, input_dim, 1)
    bin_idx = _searchsorted(
        cumheights + eps if inverse else cumwidths + eps, inputs
    ).unsqueeze(-1)

    # Select the value for the relevant bin for the variables used in the main calculation
    input_widths = _select_bins(widths, bin_idx)
    input_cumwidths = _select_bins(cumwidths, bin_idx)
    input_cumheights = _select_bins(cumheights, bin_idx)
    input_delta = _select_bins(heights / widths, bin_idx)
    input_derivatives = _select_bins(derivatives, bin_idx)
    input_derivatives_plus_one = _select_bins(derivatives[..., 1:], bin_idx)
    input_heights = _select_bins(heights, bin_idx)

    # Calculate monotonic *linear* rational spline
    if lambdas is not None:
        lambdas = (1 - 2 * min_lambda) * lambdas + min_lambda
        input_lambdas = _select_bins(lambdas, bin_idx)

        # The weight, w_a, at the left-hand-side of each bin
        # We are free to choose w_a, so set it to 1
        wa = 1.0

        # The weight, w_b, at the right-hand-side of each bin
        # This turns out to be a multiple of the w_a
        # TODO: Should this be done in log space for numerical stability?
        wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa

        # The weight, w_c, at the division point of each bin
        # Recall that each bin is divided into two parts so we have enough d.o.f. to fit spline
        wc = (
            input_lambdas * wa * input_derivatives
            + (1 - input_lambdas) * wb * input_derivatives_plus_one
        ) / input_delta

        # Calculate y coords of bins
        ya = input_cumheights
        yb = input_heights + input_cumheights
        yc = ((1.0 - input_lambdas) * wa * ya + input_lambdas * wb * yb) / (
            (1.0 - input_lambdas) * wa + input_lambdas * wb
        )

        if inverse:
            numerator = (input_lambdas * wa * (ya - inputs)) * (
                inputs <= yc
            ).float() + (
                (wc - input_lambdas * wb) * inputs + input_lambdas * wb * yb - wc * yc
            ) * (
                inputs > yc
            ).float()

            denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (
                inputs <= yc
            ).float() + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

            theta = numerator / denominator

            outputs = theta * input_widths + input_cumwidths

            derivative_numerator = (
                wa * wc * input_lambdas * (yc - ya) * (inputs <= yc).float()
                + wb * wc * (1 - input_lambdas) * (yb - yc) * (inputs > yc).float()
            ) * input_widths

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(
                torch.abs(denominator)
            )  # [TODO] need to check if - is needed before the equation.

        else:
            theta = (inputs - input_cumwidths) / input_widths

            numerator = (wa * ya * (input_lambdas - theta) + wc * yc * theta) * (
                theta <= input_lambdas
            ).float() + (wc * yc * (1 - theta) + wb * yb * (theta - input_lambdas)) * (
                theta > input_lambdas
            ).float()

            denominator = (wa * (input_lambdas - theta) + wc * theta) * (
                theta <= input_lambdas
            ).float() + (wc * (1 - theta) + wb * (theta - input_lambdas)) * (
                theta > input_lambdas
            ).float()

            outputs = numerator / denominator

            derivative_numerator = (
                wa * wc * input_lambdas * (yc - ya) * (theta <= input_lambdas).float()
                + wb
                * wc
                * (1 - input_lambdas)
                * (yb - yc)
                * (theta > input_lambdas).float()
            ) / input_widths

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(
                torch.abs(denominator)
            )

    # Calculate monotonic *quadratic* rational spline
    else:
        if inverse:
            a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            ) + input_heights * (input_delta - input_derivatives)
            b = input_heights * input_derivatives - (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            )
            c = -input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            # Make sure outside_interval input can be reversed as identity.
            discriminant = discriminant.masked_fill(outside_interval_mask, 0)
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            logabsdet = -(torch.log(derivative_numerator) - 2 * torch.log(denominator))

        else:
            theta = (inputs - input_cumwidths) / input_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (
                input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
            )
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    # Apply the identity function outside the bounding box
    outputs = outputs.to(dtype=inputs.dtype)
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0.0
    return outputs, logabsdet


class ConditionedGeneralizedPermute(Transform):
    domain = constraints.independent(constraints.real, 3)
    codomain = constraints.independent(constraints.real, 3)
    bijective = True

    def __init__(self, permutation=None, LU=None):
        super(ConditionedGeneralizedPermute, self).__init__(cache_size=1)

        self.permutation = permutation
        self.LU = LU

    @property
    def U_diag(self):
        return self.LU.diag()

    @property
    def L(self):
        return self.LU.tril(diagonal=-1) + torch.eye(
            self.LU.size(-1), dtype=self.LU.dtype, device=self.LU.device
        )

    @property
    def U(self):
        return self.LU.triu()

    def _call(self, x):
        """
        x: (batch_size, latent_size)

        When u: (latent_size, ), W @ u: (latent_size);
        x^T: (latent_size, batch_size), W @ x^T: (latent_size, batch_size)
        => y = x @ W^T: (batch_size, latent_size)
        """
        W = self.permutation @ self.L @ self.U  # (latent_size, latent_size)
        y = x @ W.T  # (batch_size, latent_size)
        return y

    def _inverse(self, y: torch.Tensor):
        LUx_T = self.permutation.T @ y.T
        Ux_T = torch.linalg.solve_triangular(self.L, LUx_T, upper=False)
        x = torch.linalg.solve_triangular(self.U, Ux_T, upper=True).T
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs(det(dy/dx))).

        We apply the same permutation to all instances in the batch, so the log det is the
        same.
        """
        log_det = self.U_diag.abs().log().sum()  # (1,)
        return log_det * torch.ones(
            x.shape[0], dtype=x.dtype, layout=x.layout, device=x.device
        )  # (batch_size, )


class GeneralizedPermute(ConditionedGeneralizedPermute, TransformModule):
    r"""Perform the generalized permutation (invertible linear transformation) as introduced in
    Durkan et al. 2019.

    The permutation is performed on the last dimension of the input.

    W = PLU
        - P is a fixed permuation matrix.
        - L is a lower triangular matrix with ones on the diagonal.
        - U is an upper triangular matrix.

    References:
        (1) The code is modified based on pyro's implementation: https://docs.pyro.ai/en/stable/_modules/pyro/distributions/transforms/generalized_channel_permute.html#ConditionalGeneralizedChannelPermute.
    """

    domain = constraints.independent(constraints.real, 3)
    codomain = constraints.independent(constraints.real, 3)
    bijective = True

    def __init__(self, permutation=None):
        super(GeneralizedPermute, self).__init__()
        self.__delattr__("permutation")

        input_dim = len(permutation)
        # Sample a random orthogonal matrix
        W, _ = torch.linalg.qr(torch.randn(input_dim, input_dim))

        # Construct the partially pivoted LU-form and the pivots
        LU, pivots = torch.linalg.lu_factor(W)

        # Convert the pivots into the permutation matrix
        if permutation is None:
            P, _, _ = torch.lu_unpack(LU, pivots)
        else:
            P = torch.eye(input_dim, input_dim)[permutation.type(dtype=torch.int64)]

        # We register the permutation matrix so that the model can be serialized
        self.register_buffer("permutation", P)

        # NOTE: For this implementation I have chosen to store the parameters densely, rather than
        # storing L, U, and s separately
        self.LU = torch.nn.Parameter(LU)


def generalized_permute(input_dim: int, permutation: torch.LongTensor = None):
    """
    A helper function to create a GeneralizedPermute object.

    The permutation is performed on the last dimension of the input.
    """
    if permutation is None:
        permutation = torch.randperm(input_dim)
    return GeneralizedPermute(permutation)
