"""
This file contains the nf-enhanced latent model, which includes the inference network q(z0|x) and the 
    normalizing flow model.
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from flowsum.nf.model import NormalizingFlowModel, LegacyNormalizingFlowModel
from flowsum.utils.model_utils import (
    get_activation,
    get_flow_class_from_name,
    get_pyro_transform_cls_from_name,
)
from flowsum.nf.transforms import generalized_permute


class NFLatentModel(nn.Module):
    """This class is equivalent to LegacyNFLatentModel, except that it is built on top of the Pyro package,
    whereas the NormalizingFlowModel class is built on top of the functions I myself wrote.
    """

    def __init__(
        self,
        input_size,
        q_hidden_dims: List[int] = [200, 200],
        q_act="tanh",
        q_dropout=0.1,
        nf_name="planar",
        nf_latent_size=300,
        nf_num_layers=4,
    ) -> None:
        """
        Args:
            input_size: the size of the input to the nflatent model.
            q_hidden_dims: the hidden dimensions of the q inference network, should be a list of integers.
            q_act: the activation function used in the q inference network.
            q_dropout: the dropout rate used in the q inference network.
            nf_name: the name of the normalizing flows. The current supported list is
                ["planar", "radial", "realnvp"].
            nf_latent_size: the size of the latent vector z_0, z_1, ..., z_K.
            nf_num_layers: the number of layers of normalizing flows.

        Note:
            (1) Currently, the q_net is default as a 3-layer MLP. Can customize this for more flexibility.
            (2) self.q_net is used as the inference network, which learns the mu and log_sigma from the average embedding
                of the input text.
            (3) self.nf_model takes in mu and log_sigma, and outputs z_K, log q(z_K) and log p(z_K | x).
            (4) Currently, we only consider nf_latent_size as the parameter for defining the transforms.
                [TODO] Can extend to incorporate more arguments.
            (5) The reason for not using Permute in pyro is that Permute there is not a subclass of nn.Module.
        """
        super().__init__()
        self.nf_latent_size = nf_latent_size

        layers = [
            nn.Linear(input_size, q_hidden_dims[0], bias=True),
            get_activation(q_act),
        ]
        for i in range(len(q_hidden_dims) - 1):
            layers += [
                nn.Linear(q_hidden_dims[i], q_hidden_dims[i + 1], bias=True),
                get_activation(q_act),
            ]
        layers += [
            nn.Dropout(q_dropout),
            nn.Linear(q_hidden_dims[-1], 2 * nf_latent_size),
        ]
        self.q_net = nn.Sequential(*layers)

        transform_cls = get_pyro_transform_cls_from_name(transform_name=nf_name)
        transforms = [transform_cls(nf_latent_size) for _ in range(nf_num_layers)]
        if nf_name.endswith("_coupling") or nf_name == "iaf":
            permutations = [
                generalized_permute(nf_latent_size) for _ in range(nf_num_layers)
            ]  # use torch.randperm() by default, [TODO] debug Permute is not a subclass of nn.Module
            transforms = [
                val for pair in zip(transforms, permutations) for val in pair
            ][
                :-1
            ]  # interweave coupling and permuation, and drop the last permutation
        self.nf_model = NormalizingFlowModel(transforms=transforms)

    def forward(self, nf_input) -> Tuple:
        """
        Args:
            nf_input: (batch_size, input_size)
                - if q_input_type == "avg_embed", then it is the average embedding of the input source
                text of shape (batch_size, embed_size);
                - if q_input_type == "bows", then it is the bag-of-words of shape (batch_size, vocab_size).

        Returns:
            nf_latent:  z_K,             (batch_size, nf_latent_size)
            log_q:      log q_K(z_K),    (batch_size, )
            log_prior:  log p(z_K|x),    (batch_size, )
        """
        mu, log_sigma = torch.split(
            self.q_net(nf_input), split_size_or_sections=self.nf_latent_size, dim=-1
        )
        return self.nf_model(
            mu=mu, log_sigma=log_sigma
        )  # returns a tuple (nf_latent, log_q, log_prior)


class LegacyNFLatentModel(nn.Module):
    """Legacy NFLatentModel, built on top of LegacyNormalizingFLowModel, which is based on self-written nf functions."""

    def __init__(
        self,
        embed_size,
        q_hidden_size=200,
        q_act="tanh",
        q_dropout=0.1,
        nf_name="planar",
        nf_latent_size=300,
        nf_num_layers=4,
    ) -> None:
        """
        Args:
            q_hidden_size: the hidden size of the q inference network.
            q_act: the activation function used in the q inference network.
            q_dropout: the dropout rate used in the q inference network.
            nf_name: the name of the normalizing flows. The current supported list is
                ["planar", "radial", "realnvp"].
            nf_latent_size: the size of the latent vector z_0, z_1, ..., z_K.
            nf_num_layers: the number of layers of normalizing flows.

        Note:
            (1) Currently, the q_net is default as a 3-layer MLP. Can customize this for more flexibility.
        """
        super().__init__()
        self.nf_latent_size = nf_latent_size
        self.q_net = nn.Sequential(
            nn.Linear(embed_size, q_hidden_size, bias=True),
            get_activation(q_act),
            nn.Linear(q_hidden_size, q_hidden_size, bias=True),
            get_activation(q_act),
            nn.Dropout(q_dropout),
            nn.Linear(q_hidden_size, 2 * nf_latent_size),
        )

        flow_class = get_flow_class_from_name(flow_name=nf_name)
        self.nf_model = LegacyNormalizingFlowModel(
            flows=[flow_class(latent_size=nf_latent_size) for _ in range(nf_num_layers)]
        )

    def forward(self, avg_embed) -> Tuple:
        """
        Args:
            avg_embed: the average embedding of the input source text. (batch_size, embed_size)

        Returns:
            nf_latent:  z_K,             (batch_size, nf_latent_size)
            log_q:      log q_K(z_K),    (batch_size, )
            log_prior:  log p(z_K|x),    (batch_size, )
        """
        mu, log_sigma = torch.split(
            self.q_net(avg_embed), split_size_or_sections=self.nf_latent_size, dim=-1
        )
        return self.nf_model(
            mu=mu, log_sigma=log_sigma
        )  # returns a tuple (nf_latent, log_q, log_prior)
