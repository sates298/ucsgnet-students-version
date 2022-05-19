import torch
from torch.nn import functional as F

from ucsgnet.flows.nflows_impl import StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation


class SimpleRealNVP(Flow):
    def __init__(
            self,
            features,
            hidden_features,
            context_features=None,
            num_layers=4,
            num_blocks_per_layer=2,
            use_volume_preserving=False,
            activation=F.relu,
            dropout_probability=0.0,
            batch_norm_within_layers=False,
            batch_norm_between_layers=False
    ):
        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_residual(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_residual
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        # layers = []
        # for _ in range(num_layers):
        #     layers.append(ReversePermutation(features=features))
        #     layers.append(MaskedAffineAutoregressiveTransform(features=features, hidden_features=hidden_features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features])
        )


class MaskedAutoregressiveFlow(Flow):
    def __init__(
            self,
            features,
            hidden_features,
            context_features=None,
            embedding_features=None,
            num_layers=4,
            num_blocks_per_layer=2,
            use_residual_blocks=True,
            use_random_masks=False,
            use_random_permutations=False,
            activation=F.relu,
            dropout_probability=0.0,
            batch_norm_within_layers=False,
            batch_norm_between_layers=False,
    ):

        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    context_features=context_features if embedding_features is None else embedding_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        embedding_net = torch.nn.Identity() if embedding_features is None else torch.nn.Linear(context_features, embedding_features)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
            embedding_net=embedding_net
        )