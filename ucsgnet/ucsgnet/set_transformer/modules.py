import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import typing as t


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetCSGLayer(nn.Module):
    def __init__(self,
                 num_in_shapes: int,
                 features_dim: int,
                 num_out_shapes: int,
                 latent_size: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.heads_num = 1
        self.num_in_shapes = num_in_shapes
        self.num_out_shapes = num_out_shapes
        self.latent_size = latent_size
        self.features_dim = features_dim

        self.encoder1 = MAB(latent_size, features_dim, hidden_dim, self.heads_num)
        self.encoder2 = MAB(hidden_dim, hidden_dim, hidden_dim, self.heads_num)

        self.decoder_pma = PMA(hidden_dim, 1, num_out_shapes)
        self.decoder_head1 = MAB(hidden_dim, hidden_dim, features_dim, self.heads_num)
        self.decoder_head2 = MAB(hidden_dim, hidden_dim, features_dim, self.heads_num)

        self.operations_before_clamping: t.Optional[torch.Tensor] = None
        self.operations_after_clamping: t.Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        # x: (batch_size,  num_shapes, num_points)
        # latent_vector: (batch_size, latent_size)

        latent_vector = latent_vector.expand(
            x.shape[1],
            latent_vector.shape[0],
            latent_vector.shape[1]
        ).permute(1, 0, 2)  # shape (batch_size, num_shapes, latent_size)

        z = self.encoder1(latent_vector, x)
        z = self.encoder2(z, z)

        decoded = self.decoder_pma(z)
        head1 = self.decoder_head1(decoded, decoded)
        head2 = self.decoder_head2(decoded, decoded)  # (batch, num_output_shapes, num_points)

        head1 = head1.permute(0, 2, 1)
        head2 = head2.permute(0, 2, 1)

        out = torch.cat(
            [
                head1 + head2,  # union
                head1 + head2 - 1,  # intersection
                head1 - head2,  # diff
                head2 - head1  # diff
            ],
            dim=-1
        )

        self.operations_before_clamping = out
        out = out.clamp(0, 1)
        self.operations_after_clamping = out
        return out
