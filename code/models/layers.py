import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Sequential, Module, Identity, Dropout
from utils.sparse_utils import SparseMat
from utils.pos_enc_utils import get_embedder




def get_linear_layers(feats, final_layer=False, batchnorm=True):
    layers = []

    # Add layers
    for i in range(len(feats) - 2):
        layers.append(Linear(feats[i], feats[i + 1]))

        if batchnorm:
            layers.append(BatchNorm1d(feats[i + 1], track_running_stats=False))

        layers.append(ReLU())

    # Add final layer
    layers.append(Linear(feats[-2], feats[-1]))
    if not final_layer:
        if batchnorm:
            layers.append(BatchNorm1d(feats[-1], track_running_stats=False))

        layers.append(ReLU())

    return Sequential(*layers)


class Parameter3DPts(torch.nn.Module):
    def __init__(self, n_pts):
        super().__init__()

        # Init points randomly
        pts_3d = torch.normal(mean=0, std=0.1, size=(3, n_pts), requires_grad=True)

        self.pts_3d = torch.nn.Parameter(pts_3d)

    def forward(self):
        return self.pts_3d


class SetOfSetLayer(Module):
    def __init__(self, d_in, d_out):
        super(SetOfSetLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)
        self.lin_n = Linear(d_in, d_out)
        self.lin_m = Linear(d_in, d_out)
        self.lin_both = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        out_all = self.lin_all(x.values)  # [all_points_everywhere, d_in] -> [all_points_everywhere, d_out]

        mean_rows = x.mean(dim=0) # [m,n,d_in] -> [n,d_in]
        out_rows = self.lin_n(mean_rows)  # [n,d_in] -> [n,d_out]  # each track's mean representation gets weighted

        mean_cols = x.mean(dim=1) # [m,n,d_in] -> [m,d_in]
        out_cols = self.lin_m(mean_cols)  # [m,d_in] -> [m,d_out]  # each camera's mean representation gets weighted

        out_both = self.lin_both(x.values.mean(dim=0, keepdim=True))  # [1,d_in] -> [1,d_out]

        new_features = (out_all + out_rows[x.indices[1], :] + out_cols[x.indices[0], :] + out_both) / 4  # [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])

        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)

class ProjLayer(Module):
    def __init__(self, d_in, d_out):
        super(ProjLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        new_features = self.lin_all(x.values)  # [nnz,d_in] -> [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class NormalizationLayer(Module):
    def forward(self, x):
        features = x.values
        norm_features = features - features.mean(dim=0, keepdim=True)
        # norm_features = norm_features / norm_features.std(dim=0, keepdim=True)
        return SparseMat(norm_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class ActivationLayer(Module):
    def __init__(self):
        super(ActivationLayer, self).__init__()
        self.relu = ReLU()

    def forward(self, x):
        new_features = self.relu(x.values)
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class IdentityLayer(Module):
    def forward(self, x):
        return x


class EmbeddingLayer(Module):
    def __init__(self, multires, in_dim, split_feat_dim=0):
        """split_feat_dim > 0: positionally encode only the leading
        (in_dim - split_feat_dim) coordinate channels and pass the trailing
        feature channels through a learned linear projection instead (the
        integration ablation); 0 keeps the original all-channel encoding."""
        super(EmbeddingLayer, self).__init__()
        self.split_feat_dim = split_feat_dim
        coord_dim = in_dim - split_feat_dim
        if multires > 0:
            self.embed, d_coord = get_embedder(multires, coord_dim)
        else:
            self.embed, d_coord = (Identity(), coord_dim)
        if split_feat_dim > 0:
            self.feat_proj = torch.nn.Linear(split_feat_dim, d_coord)
            self.d_out = d_coord * 2
        else:
            self.d_out = d_coord

    def forward(self, x):
        if self.split_feat_dim > 0:
            k = x.values.shape[1] - self.split_feat_dim
            embeded_features = torch.cat(
                [self.embed(x.values[:, :k]), self.feat_proj(x.values[:, k:])], dim=1)
        else:
            embeded_features = self.embed(x.values)
        new_shape = (x.shape[0], x.shape[1], embeded_features.shape[1])
        return SparseMat(embeded_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)



class SparseLayerNorm(Module):
    """Layer normalization for sparse tensors"""
    def __init__(self, normalized_shape, eps=1e-5):
        super(SparseLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        # Learnable parameters
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        # Apply layer norm to the values of the sparse matrix
        features = x.values  # [nnz, d]
        
        # Compute mean and variance
        mean = features.mean(dim=-1, keepdim=True)
        var = features.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        norm_features = (features - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable parameters
        norm_features = norm_features * self.gamma + self.beta
        
        return SparseMat(norm_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class SparseDropout(Module):
    """Dropout for sparse tensors"""
    def __init__(self, p=0.1):
        super(SparseDropout, self).__init__()
        self.p = p
        self.dropout = Dropout(p)
    
    def forward(self, x):
        if self.training:
            # Apply dropout to the values
            dropped_features = self.dropout(x.values)
            return SparseMat(dropped_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)
        return x

