import torch
from torch import nn
import utils.dataset_utils
from models.baseNet import BaseNet
from models.layers import *
from models.layers import SparseLayerNorm, SparseDropout
from utils.sparse_utils import SparseMat
from datasets.SceneData import SceneData
from utils import general_utils
from utils.Phases import Phases

class SetOfSetBlock(nn.Module):
    def __init__(self, d_in, d_out, conf):
        super(SetOfSetBlock, self).__init__()
        self.block_size = conf.get_int("model.block_size")
        self.use_skip = conf.get_bool("model.use_skip")
        type_name = conf.get_string("model.layer_type", default='SetOfSetLayer')
        self.layer_type = general_utils.get_class("models.layers." + type_name)
        self.layer_kwargs = dict(conf['model'].get('layer_extra_params', {}))

        modules = []
        modules.extend([self.layer_type(d_in, d_out, **self.layer_kwargs), NormalizationLayer()])
        for i in range(1, self.block_size):
            modules.extend([ActivationLayer(), self.layer_type(d_out, d_out, **self.layer_kwargs), NormalizationLayer()])
        self.layers = nn.Sequential(*modules)

        self.final_act = ActivationLayer()

        if self.use_skip:
            if d_in == d_out:
                self.skip = IdentityLayer()
            else:
                self.skip = nn.Sequential(ProjLayer(d_in, d_out), NormalizationLayer())

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        xl = self.layers(x)
        if self.use_skip:
            xl = self.skip(x) + xl

        out = self.final_act(xl)
        return out

class DeepSetOfSetBlock(nn.Module):
    """Enhanced block with residual connections, layer norm, and dropout"""
    def __init__(self, d_in, d_out, conf, use_residual=True, use_layer_norm=True):
        super(DeepSetOfSetBlock, self).__init__()
        self.block_size = conf.get_int("model.block_size", default=2)
        self.use_skip = conf.get_bool("model.use_skip", default=True)
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        dropout_rate = conf.get_float("model.dropout_rate", default=0.1)
        
        # For backward compatibility with original SetOfSetOutliersNet
        type_name = conf.get_string("model.layer_type", default='SetOfSetLayer')
        self.layer_type = general_utils.get_class("models.layers." + type_name)
        self.layer_kwargs = dict(conf.get('model.layer_extra_params', {})) if 'model.layer_extra_params' in conf else {}
        
        modules = []
        
        # First layer in block
        modules.append(self.layer_type(d_in, d_out, **self.layer_kwargs))
        if self.use_layer_norm:
            modules.append(SparseLayerNorm(d_out))
        else:
            modules.append(NormalizationLayer())
        
        # Additional layers in block
        for i in range(1, self.block_size):
            modules.append(ActivationLayer())
            
            # Add dropout for regularization in deep networks
            if dropout_rate > 0 and self.training:
                modules.append(SparseDropout(dropout_rate))
            
            modules.append(self.layer_type(d_out, d_out, **self.layer_kwargs))
            
            if self.use_layer_norm:
                modules.append(SparseLayerNorm(d_out))
            else:
                modules.append(NormalizationLayer())
        
        self.layers = nn.Sequential(*modules)
        self.final_act = ActivationLayer()
        
        # Skip connection for the entire block
        if self.use_skip:
            if d_in == d_out:
                self.skip = IdentityLayer()
            else:
                self.skip = nn.Sequential(ProjLayer(d_in, d_out))
                if self.use_layer_norm:
                    self.skip.add_module('norm', SparseLayerNorm(d_out))
                else:
                    self.skip.add_module('norm', NormalizationLayer())

    def forward(self, x):
        # Process through layers
        xl = self.layers(x)
        
        # Add skip connection if enabled
        if self.use_skip:
            skip_out = self.skip(x)
            xl_values = xl.values + skip_out.values * 0.707  # Scale for stability
            xl = SparseMat(xl_values, xl.indices, xl.cam_per_pts, xl.pts_per_cam, xl.shape)
        
        out = self.final_act(xl)
        return out


class DeepSetOfSetOutliersNet(BaseNet):
    """Deep SetOfSet network with outlier detection capabilities - FIXED VERSION"""
    def __init__(self, conf, phase=None):
        super(DeepSetOfSetOutliersNet, self).__init__(conf)
        
        # Deep network default configuration parameters
        num_blocks = conf.get_int('model.num_blocks', default=2)
        block_size = conf.get_int('model.block_size', default=3)
        num_feats = conf.get_int('model.num_features', default=256)
        multires = conf.get_int('model.multires', default=0)
        
        # Deep architecture features enabled by default
        use_progressive = conf.get_bool('model.use_progressive', default=False)
        use_layer_norm = conf.get_bool('model.use_layer_norm', default=True)
        use_residual = conf.get_bool('model.use_residual', default=True)
        dropout_rate = conf.get_float('model.dropout_rate', default=0.1)
        
        # Store config for progressive training
        self.use_progressive = use_progressive
        self.current_depth = 1 if use_progressive else num_blocks
        self.max_depth = num_blocks
        
        total_layers = num_blocks * block_size
        
        # Output dimensions
        n_d_out = 3  # 3D points output dimension
        m_d_out = self.out_channels  # Camera parameters output dimension
        d_in = 2 + conf.get_int('model.obs_feat_dim', 0)  # 2D obs (+ optional per-obs feature channels)
        
        # Input embedding layer with optional positional encoding
        self.embed = EmbeddingLayer(multires, d_in)
        
        # Create equivariant blocks for permutation-invariant processing
        self.equivariant_blocks = torch.nn.ModuleList()
        
        # FIXED: First block - explicitly pass use_residual parameter
        # Set to False for first block to match DeepSetOfSetNet pattern
        self.equivariant_blocks.append(
            DeepSetOfSetBlock(
                self.embed.d_out, 
                num_feats, 
                conf,
                use_residual=True,           # ← NOW PASSED! First block: with residual
                use_layer_norm=use_layer_norm # ← NOW PASSED!
            )
        )
        
        # FIXED: Additional blocks with residual connections for deeper layers
        for i in range(1, num_blocks):
            # Enable residual connections starting from the third block for stability
            # This matches the pattern in DeepSetOfSetNet
            use_residual_block = use_residual  # Enable for all blocks including Block 1
            
            self.equivariant_blocks.append(
                DeepSetOfSetBlock(
                    num_feats, 
                    num_feats, 
                    conf,
                    use_residual=use_residual_block,  # ← NOW PASSED! Actually used
                    use_layer_norm=use_layer_norm     # ← NOW PASSED!
                )
            )
        
        # Dropout layer for regularization (used in prediction heads)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Camera parameters prediction head with intermediate layers
        self.m_net = nn.Sequential(
            Linear(num_feats, num_feats),
            ReLU(),
            self.dropout,
            Linear(num_feats, num_feats // 2),
            ReLU(),
            self.dropout,
            Linear(num_feats // 2, m_d_out)
        )
        
        # 3D points prediction head with intermediate layers
        self.n_net = nn.Sequential(
            Linear(num_feats, num_feats),
            ReLU(),
            self.dropout,
            Linear(num_feats, num_feats // 2),
            ReLU(),
            self.dropout,
            Linear(num_feats // 2, n_d_out)
        )
        
        # Outlier prediction head with deeper architecture
        self.outlier_net = nn.Sequential(
            Linear(num_feats, num_feats),
            ReLU(),
            self.dropout,
            Linear(num_feats, num_feats // 2),
            ReLU(),
            self.dropout,
            Linear(num_feats // 2, 1)
        )
        
        # Set training mode based on phase
        if phase is Phases.FINE_TUNE:
            # Default 1 = RESfM-style fine-tune (cameras/points only). U-RESfM TTT
            # (SPEC_cvpr_experiments R1) sets train.fine_tune_output_mode = 3 so the
            # outlier head stays active and the full unsupervised loss can adapt it.
            self.mode = conf.get_int('train.fine_tune_output_mode', default=1)
        else:
            self.mode = conf.get_int('train.output_mode', default=3)  # Default: train all heads
        
        # Freeze appropriate components based on training mode
        if self.mode == 2:  # Mode 2: Only train outlier detection
            # Freeze camera and points prediction heads
            for param in self.m_net.parameters():
                param.requires_grad = False
            self.m_net.eval()
            for param in self.n_net.parameters():
                param.requires_grad = False
            self.n_net.eval()
        
        if self.mode == 1:  # Mode 1: Only train camera and points
            # Freeze outlier detection head
            for param in self.outlier_net.parameters():
                param.requires_grad = False
            self.outlier_net.eval()
        
        # Print initialization summary
        print(f"\n{'='*70}")
        print(f"DeepSetOfSetOutliersNet (FIXED) Initialization")
        print(f"{'='*70}")
        print(f"Architecture:")
        print(f"  - Blocks: {num_blocks}, Layers per block: {block_size}")
        print(f"  - Total depth: {total_layers} layers, Features: {num_feats}")
        print(f"Configuration:")
        print(f"  - Layer Norm: {use_layer_norm}")
        print(f"  - Residual: {use_residual} (applied from block 3 onwards)")
        print(f"  - Dropout: {dropout_rate}")
        print(f"  - Progressive Training: {use_progressive}")
        print(f"Training:")
        print(f"  - Mode: {self.mode} (1=ESFM, 2=outliers, 3=both)")
        print(f"{'='*70}\n")
    
    def set_active_depth(self, depth):
        """Set the active depth for progressive training
        
        Args:
            depth: Number of blocks to activate (1 to max_depth)
        """
        if self.use_progressive:
            self.current_depth = min(depth, self.max_depth)
            print(f"Active depth set to {self.current_depth} blocks")
    
    def forward(self, data):
        """Forward pass through the network
        
        Args:
            data: Input data containing sparse matrix x of shape [m,n,d]
                  m: number of cameras, n: number of points, d: input dimension
        
        Returns:
            pred_cam: Camera predictions (if mode != 2)
            outliers_out: Outlier predictions (if mode != 1)
        """
        x = data.x  # Extract sparse matrix [m,n,d]
        
        # Embed input with optional positional encoding
        x = self.embed(x)
        
        # Process through equivariant blocks with optional progressive depth
        active_blocks = self.current_depth if self.use_progressive else len(self.equivariant_blocks)
        
        for i in range(active_blocks):
            x = self.equivariant_blocks[i](x)
        
        # Outlier predictions (skip if mode == 1)
        if self.mode != 1:
            # Apply outlier network to all feature vectors
            outliers_out = self.outlier_net(x.values)
            # Apply sigmoid for probability output
            outliers_out = torch.sigmoid(outliers_out)
        else:
            outliers_out = None
        
        # Camera and points predictions (skip if mode == 2)
        if self.mode != 2:
            # Camera predictions: aggregate across points dimension
            m_input = x.mean(dim=1)  # [m, num_feats]
            m_out = self.m_net(m_input)  # [m, m_d_out]
            
            # Points predictions: aggregate across camera dimension
            n_input = x.mean(dim=0)  # [n, num_feats]
            n_out = self.n_net(n_input).T  # [n_d_out, n]
            
            # Extract final camera parameters from network outputs
            pred_cam = self.extract_model_outputs(m_out, n_out, data)
        else:
            pred_cam = None
        
        return pred_cam, outliers_out

 

class SetOfSetOutliersNet(BaseNet):
    def __init__(self, conf, phase=None):
        super(SetOfSetOutliersNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        multires = conf.get_int('model.multires')

        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2 + conf.get_int('model.obs_feat_dim', 0)

        self.embed = EmbeddingLayer(multires, d_in)

        self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

        self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
        self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)
        self.outlier_net = get_linear_layers([num_feats] * 2 + [1], final_layer=True, batchnorm=False)
        if phase is Phases.FINE_TUNE:
            self.mode = conf.get_int('train.fine_tune_output_mode', default=1)
        else:
            self.mode = conf.get_int('train.output_mode', default=3)

        if self.mode == 2:
            for param in self.m_net.parameters():
                param.requires_grad = False
            self.m_net.eval()
            for param in self.n_net.parameters():
                param.requires_grad = False
            self.n_net.eval()

        if self.mode == 1:
            for param in self.outlier_net.parameters():
                param.requires_grad = False
            self.outlier_net.eval()


    def forward(self, data: SceneData):

        x: SparseMat = data.x  # x is [m,n,d] sparse matrix
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]

        if self.mode != 1:
            # outliers predictions

            outliers_out = self.outlier_net(x.values)
            outliers_out = torch.sigmoid(outliers_out)
        else:
            outliers_out = None

        if self.mode != 2:

            # Cameras predictions
            m_input = x.mean(dim=1) # [m,d_out]
            m_out = self.m_net(m_input)  # [m, d_m]

            # Points predictions
            n_input = x.mean(dim=0) # [n,d_out]
            n_out = self.n_net(n_input).T  # [n, d_n] -> [d_n, n]

            # predict extrinsic matrix
            pred_cam = self.extract_model_outputs(m_out, n_out, data)

        else:
            pred_cam = None



        return pred_cam, outliers_out


class SetOfSetNet(BaseNet):
    def __init__(self, conf):
        super(SetOfSetNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        multires = conf.get_int('model.multires')

        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2 + conf.get_int('model.obs_feat_dim', 0)

        self.embed = EmbeddingLayer(multires, d_in)

        self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

        self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
        self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)

    def forward(self, data: SceneData):
        x = data.x  # x is [m,n,d] sparse matrix
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]

        # Cameras predictions
        m_input = x.mean(dim=1) # [m,d_out]
        m_out = self.m_net(m_input)  # [m, d_m]

        # Points predictions
        n_input = x.mean(dim=0) # [n,d_out]
        n_out = self.n_net(n_input).T  # [n, d_n] -> [d_n, n]

        # predict extrinsic matrix
        pred_cam = self.extract_model_outputs(m_out, n_out, data)

        return pred_cam, None

        
class DeepSetOfSetNet(BaseNet):
    """Standalone deep SetOfSet network without outlier detection"""
    def __init__(self, conf):
        super(DeepSetOfSetNet, self).__init__(conf)
        
        # Configuration parameters
        num_blocks = conf.get_int('model.num_blocks', default=3)
        block_size = conf.get_int('model.block_size', default=2)
        num_feats = conf.get_int('model.num_features', default=512)
        multires = conf.get_int('model.multires', default=0)
        use_progressive = conf.get_bool('model.use_progressive', default=False)
        use_layer_norm = conf.get_bool('model.use_layer_norm', default=True)
        use_residual = conf.get_bool('model.use_residual', default=True)
        dropout_rate = conf.get_float('model.dropout_rate', default=0.1)
        
        # Store config for progressive training
        self.use_progressive = use_progressive
        self.current_depth = 1 if use_progressive else num_blocks
        self.max_depth = num_blocks
        
        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2 + conf.get_int('model.obs_feat_dim', 0)
        
        # Input embedding
        self.embed = EmbeddingLayer(multires, d_in)
        
        # Create equivariant blocks
        self.equivariant_blocks = torch.nn.ModuleList()
        
        # First block
        self.equivariant_blocks.append(
            DeepSetOfSetBlock(self.embed.d_out, num_feats, conf, 
                         use_residual=True,
                         use_layer_norm=use_layer_norm)
        )
        
        # Middle blocks with residual connections
        for i in range(1, num_blocks):
            use_residual_block = use_residual  # Enable for all blocks including Block 1
            
            self.equivariant_blocks.append(
                DeepSetOfSetBlock(num_feats, num_feats, conf,
                            use_residual=use_residual_block,
                            use_layer_norm=use_layer_norm)
            )
        
        # Output heads with dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Camera head
        self.m_net = nn.Sequential(
            Linear(num_feats, num_feats),
            ReLU(),
            self.dropout,
            Linear(num_feats, num_feats // 2),
            ReLU(),
            self.dropout,
            Linear(num_feats // 2, m_d_out)
        )
        
        # Points head
        self.n_net = nn.Sequential(
            Linear(num_feats, num_feats),
            ReLU(),
            self.dropout,
            Linear(num_feats, num_feats // 2),
            ReLU(),
            self.dropout,
            Linear(num_feats // 2, n_d_out)
        )
        
        print(f"DeepSetOfSetNet initialized with {num_blocks} blocks, {block_size} layers per block")
        print(f"Total depth: {num_blocks * block_size} layers")
    
    def forward(self, data):
        x = data.x
        x = self.embed(x)
        
        active_blocks = self.current_depth if self.use_progressive else len(self.equivariant_blocks)
        for i in range(active_blocks):
            x = self.equivariant_blocks[i](x)
        
        m_input = x.mean(dim=1)
        m_out = self.m_net(m_input)
        
        n_input = x.mean(dim=0)
        n_out = self.n_net(n_input).T
        
        pred_cam = self.extract_model_outputs(m_out, n_out, data)
        return pred_cam, None

# import torch
# from torch import nn
# import utils.dataset_utils
# from models.baseNet import BaseNet
# from models.layers import *
# from models.layers import SparseLayerNorm, SparseDropout
# from utils.sparse_utils import SparseMat
# from datasets.SceneData import SceneData
# from utils import general_utils
# from utils.Phases import Phases

# # class SetOfSetBlock(nn.Module):
# #     def __init__(self, d_in, d_out, conf):
# #         super(SetOfSetBlock, self).__init__()
# #         self.block_size = conf.get_int("model.block_size")
# #         self.use_skip = conf.get_bool("model.use_skip")
# #         type_name = conf.get_string("model.layer_type", default='SetOfSetLayer')
# #         self.layer_type = general_utils.get_class("models.layers." + type_name)
# #         self.layer_kwargs = dict(conf['model'].get('layer_extra_params', {}))

# #         modules = []
# #         modules.extend([self.layer_type(d_in, d_out, **self.layer_kwargs), NormalizationLayer()])
# #         for i in range(1, self.block_size):
# #             modules.extend([ActivationLayer(), self.layer_type(d_out, d_out, **self.layer_kwargs), NormalizationLayer()])
# #         self.layers = nn.Sequential(*modules)

# #         self.final_act = ActivationLayer()

# #         if self.use_skip:
# #             if d_in == d_out:
# #                 self.skip = IdentityLayer()
# #             else:
# #                 self.skip = nn.Sequential(ProjLayer(d_in, d_out), NormalizationLayer())

# #     def forward(self, x):
# #         # x is [m,n,d] sparse matrix
# #         xl = self.layers(x)
# #         if self.use_skip:
# #             xl = self.skip(x) + xl

# #         out = self.final_act(xl)
# #         return out

# class SetOfSetBlock(nn.Module):
#     """Enhanced block with residual connections, layer norm, and dropout"""
#     def __init__(self, d_in, d_out, conf, use_residual=True, use_layer_norm=True):
#         super(SetOfSetBlock, self).__init__()
#         self.block_size = conf.get_int("model.block_size", default=2)
#         self.use_skip = conf.get_bool("model.use_skip", default=True)
#         self.use_residual = use_residual
#         self.use_layer_norm = use_layer_norm
#         dropout_rate = conf.get_float("model.dropout_rate", default=0.1)
        
#         # For backward compatibility with original SetOfSetOutliersNet
#         type_name = conf.get_string("model.layer_type", default='SetOfSetLayer')
#         self.layer_type = general_utils.get_class("models.layers." + type_name)
#         self.layer_kwargs = dict(conf.get('model.layer_extra_params', {})) if 'model.layer_extra_params' in conf else {}
        
#         modules = []
        
#         # First layer in block
#         modules.append(self.layer_type(d_in, d_out, **self.layer_kwargs))
#         if self.use_layer_norm:
#             modules.append(SparseLayerNorm(d_out))
#         else:
#             modules.append(NormalizationLayer())
        
#         # Additional layers in block
#         for i in range(1, self.block_size):
#             modules.append(ActivationLayer())
            
#             # Add dropout for regularization in deep networks
#             if dropout_rate > 0 and self.training:
#                 modules.append(SparseDropout(dropout_rate))
            
#             modules.append(self.layer_type(d_out, d_out, **self.layer_kwargs))
            
#             if self.use_layer_norm:
#                 modules.append(SparseLayerNorm(d_out))
#             else:
#                 modules.append(NormalizationLayer())
        
#         self.layers = nn.Sequential(*modules)
#         self.final_act = ActivationLayer()
        
#         # Skip connection for the entire block
#         if self.use_skip:
#             if d_in == d_out:
#                 self.skip = IdentityLayer()
#             else:
#                 self.skip = nn.Sequential(ProjLayer(d_in, d_out))
#                 if self.use_layer_norm:
#                     self.skip.add_module('norm', SparseLayerNorm(d_out))
#                 else:
#                     self.skip.add_module('norm', NormalizationLayer())

#     def forward(self, x):
#         # Process through layers
#         xl = self.layers(x)
        
#         # Add skip connection if enabled
#         if self.use_skip:
#             skip_out = self.skip(x)
#             xl_values = xl.values + skip_out.values * 0.707  # Scale for stability
#             xl = SparseMat(xl_values, xl.indices, xl.cam_per_pts, xl.pts_per_cam, xl.shape)
        
#         out = self.final_act(xl)
#         return out


# class SetOfSetOutliersNet(BaseNet):
#     def __init__(self, conf, phase=None):
#         super(SetOfSetOutliersNet, self).__init__(conf)
#         # n is the number of points and m is the number of cameras
#         num_blocks = conf.get_int('model.num_blocks')
#         num_feats = conf.get_int('model.num_features')
#         multires = conf.get_int('model.multires')

#         n_d_out = 3
#         m_d_out = self.out_channels
#         d_in = 2

#         self.embed = EmbeddingLayer(multires, d_in)

#         self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
#         for i in range(num_blocks - 1):
#             self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

#         self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
#         self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)
#         self.outlier_net = get_linear_layers([num_feats] * 2 + [1], final_layer=True, batchnorm=False)
#         if phase is Phases.FINE_TUNE:
#             self.mode = 1
#         else:
#             self.mode = conf.get_int('train.output_mode', default=3)

#         if self.mode == 2:
#             for param in self.m_net.parameters():
#                 param.requires_grad = False
#             self.m_net.eval()
#             for param in self.n_net.parameters():
#                 param.requires_grad = False
#             self.n_net.eval()

#         if self.mode == 1:
#             for param in self.outlier_net.parameters():
#                 param.requires_grad = False
#             self.outlier_net.eval()


#     def forward(self, data: SceneData):

#         x: SparseMat = data.x  # x is [m,n,d] sparse matrix
#         x = self.embed(x)
#         for eq_block in self.equivariant_blocks:
#             x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]

#         if self.mode != 1:
#             # outliers predictions

#             outliers_out = self.outlier_net(x.values)
#             outliers_out = torch.sigmoid(outliers_out)
#         else:
#             outliers_out = None

#         if self.mode != 2:

#             # Cameras predictions
#             m_input = x.mean(dim=1) # [m,d_out]
#             m_out = self.m_net(m_input)  # [m, d_m]

#             # Points predictions
#             n_input = x.mean(dim=0) # [n,d_out]
#             n_out = self.n_net(n_input).T  # [n, d_n] -> [d_n, n]

#             # predict extrinsic matrix
#             pred_cam = self.extract_model_outputs(m_out, n_out, data)

#         else:
#             pred_cam = None

#         return pred_cam, outliers_out

# # class DeepSetOfSetOutliersNet(BaseNet):
# #     """Deep SetOfSet network with outlier detection capabilities"""
# #     def __init__(self, conf, phase=None):
# #         super(DeepSetOfSetOutliersNet, self).__init__(conf)
        
# #         # Deep network default configuration parameters
# #         num_blocks = conf.get_int('model.num_blocks', default=2)  # Default to 2 blocks for deep network
# #         block_size = conf.get_int('model.block_size', default=3)  # Default to 3 layers per block
# #         num_feats = conf.get_int('model.num_features', default=256)  # experiment also with Larger feature dimension for deep network 512 
# #         multires = conf.get_int('model.multires', default=0)
        
# #         # Deep architecture features enabled by default
# #         use_progressive = conf.get_bool('model.use_progressive', default=False)
# #         use_layer_norm = conf.get_bool('model.use_layer_norm', default=True)  # Enabled by default for deep network
# #         use_residual = conf.get_bool('model.use_residual', default=True)  # Enabled by default for deep network
# #         dropout_rate = conf.get_float('model.dropout_rate', default=0.1)  # Default dropout for regularization
        
# #         # Store config for progressive training
# #         self.use_progressive = use_progressive
# #         self.current_depth = 1 if use_progressive else num_blocks
# #         self.max_depth = num_blocks
        
# #         total_layers = num_blocks * block_size
        
# #         # Output dimensions
# #         n_d_out = 3  # 3D points output dimension
# #         m_d_out = self.out_channels  # Camera parameters output dimension
# #         d_in = 2  # Input dimension (2D observations)
        
# #         # Input embedding layer with optional positional encoding
# #         self.embed = EmbeddingLayer(multires, d_in)
        
# #         # Create equivariant blocks for permutation-invariant processing
# #         self.equivariant_blocks = torch.nn.ModuleList()
        
# #         # First block without residual connection
# #         self.equivariant_blocks.append(
# #             SetOfSetBlock(self.embed.d_out, num_feats, conf)
# #         )
        
# #         # Additional blocks with residual connections for deeper layers
# #         for i in range(1, num_blocks):
# #             # Enable residual connections starting from the third block for stability
# #             use_residual_block = use_residual and (i >= 2)
            
# #             self.equivariant_blocks.append(
# #                 SetOfSetBlock(num_feats, num_feats, conf)
# #             )
        
# #         # Dropout layer for regularization
# #         self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
# #         # Camera parameters prediction head with intermediate layers
# #         self.m_net = nn.Sequential(
# #             Linear(num_feats, num_feats),
# #             ReLU(),
# #             self.dropout,
# #             Linear(num_feats, num_feats // 2),
# #             ReLU(),
# #             self.dropout,
# #             Linear(num_feats // 2, m_d_out)
# #         )
        
# #         # 3D points prediction head with intermediate layers
# #         self.n_net = nn.Sequential(
# #             Linear(num_feats, num_feats),
# #             ReLU(),
# #             self.dropout,
# #             Linear(num_feats, num_feats // 2),
# #             ReLU(),
# #             self.dropout,
# #             Linear(num_feats // 2, n_d_out)
# #         )
        
# #         # Outlier prediction head with deeper architecture
# #         self.outlier_net = nn.Sequential(
# #             Linear(num_feats, num_feats),
# #             ReLU(),
# #             self.dropout,
# #             Linear(num_feats, num_feats // 2),
# #             ReLU(),
# #             self.dropout,
# #             Linear(num_feats // 2, 1)
# #         )
        
# #         # Set training mode based on phase
# #         if phase is Phases.FINE_TUNE:
# #             self.mode = 1  # Fine-tune mode: only train camera and points heads
# #         else:
# #             self.mode = conf.get_int('train.output_mode', default=3)  # Default: train all heads
        
# #         # Freeze appropriate components based on training mode
# #         if self.mode == 2:  # Mode 2: Only train outlier detection
# #             # Freeze camera and points prediction heads
# #             for param in self.m_net.parameters():
# #                 param.requires_grad = False
# #             self.m_net.eval()
# #             for param in self.n_net.parameters():
# #                 param.requires_grad = False
# #             self.n_net.eval()
        
# #         if self.mode == 1:  # Mode 1: Only train camera and points
# #             # Freeze outlier detection head
# #             for param in self.outlier_net.parameters():
# #                 param.requires_grad = False
# #             self.outlier_net.eval()
        
# #         # Print initialization summary
# #         print(f"DeepSetOfSetOutliersNet initialized with {num_blocks} blocks, {block_size} layers per block")
# #         print(f"Total depth: {total_layers} layers, Features: {num_feats}")
# #         print(f"Layer Norm: {use_layer_norm}, Residual: {use_residual}, Dropout: {dropout_rate}")
# #         print(f"Progressive Training: {use_progressive}, Mode: {self.mode}")
    
# #     def set_active_depth(self, depth):
# #         """Set the active depth for progressive training
        
# #         Args:
# #             depth: Number of blocks to activate (1 to max_depth)
# #         """
# #         if self.use_progressive:
# #             self.current_depth = min(depth, self.max_depth)
# #             print(f"Active depth set to {self.current_depth} blocks")
    
# #     def forward(self, data):
# #         """Forward pass through the network
        
# #         Args:
# #             data: Input data containing sparse matrix x of shape [m,n,d]
# #                   m: number of cameras, n: number of points, d: input dimension
        
# #         Returns:
# #             pred_cam: Camera predictions (if mode != 2)
# #             outliers_out: Outlier predictions (if mode != 1)
# #         """
# #         x = data.x  # Extract sparse matrix [m,n,d]
        
# #         # Embed input with optional positional encoding
# #         x = self.embed(x)
        
# #         # Process through equivariant blocks with optional progressive depth
# #         active_blocks = self.current_depth if self.use_progressive else len(self.equivariant_blocks)
        
# #         for i in range(active_blocks):
# #             x = self.equivariant_blocks[i](x)
        
# #         # Outlier predictions (skip if mode == 1)
# #         if self.mode != 1:
# #             # Apply outlier network to all feature vectors
# #             outliers_out = self.outlier_net(x.values)
# #             # Apply sigmoid for probability output
# #             outliers_out = torch.sigmoid(outliers_out)
# #         else:
# #             outliers_out = None
        
# #         # Camera and points predictions (skip if mode == 2)
# #         if self.mode != 2:
# #             # Camera predictions: aggregate across points dimension
# #             m_input = x.mean(dim=1)  # [m, num_feats]
# #             m_out = self.m_net(m_input)  # [m, m_d_out]
            
# #             # Points predictions: aggregate across camera dimension
# #             n_input = x.mean(dim=0)  # [n, num_feats]
# #             n_out = self.n_net(n_input).T  # [n_d_out, n]
            
# #             # Extract final camera parameters from network outputs
# #             pred_cam = self.extract_model_outputs(m_out, n_out, data)
# #         else:
# #             pred_cam = None
        
# #         return pred_cam, outliers_out

# class DeepSetOfSetOutliersNet(BaseNet):
#     """Deep SetOfSet network with outlier detection capabilities - FIXED VERSION"""
#     def __init__(self, conf, phase=None):
#         super(DeepSetOfSetOutliersNet, self).__init__(conf)
        
#         # Deep network default configuration parameters
#         num_blocks = conf.get_int('model.num_blocks', default=2)
#         block_size = conf.get_int('model.block_size', default=3)
#         num_feats = conf.get_int('model.num_features', default=256)
#         multires = conf.get_int('model.multires', default=0)
        
#         # Deep architecture features enabled by default
#         use_progressive = conf.get_bool('model.use_progressive', default=False)
#         use_layer_norm = conf.get_bool('model.use_layer_norm', default=True)
#         use_residual = conf.get_bool('model.use_residual', default=True)
#         dropout_rate = conf.get_float('model.dropout_rate', default=0.1)
        
#         # Store config for progressive training
#         self.use_progressive = use_progressive
#         self.current_depth = 1 if use_progressive else num_blocks
#         self.max_depth = num_blocks
        
#         total_layers = num_blocks * block_size
        
#         # Output dimensions
#         n_d_out = 3  # 3D points output dimension
#         m_d_out = self.out_channels  # Camera parameters output dimension
#         d_in = 2  # Input dimension (2D observations)
        
#         # Input embedding layer with optional positional encoding
#         self.embed = EmbeddingLayer(multires, d_in)
        
#         # Create equivariant blocks for permutation-invariant processing
#         self.equivariant_blocks = torch.nn.ModuleList()
        
#         # FIXED: First block - explicitly pass use_residual parameter
#         # Set to False for first block to match DeepSetOfSetNet pattern
#         self.equivariant_blocks.append(
#             SetOfSetBlock(
#                 self.embed.d_out, 
#                 num_feats, 
#                 conf,
#                 use_residual=False,          # ← NOW PASSED! First block: no residual
#                 use_layer_norm=use_layer_norm # ← NOW PASSED!
#             )
#         )
        
#         # FIXED: Additional blocks with residual connections for deeper layers
#         for i in range(1, num_blocks):
#             # Enable residual connections starting from the third block for stability
#             # This matches the pattern in DeepSetOfSetNet
#             use_residual_block = use_residual and (i >= 2)
            
#             self.equivariant_blocks.append(
#                 SetOfSetBlock(
#                     num_feats, 
#                     num_feats, 
#                     conf,
#                     use_residual=use_residual_block,  # ← NOW PASSED! Actually used
#                     use_layer_norm=use_layer_norm     # ← NOW PASSED!
#                 )
#             )
        
#         # Dropout layer for regularization (used in prediction heads)
#         self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
#         # Camera parameters prediction head with intermediate layers
#         self.m_net = nn.Sequential(
#             Linear(num_feats, num_feats),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats, num_feats // 2),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats // 2, m_d_out)
#         )
        
#         # 3D points prediction head with intermediate layers
#         self.n_net = nn.Sequential(
#             Linear(num_feats, num_feats),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats, num_feats // 2),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats // 2, n_d_out)
#         )
        
#         # Outlier prediction head with deeper architecture
#         self.outlier_net = nn.Sequential(
#             Linear(num_feats, num_feats),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats, num_feats // 2),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats // 2, 1)
#         )
        
#         # Set training mode based on phase
#         if phase is Phases.FINE_TUNE:
#             self.mode = 1  # Fine-tune mode: only train camera and points heads
#         else:
#             self.mode = conf.get_int('train.output_mode', default=3)  # Default: train all heads
        
#         # Freeze appropriate components based on training mode
#         if self.mode == 2:  # Mode 2: Only train outlier detection
#             # Freeze camera and points prediction heads
#             for param in self.m_net.parameters():
#                 param.requires_grad = False
#             self.m_net.eval()
#             for param in self.n_net.parameters():
#                 param.requires_grad = False
#             self.n_net.eval()
        
#         if self.mode == 1:  # Mode 1: Only train camera and points
#             # Freeze outlier detection head
#             for param in self.outlier_net.parameters():
#                 param.requires_grad = False
#             self.outlier_net.eval()
        
#         # Print initialization summary
#         print(f"\n{'='*70}")
#         print(f"DeepSetOfSetOutliersNet (FIXED) Initialization")
#         print(f"{'='*70}")
#         print(f"Architecture:")
#         print(f"  - Blocks: {num_blocks}, Layers per block: {block_size}")
#         print(f"  - Total depth: {total_layers} layers, Features: {num_feats}")
#         print(f"Configuration:")
#         print(f"  - Layer Norm: {use_layer_norm}")
#         print(f"  - Residual: {use_residual} (applied from block 3 onwards)")
#         print(f"  - Dropout: {dropout_rate}")
#         print(f"  - Progressive Training: {use_progressive}")
#         print(f"Training:")
#         print(f"  - Mode: {self.mode} (1=ESFM, 2=outliers, 3=both)")
#         print(f"{'='*70}\n")
    
#     def set_active_depth(self, depth):
#         """Set the active depth for progressive training
        
#         Args:
#             depth: Number of blocks to activate (1 to max_depth)
#         """
#         if self.use_progressive:
#             self.current_depth = min(depth, self.max_depth)
#             print(f"Active depth set to {self.current_depth} blocks")
    
#     def forward(self, data):
#         """Forward pass through the network
        
#         Args:
#             data: Input data containing sparse matrix x of shape [m,n,d]
#                   m: number of cameras, n: number of points, d: input dimension
        
#         Returns:
#             pred_cam: Camera predictions (if mode != 2)
#             outliers_out: Outlier predictions (if mode != 1)
#         """
#         x = data.x  # Extract sparse matrix [m,n,d]
        
#         # Embed input with optional positional encoding
#         x = self.embed(x)
        
#         # Process through equivariant blocks with optional progressive depth
#         active_blocks = self.current_depth if self.use_progressive else len(self.equivariant_blocks)
        
#         for i in range(active_blocks):
#             x = self.equivariant_blocks[i](x)
        
#         # Outlier predictions (skip if mode == 1)
#         if self.mode != 1:
#             # Apply outlier network to all feature vectors
#             outliers_out = self.outlier_net(x.values)
#             # Apply sigmoid for probability output
#             outliers_out = torch.sigmoid(outliers_out)
#         else:
#             outliers_out = None
        
#         # Camera and points predictions (skip if mode == 2)
#         if self.mode != 2:
#             # Camera predictions: aggregate across points dimension
#             m_input = x.mean(dim=1)  # [m, num_feats]
#             m_out = self.m_net(m_input)  # [m, m_d_out]
            
#             # Points predictions: aggregate across camera dimension
#             n_input = x.mean(dim=0)  # [n, num_feats]
#             n_out = self.n_net(n_input).T  # [n_d_out, n]
            
#             # Extract final camera parameters from network outputs
#             pred_cam = self.extract_model_outputs(m_out, n_out, data)
#         else:
#             pred_cam = None
        
#         return pred_cam, outliers_out

        
# class SetOfSetNet(BaseNet):
#     def __init__(self, conf):
#         super(SetOfSetNet, self).__init__(conf)
#         # n is the number of points and m is the number of cameras
#         num_blocks = conf.get_int('model.num_blocks')
#         num_feats = conf.get_int('model.num_features')
#         multires = conf.get_int('model.multires')

#         n_d_out = 3
#         m_d_out = self.out_channels
#         d_in = 2

#         self.embed = EmbeddingLayer(multires, d_in)

#         self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
#         for i in range(num_blocks - 1):
#             self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

#         self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
#         self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)

#     def forward(self, data: SceneData):
#         x = data.x  # x is [m,n,d] sparse matrix
#         x = self.embed(x)
#         for eq_block in self.equivariant_blocks:
#             x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]

#         # Cameras predictions
#         m_input = x.mean(dim=1) # [m,d_out]
#         m_out = self.m_net(m_input)  # [m, d_m]

#         # Points predictions
#         n_input = x.mean(dim=0) # [n,d_out]
#         n_out = self.n_net(n_input).T  # [n, d_n] -> [d_n, n]

#         # predict extrinsic matrix
#         pred_cam = self.extract_model_outputs(m_out, n_out, data)

#         return pred_cam, None



# class DeepSetOfSetNet(BaseNet):
#     """Standalone deep SetOfSet network without outlier detection"""
#     def __init__(self, conf):
#         super(DeepSetOfSetNet, self).__init__(conf)
        
#         # Configuration parameters
#         num_blocks = conf.get_int('model.num_blocks', default=3)
#         block_size = conf.get_int('model.block_size', default=2)
#         num_feats = conf.get_int('model.num_features', default=512)
#         multires = conf.get_int('model.multires', default=0)
#         use_progressive = conf.get_bool('model.use_progressive', default=False)
#         use_layer_norm = conf.get_bool('model.use_layer_norm', default=True)
#         use_residual = conf.get_bool('model.use_residual', default=True)
#         dropout_rate = conf.get_float('model.dropout_rate', default=0.1)
        
#         # Store config for progressive training
#         self.use_progressive = use_progressive
#         self.current_depth = 1 if use_progressive else num_blocks
#         self.max_depth = num_blocks
        
#         n_d_out = 3
#         m_d_out = self.out_channels
#         d_in = 2
        
#         # Input embedding
#         self.embed = EmbeddingLayer(multires, d_in)
        
#         # Create equivariant blocks
#         self.equivariant_blocks = torch.nn.ModuleList()
        
#         # First block
#         self.equivariant_blocks.append(
#             SetOfSetBlock(self.embed.d_out, num_feats, conf, 
#                          use_residual=False,
#                          use_layer_norm=use_layer_norm)
#         )
        
#         # Middle blocks with residual connections
#         for i in range(1, num_blocks):
#             use_residual_block = use_residual and (i >= 2)
            
#             self.equivariant_blocks.append(
#                 SetOfSetBlock(num_feats, num_feats, conf,
#                             use_residual=use_residual_block,
#                             use_layer_norm=use_layer_norm)
#             )
        
#         # Output heads with dropout
#         self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
#         # Camera head
#         self.m_net = nn.Sequential(
#             Linear(num_feats, num_feats),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats, num_feats // 2),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats // 2, m_d_out)
#         )
        
#         # Points head
#         self.n_net = nn.Sequential(
#             Linear(num_feats, num_feats),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats, num_feats // 2),
#             ReLU(),
#             self.dropout,
#             Linear(num_feats // 2, n_d_out)
#         )
        
#         print(f"DeepSetOfSetNet initialized with {num_blocks} blocks, {block_size} layers per block")
#         print(f"Total depth: {num_blocks * block_size} layers")
    
#     def forward(self, data):
#         x = data.x
#         x = self.embed(x)
        
#         active_blocks = self.current_depth if self.use_progressive else len(self.equivariant_blocks)
#         for i in range(active_blocks):
#             x = self.equivariant_blocks[i](x)
        
#         m_input = x.mean(dim=1)
#         m_out = self.m_net(m_input)
        
#         n_input = x.mean(dim=0)
#         n_out = self.n_net(n_input).T
        
#         pred_cam = self.extract_model_outputs(m_out, n_out, data)
#         return pred_cam, None
