import torch


class SparseMat:
    """
    Sparse matrix representation for efficient memory usage.
    Fixed for mixed precision training (FP16/FP32 compatibility).
    """
    def __init__(self, values, indices, cam_per_pts, pts_per_cam, shape):
        assert len(shape) == 3
        self.values = values  # [all_points_anywhere, 2], 2 for (x, y) within any image
        self.indices = indices  # [2, all_points_anywhere], 2 for (camera_ind, track_ind)
        self.shape = shape  # shape of a sparse matrix, (num_cameras, num_tracks)
        self.cam_per_pts = cam_per_pts  # [n_pts, 1]
        self.pts_per_cam = pts_per_cam  # [n_cams, 1]
        self.device = self.values.device

    @property
    def size(self):
        return self.shape

    def sum(self, dim):
        """
        Equivalent to M.sum(dim), where M is sparse and points that don't exist are (0, 0).
        
        Fixed for mixed precision training:
        - Ensures mat_sum has same dtype as self.values
        - Compatible with both FP16 and FP32
        """
        assert dim == 1 or dim == 0
        n_features = self.values.shape[-1]
        out_size = self.shape[0] if dim == 1 else self.shape[1]
        indices_index = 0 if dim == 1 else 1
        
        # ⭐ FIX: Create zeros tensor with same dtype as values
        # This ensures compatibility with mixed precision training
        mat_sum = torch.zeros(out_size, n_features, 
                              device=self.device,
                              dtype=self.values.dtype)  # ← Key fix!
        
        return mat_sum.index_add(0, self.indices[indices_index], self.values)

    def mean(self, dim):
        """
        Calculate mean along dimension.
        
        Fixed for mixed precision training:
        - Ensures cam_per_pts and pts_per_cam match dtype of sum result
        - Handles division with correct precision
        """
        assert dim == 1 or dim == 0
        
        if dim == 0:
            # Sum along cameras (dim=0)
            sum_result = self.sum(dim=0)
            
            # ⭐ FIX: Convert cam_per_pts to match sum_result dtype
            cam_per_pts_typed = self.cam_per_pts.to(sum_result.dtype)
            
            # Calculate mean
            mean = sum_result / cam_per_pts_typed
            
            # Set mean to 0 for points with no cameras
            # (torch.where instead of masked assignment: torch 2.0.1's deterministic
            # index_put asserts when broadcasting a scalar into a multi-row bool mask)
            mean = torch.where((self.cam_per_pts == 0).reshape(-1, 1), torch.zeros_like(mean), mean)
            return mean
        else:
            # Sum along points (dim=1)
            sum_result = self.sum(dim=1)
            
            # ⭐ FIX: Convert pts_per_cam to match sum_result dtype
            pts_per_cam_typed = self.pts_per_cam.to(sum_result.dtype)
            
            # Calculate mean
            mean = sum_result / pts_per_cam_typed
            
            # Set mean to 0 for cameras with no points
            # (see note above about torch 2.0.1 deterministic index_put)
            mean = torch.where((self.pts_per_cam == 0).reshape(-1, 1), torch.zeros_like(mean), mean)
            return mean

    def to(self, device, **kwargs):
        """
        Move sparse matrix to device with optional dtype conversion.
        
        Args:
            device: Target device (cuda/cpu)
            **kwargs: Additional arguments (e.g., dtype)
        """
        self.device = device
        self.values = self.values.to(device, **kwargs)
        self.indices = self.indices.to(device, **kwargs)
        self.pts_per_cam = self.pts_per_cam.to(device, **kwargs)
        self.cam_per_pts = self.cam_per_pts.to(device, **kwargs)
        return self

    def __add__(self, other):
        """
        Add two sparse matrices element-wise.
        
        Args:
            other: Another SparseMat with same shape
            
        Returns:
            New SparseMat with summed values
        """
        assert self.shape == other.shape
        # assert (self.indices == other.indices).all()  # removed due to runtime
        new_values = self.values + other.values
        return SparseMat(new_values, self.indices, self.cam_per_pts, 
                         self.pts_per_cam, self.shape)

                         

# import torch


# class SparseMat:
#     def __init__(self, values, indices, cam_per_pts, pts_per_cam, shape):
#         assert len(shape) == 3
#         self.values = values  # [all_points_anywhere, 2], 2 for (x, y) within any image
#         self.indices = indices  # [2, all_points_anywhere], 2 for (camera_ind, track_ind)
#         self.shape = shape  # shape of a sparse matrix, (num_cameras, num_tracks)
#         self.cam_per_pts = cam_per_pts  # [n_pts, 1]
#         self.pts_per_cam = pts_per_cam  # [n_cams, 1]
#         self.device = self.values.device

#     @property
#     def size(self):
#         return self.shape

#     def sum(self, dim):
#         # equivalent to M.sum(dim), where M is sparse and points that don't exist are (0, 0)
#         assert dim == 1 or dim == 0
#         n_features = self.values.shape[-1]
#         out_size = self.shape[0] if dim == 1 else self.shape[1]
#         indices_index = 0 if dim == 1 else 1
#         mat_sum = torch.zeros(out_size, n_features, device=self.device)
#         return mat_sum.index_add(0, self.indices[indices_index], self.values)


#     def mean(self, dim):
#         assert dim == 1 or dim == 0
#         if dim == 0:
#             mean = self.sum(dim=0) / self.cam_per_pts
#             mean[(self.cam_per_pts == 0).squeeze(), :] = 0
#             return mean
#         else:
#             mean = self.sum(dim=1) / self.pts_per_cam
#             mean[(self.pts_per_cam == 0).squeeze(), :] = 0
#             return mean

#     def to(self, device, **kwargs):
#         self.device = device
#         self.values = self.values.to(device, **kwargs)
#         self.indices = self.indices.to(device, **kwargs)
#         self.pts_per_cam = self.pts_per_cam.to(device, **kwargs)
#         self.cam_per_pts = self.cam_per_pts.to(device, **kwargs)
#         return self

#     def __add__(self, other):
#         assert self.shape == other.shape
#         # assert (self.indices == other.indices).all()  # removed due to runtime
#         new_values = self.values + other.values
#         return SparseMat(new_values, self.indices, self.cam_per_pts, self.pts_per_cam, self.shape)

