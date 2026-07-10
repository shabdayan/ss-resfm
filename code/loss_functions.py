import torch
from utils import geo_utils
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import os


class ESFMLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.infinity_pts_margin = conf.get_float("loss.infinity_pts_margin")
        self.normalize_grad = conf.get_bool("loss.normalize_grad")

        self.hinge_loss = conf.get_bool("loss.hinge_loss")
        if self.hinge_loss:
            self.hinge_loss_weight = conf.get_float("loss.hinge_loss_weight")
        else:
            self.hinge_loss_weight = 0

    def forward(self, pred_cam, data, epoch=None):
        Ps = pred_cam["Ps_norm"]  # [m, 3, 4]
        pts_2d = Ps @ pred_cam["pts3D"]  # [m, 3, 4] @ [4, n] -> [m, 3, n]

        # Normalize gradient
        if self.normalize_grad:
            pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / data.valid_pts.sum())

        # Get point for reprojection loss
        if self.hinge_loss:
            # mark as False points with very small w in homogeneous coordinates, that is, very large u, v in inhomogeneous coordinates
            projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)  # [m, n], boolean mask
        else:
            projected_points = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)

        # Calculate hinge Loss
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight

        # Calculate reprojection error
        # From homogeneous coordinates to in inhomogeneous coordinates for all projected_points
        pts_2d = (pts_2d / torch.where(projected_points, pts_2d[:, 2, :], torch.ones_like(projected_points).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)
        
        # use reprojection error for projected_points, hinge_loss everywhere else
        return torch.where(projected_points, reproj_err, hinge_loss)[data.valid_pts].mean()


class ESFMLoss_weighted_by_rep_err(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.infinity_pts_margin = conf.get_float("loss.infinity_pts_margin")
        self.normalize_grad = conf.get_bool("loss.normalize_grad")

        self.hinge_loss = conf.get_bool("loss.hinge_loss")
        if self.hinge_loss:
            self.hinge_loss_weight = conf.get_float("loss.hinge_loss_weight")
        else:
            self.hinge_loss_weight = 0

    def forward(self, pred_cam, data, epoch=None):
        pred_outliers = data.proj_err_weight if hasattr(data, 'proj_err_weight') else None
    
        # Add safety check
        if pred_outliers is None:
            raise ValueError(
                "ESFMLoss_weighted requires reprojection weights! "
                "Make sure data.reprojection_weights is set for stage 2.")
        
        Ps = pred_cam["Ps_norm"]  # [m, 3, 4]
        pts_2d = Ps @ pred_cam["pts3D"]  # [m, 3, 4] @ [4, n] -> [m, 3, n]

        # Normalize gradient
        if self.normalize_grad:
            pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / data.valid_pts.sum())

        # Get point for reprojection loss
        if self.hinge_loss:
            # mark as False points with very small w in homogeneous coordinates, that is, very large u, v in inhomogeneous coordinates
            projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)  # [m, n], boolean mask #
        else:
            projected_points = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)

        # Calculate hinge Loss
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight

        # Calculate reprojection error
        # From homogeneous coordinates to in inhomogeneous coordinates for all projected_points
        pts_2d = (pts_2d / torch.where(projected_points, pts_2d[:, 2, :], torch.ones_like(projected_points).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)


        
        projected_points = projected_points[data.valid_pts]
        reproj_err = reproj_err[data.valid_pts]
        hinge_loss = hinge_loss[data.valid_pts]

        pred_outliers = pred_outliers[data.valid_pts]
        reproj_err = (1 - pred_outliers) * reproj_err # Outlier-weighted loss
        weightedLoss = torch.where(projected_points, reproj_err, hinge_loss)

        return weightedLoss.mean()



class ESFMLoss_weighted(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.infinity_pts_margin = conf.get_float("loss.infinity_pts_margin")
        self.normalize_grad = conf.get_bool("loss.normalize_grad")

        self.hinge_loss = conf.get_bool("loss.hinge_loss")
        if self.hinge_loss:
            self.hinge_loss_weight = conf.get_float("loss.hinge_loss_weight")
        else:
            self.hinge_loss_weight = 0

    def forward(self, pred_cam, pred_outliers, data, epoch=None):
        Ps = pred_cam["Ps_norm"]  # [m, 3, 4]
        pts_2d = Ps @ pred_cam["pts3D"]  # [m, 3, 4] @ [4, n] -> [m, 3, n]

        # Normalize gradient
        if self.normalize_grad:
            pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / data.valid_pts.sum())

        # Get point for reprojection loss
        if self.hinge_loss:
            # mark as False points with very small w in homogeneous coordinates, that is, very large u, v in inhomogeneous coordinates
            projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)  # [m, n], boolean mask #
        else:
            projected_points = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)

        # Calculate hinge Loss
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight

        # Calculate reprojection error
        # From homogeneous coordinates to in inhomogeneous coordinates for all projected_points
        pts_2d = (pts_2d / torch.where(projected_points, pts_2d[:, 2, :], torch.ones_like(projected_points).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)


        # use reprojection error for projected_points, hinge_loss everywhere else
        projected_points = projected_points[data.valid_pts]
        reproj_err = reproj_err[data.valid_pts]
        hinge_loss = hinge_loss[data.valid_pts]

        pred_outliers = pred_outliers.squeeze(dim=-1)
        reproj_err = (1 - pred_outliers) * reproj_err # Outlier-weighted loss
        weightedLoss = torch.where(projected_points, reproj_err, hinge_loss)

        return weightedLoss.mean()



class GT_Loss_Outliers(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, pred_outliers, data, epoch=None):
        gt_outliers = data.outlier_indices[data.x.indices.T[:, 0], data.x.indices.T[:, 1]]

        # Compute class-balanced BCE loss
        outliers_ratio = gt_outliers.sum() / gt_outliers.shape[0]
        weights = (gt_outliers.float() * ((1.0 / outliers_ratio) * 1 - 2)) + 1 #class balancing
        bce_loss = F.binary_cross_entropy(pred_outliers.squeeze(), gt_outliers.float(), weight=weights)

        return bce_loss

class OutliersLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.outliers_loss = GT_Loss_Outliers(conf)

    def forward(self, pred_outliers, data):
        loss = self.outliers_loss(pred_outliers, data)
        return loss



class CombinedLoss(nn.Module):
    """
    Combined loss: α * ESFMLoss + β * ClassificationLoss
    With CSV logging of loss magnitudes for hyperparameter analysis
    """
    def __init__(self, conf):
        super().__init__()
        
        # Loss components. Per the report's definition (sec. 2.2.2):
        #   Adaptive Confidence Weighted Outlier Loss = alpha * ESFM Loss + beta * Classification Loss
        # The geometric term is the PLAIN (unweighted) ESFM reprojection loss; the
        # outlier scores never weight it during training. The detector trains only
        # via the confident-pseudo-label BCE, and its scores are used at test time.
        self.outliers_loss = AdaptiveConfidenceWeightedOutliersLoss(conf)
        self.esfm_loss = ESFMLoss(conf)
        
        # Loss weights (your existing code)
        self.alpha = conf.get_float('loss.reproj_loss_weight', default=1.0)
        self.beta = conf.get_float('loss.classification_loss_weight', default=0.3)

        # CSV logging grows a DataFrame every forward pass; multi-scene training
        # (many scenes x many epochs) must be able to turn it off.
        self.enable_csv_logging = conf.get_bool('loss.enable_csv_logging', default=True)

        # ============================================
        # NEW: CSV LOGGING SETUP
        # ============================================
        self.loss_tracking_df = pd.DataFrame(columns=[
            'epoch',
            'iteration',
            'esfm_loss_raw',
            'classification_loss_raw',
            'loss_magnitude_ratio',
            'weighted_esfm_loss',
            'weighted_classification_loss',
            'total_loss',
            'esfm_contribution_percent',
            'classification_contribution_percent',
            'alpha',
            'beta'
        ])
        
        # Get results path from config for saving CSV
        self.results_path = conf.get_string('results_path', default='./results')
        self.csv_save_path = Path(self.results_path) / 'loss_magnitudes_tracking.csv'
        
        # Create directory if it doesn't exist
        os.makedirs(Path(self.results_path), exist_ok=True)
        
        # Tracking variables
        self.iteration_count = 0
        self.save_frequency = 10  # Save CSV every N epochs
        self.last_saved_epoch = -1
        
        print(f"\n[CombinedLoss] Initialized with CSV logging")
        print(f"  CSV will be saved to: {self.csv_save_path}")
        print(f"  Alpha (reproj weight): {self.alpha}")
        print(f"  Beta (classification weight): {self.beta}")
        print(f"  Save frequency: every {self.save_frequency} epochs\n")

    def forward(self, pred_cam, pred_outliers, data, epoch=None):
        """
        Compute combined loss with CSV logging
        
        Line-by-line explanation:
        - Initialize loss tensors on correct device
        - Compute ESFMLoss if alpha > 0 using weighted_ESFM_loss
        - Compute classification loss if beta > 0 using outliers_loss
        - Combine weighted losses: alpha * ESFM + beta * classification
        - Log raw and weighted values to DataFrame for analysis
        - Periodically save DataFrame to CSV file
        - Return combined total loss for backpropagation
        """
        
        # Initialize losses (your existing code)
        classificationLoss = torch.tensor([0.0], device=pred_outliers.device, dtype=torch.float32)
        ESFMLoss = torch.tensor([0.0], device=pred_outliers.device, dtype=torch.float32)

        # Compute Reprojection loss (geometric loss) - plain ESFM loss per the formula
        if self.alpha:
            ESFMLoss = self.esfm_loss(pred_cam, data)

        # Compute Outlier classification loss - your existing code
        if self.beta:
            classificationLoss = self.outliers_loss(pred_cam, pred_outliers, data, epoch)

        # Compute combined loss - your existing code
        loss = self.alpha * ESFMLoss + self.beta * classificationLoss

        # ============================================
        # NEW: LOG TO CSV
        # ============================================
        if self.enable_csv_logging:
            self._log_to_dataframe(
                epoch=epoch,
                esfm_loss_raw=ESFMLoss.item(),
                classification_loss_raw=classificationLoss.item(),
                total_loss=loss.item()
            )

            # Save CSV periodically
            if epoch is not None and epoch != self.last_saved_epoch and epoch % self.save_frequency == 0:
                self._save_to_csv()
                self.last_saved_epoch = epoch

        return loss
    
    def _log_to_dataframe(self, epoch, esfm_loss_raw, classification_loss_raw, total_loss):
        """
        Log loss magnitudes to DataFrame
        
        Line-by-line explanation:
        - Increment iteration counter for tracking
        - Calculate magnitude ratio (ESFM / classification)
        - Compute weighted losses: alpha * ESFM, beta * classification
        - Calculate contribution percentages for each component
        - Create dictionary with all tracked metrics
        - Append new row to DataFrame using pd.concat
        """
        self.iteration_count += 1
        
        # Calculate ratios and weighted values
        magnitude_ratio = esfm_loss_raw / (classification_loss_raw + 1e-8)
        
        weighted_esfm = self.alpha * esfm_loss_raw
        weighted_classification = self.beta * classification_loss_raw
        
        # Calculate contribution percentages
        total_weighted = weighted_esfm + weighted_classification + 1e-8
        esfm_contribution = 100 * weighted_esfm / total_weighted
        classification_contribution = 100 * weighted_classification / total_weighted
        
        # Create new row
        new_row = {
            'epoch': epoch if epoch is not None else -1,
            'iteration': self.iteration_count,
            'esfm_loss_raw': esfm_loss_raw,
            'classification_loss_raw': classification_loss_raw,
            'loss_magnitude_ratio': magnitude_ratio,
            'weighted_esfm_loss': weighted_esfm,
            'weighted_classification_loss': weighted_classification,
            'total_loss': total_loss,
            'esfm_contribution_percent': esfm_contribution,
            'classification_contribution_percent': classification_contribution,
            'alpha': self.alpha,
            'beta': self.beta
        }
        
        # Append to DataFrame
        self.loss_tracking_df = pd.concat([
            self.loss_tracking_df, 
            pd.DataFrame([new_row])
        ], ignore_index=True)
    
    def _save_to_csv(self):
        """
        Save the tracking DataFrame to CSV
        
        Line-by-line explanation:
        - Write DataFrame to CSV file at configured path
        - Print confirmation message with file location
        - Display latest statistics including epoch, losses, and contributions
        - Show percentage breakdown of how each loss contributes
        - Handle any IO errors gracefully with error message
        """
        try:
            self.loss_tracking_df.to_csv(self.csv_save_path, index=False)
            
            # Print concise update
            if len(self.loss_tracking_df) > 0:
                latest = self.loss_tracking_df.iloc[-1]
                print(f"\n{'='*80}")
                print(f"[Loss Tracking] Saved to: {self.csv_save_path}")
                print(f"  Iterations logged: {len(self.loss_tracking_df)} | Latest epoch: {int(latest['epoch'])}")
                print(f"  ─────────────────────────────────────────────")
                print(f"  Raw losses:")
                print(f"    ESFM:           {latest['esfm_loss_raw']:.6f}")
                print(f"    Classification: {latest['classification_loss_raw']:.6f}")
                print(f"    Ratio (E/C):    {latest['loss_magnitude_ratio']:.4f}")
                print(f"  ─────────────────────────────────────────────")
                print(f"  Weighted losses:")
                print(f"    ESFM:           {latest['weighted_esfm_loss']:.6f} ({latest['esfm_contribution_percent']:.1f}%)")
                print(f"    Classification: {latest['weighted_classification_loss']:.6f} ({latest['classification_contribution_percent']:.1f}%)")
                print(f"    Total:          {latest['total_loss']:.6f}")
                print(f"{'='*80}\n")
                
        except Exception as e:
            print(f"[Loss Tracking] Error saving CSV: {e}")
    
    def finalize_and_save(self):
        """
        Final save at end of training with summary statistics
        
        Line-by-line explanation:
        - Force final CSV save of all accumulated data
        - Calculate and print average statistics across all iterations
        - Show mean raw losses and magnitude ratios
        - Display average contribution percentages
        - Print final few iterations as table for review
        """
        if not self.enable_csv_logging:
            return
        print("\n" + "="*80)
        print("FINALIZING COMBINED LOSS TRACKING")
        print("="*80)
        self._save_to_csv()
        
        # Print summary statistics
        if len(self.loss_tracking_df) > 0:
            print("\n📊 SUMMARY STATISTICS (All Iterations)")
            print("─" * 80)
            
            # Mean values across all training
            print(f"Mean ESFM loss (raw):           {self.loss_tracking_df['esfm_loss_raw'].mean():.6f}")
            print(f"Mean Classification loss (raw): {self.loss_tracking_df['classification_loss_raw'].mean():.6f}")
            print(f"Mean magnitude ratio:           {self.loss_tracking_df['loss_magnitude_ratio'].mean():.4f}")
            print(f"─" * 80)
            print(f"Mean ESFM contribution:         {self.loss_tracking_df['esfm_contribution_percent'].mean():.1f}%")
            print(f"Mean Classification contribution: {self.loss_tracking_df['classification_contribution_percent'].mean():.1f}%")
            print(f"─" * 80)
            
            # Show last 5 iterations
            print("\nFinal 5 iterations:")
            print(self.loss_tracking_df[['epoch', 'esfm_loss_raw', 'classification_loss_raw', 
                                          'loss_magnitude_ratio', 'esfm_contribution_percent', 
                                          'classification_contribution_percent']].tail(5).to_string(index=False))
        
        print("="*80 + "\n")




class GTLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.calibrated = conf.get_bool('dataset.calibrated')

    def forward(self, pred_cam, data, epoch=None):
        # Get orientation
        Vs_gt = data.y[:, 0:3, 0:3].inverse().transpose(1, 2)
        if self.calibrated:
            Rs_gt = geo_utils.rot_to_quat(torch.bmm(data.Ns_invT, Vs_gt).transpose(1, 2))

        # Get Location
        t_gt = -torch.bmm(data.y[:, 0:3, 0:3].inverse(), data.y[:, 0:3, 3].unsqueeze(-1)).squeeze()

        # Normalize scene by points
        # trans = pts3D_gt.mean(dim=1)
        # scale = (pts3D_gt - trans.unsqueeze(1)).norm(p=2, dim=0).mean()

        # Normalize scene by cameras
        trans = t_gt.mean(dim=0)
        scale = (t_gt - trans).norm(p=2, dim=1).mean()

        t_gt = (t_gt - trans)/scale
        new_Ps = geo_utils.batch_get_camera_matrix_from_Vt(Vs_gt, t_gt)

        Vs_invT = pred_cam["Ps_norm"][:, 0:3, 0:3]
        Vs = torch.inverse(Vs_invT).transpose(1, 2)
        ts = torch.bmm(-Vs.transpose(1, 2), pred_cam["Ps"][:, 0:3, 3].unsqueeze(dim=-1)).squeeze()

        # Translation error
        translation_err = (t_gt - ts).norm(p=2, dim=1)

        # Calculate error
        if self.calibrated:
            Rs = geo_utils.rot_to_quat(torch.bmm(data.Ns_invT, Vs).transpose(1, 2))
            orient_err = (Rs - Rs_gt).norm(p=2, dim=1)
        else:
            Vs_gt = Vs_gt / Vs_gt.norm(p='fro', dim=(1, 2), keepdim=True)
            Vs = Vs / Vs.norm(p='fro', dim=(1, 2), keepdim=True)
            orient_err = torch.min((Vs - Vs_gt).norm(p='fro', dim=(1, 2)), (Vs + Vs_gt).norm(p='fro', dim=(1, 2)))

        orient_loss = orient_err.mean()
        tran_loss = translation_err.mean()
        loss = orient_loss + tran_loss

        if epoch is not None and epoch % 1000 == 0:
            # Print loss
            print("loss = {}, orient err = {}, trans err = {}".format(loss, orient_loss, tran_loss))

        return loss



class AdaptiveConfidenceWeightedOutliersLoss(nn.Module):
    """
    Confidence-weighted self-supervision with ADAPTIVE thresholds.
    Suitable for multi-scene training!
    """
    def __init__(self, conf):
        super().__init__()
        # Percentiles (adapt to each scene)
        self.inlier_percentile = conf.get_float('loss.inlier_percentile', default=20.0)
        self.outlier_percentile = conf.get_float('loss.outlier_percentile', default=80.0)
        
        # Safety parameters
        self.min_confident_samples = conf.get_int('loss.min_confident_samples', default=10)
        self.min_separation = conf.get_float('loss.min_threshold_separation', default=0.5)
        self.warmup_epochs = conf.get_int('loss.warmup_epochs', default=10)
        
    def forward(self, pred_cam, pred_outliers, data, epoch=None):
        """
        Generate pseudo-labels with adaptive confidence weighting
        """
     

        Ps = pred_cam["Ps_norm"]
        pts_2d = Ps @ pred_cam["pts3D"]
        
        # Compute reprojection errors
        projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, 0.1)
        pts_2d_norm = pts_2d / torch.where(
            projected_points, 
            pts_2d[:, 2, :], 
            torch.ones_like(projected_points).float()
        ).unsqueeze(dim=1)
        
        reproj_err = (pts_2d_norm[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)
        errors_flat = reproj_err[data.valid_pts].detach()
        
        # Edge case: too few points
        if len(errors_flat) < self.min_confident_samples * 2:
            return torch.tensor(0.0, device=pred_outliers.device)
        
        # ============================================
        # KEY: ADAPTIVE THRESHOLDS (per-scene)
        # ============================================
        low_threshold = torch.quantile(errors_flat, self.inlier_percentile / 100.0)
        high_threshold = torch.quantile(errors_flat, self.outlier_percentile / 100.0)
        
        # Ensure minimum separation (handle degenerate distributions)
        if high_threshold - low_threshold < self.min_separation:
            mid = (high_threshold + low_threshold) / 2.0
            low_threshold = mid - self.min_separation / 2.0
            high_threshold = mid + self.min_separation / 2.0
        
        # High-confidence samples only
        confident_inliers = errors_flat < low_threshold
        confident_outliers = errors_flat > high_threshold
        confident_mask = confident_inliers | confident_outliers
        
        # Ensure sufficient confident samples
        if confident_mask.sum() < self.min_confident_samples:
            return torch.tensor(0.0, device=pred_outliers.device)
        
        # Generate pseudo-labels (only for confident samples)
        pseudo_labels = torch.zeros_like(errors_flat)
        pseudo_labels[confident_outliers] = 1.0  # Outlier label
        pseudo_labels[confident_inliers] = 0.0   # Inlier label
        
        # Compute loss (only on confident samples)
        loss = F.binary_cross_entropy(
            pred_outliers.squeeze()[confident_mask],
            pseudo_labels[confident_mask]
        )
        
        # Optional: Warmup (gradually increase supervision)
        if epoch is not None and epoch < self.warmup_epochs:
            loss = loss * (epoch / self.warmup_epochs)

        return loss


class CombinedLossSupervised(nn.Module):
    """
    RESfM's ORIGINAL supervised combined loss (outlier-weighted reprojection +
    GT-label balanced BCE), ported from resfm-main-orig/code/loss_functions.py.
    Kept under a distinct name because in this repo `CombinedLoss` denotes the
    unsupervised U-RESfM loss. Signature matches this repo's train.py call
    (the original also received an unused pred_weights_M argument).
    """
    def __init__(self, conf):
        super().__init__()
        self.outliers_loss = OutliersLoss(conf)
        self.weighted_ESFM_loss = ESFMLoss_weighted(conf)
        self.alpha = conf.get_float('loss.reproj_loss_weight')
        self.beta = conf.get_float('loss.classification_loss_weight')

    def forward(self, pred_cam, pred_outliers, data, epoch=None):
        classification_loss = torch.tensor([0.0], device=pred_outliers.device)
        reproj_loss = torch.tensor([0.0], device=pred_outliers.device)

        # Reprojection loss (geometric loss)
        if self.alpha:
            reproj_loss = self.weighted_ESFM_loss(pred_cam, pred_outliers, data)

        # Outlier classification loss (uses GT labels in data.outlier_indices)
        if self.beta:
            classification_loss = self.outliers_loss(pred_outliers, data)

        loss = self.alpha * reproj_loss + self.beta * classification_loss

        return loss

