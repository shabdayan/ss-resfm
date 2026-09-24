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
        
        # Loss components. The report (sec 2.2.2, p11) specifies that the outlier
        # detector's scores WEIGHT the reprojection loss during training (predicted
        # inliers keep weight, predicted outliers are down-weighted). Selectable via
        # loss.reproj_weighting:
        #   'none'            plain ESFMLoss, scores do NOT weight geometry. This is
        #                     what the frozen single-scene benchmark ran (kept as
        #                     default so those confs reproduce exactly).
        #   'weighted'        ESFMLoss_weighted((1-score)*err), report-literal — the
        #                     confident-pseudo-label BCE is the anti-collapse mechanism.
        #   'weighted_detach' (1-score.detach())*err — geometry is down-weighted but
        #                     scores learn only from the BCE (extra collapse guard).
        self.reproj_weighting = conf.get_string('loss.reproj_weighting', default='none')
        self.outliers_loss = AdaptiveConfidenceWeightedOutliersLoss(conf)
        self.esfm_loss = ESFMLoss(conf)
        self.weighted_esfm_loss = ESFMLoss_weighted(conf)
        
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

        # Reprojection (geometric) term. Weighted by the detector scores per the
        # report (sec 2.2.2) when loss.reproj_weighting != 'none'.
        if self.alpha:
            if self.reproj_weighting == 'none':
                ESFMLoss = self.esfm_loss(pred_cam, data)
            elif self.reproj_weighting == 'weighted':
                ESFMLoss = self.weighted_esfm_loss(pred_cam, pred_outliers, data, epoch)
            elif self.reproj_weighting == 'weighted_detach':
                ESFMLoss = self.weighted_esfm_loss(pred_cam, pred_outliers.detach(), data, epoch)
            else:
                raise ValueError(f"Unknown loss.reproj_weighting: {self.reproj_weighting}")

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
        
        # Safety parameters.
        self.min_confident_samples = conf.get_int('loss.min_confident_samples', default=10)
        # Minimum gap between the inlier/outlier thresholds, to keep pseudo-labels
        # confident when the error distribution is peaked. This MUST be scale-aware:
        # reprojection errors here live in NORMALIZED image coordinates (~0.01-0.2),
        # not pixels, so the old absolute default of 0.5 was larger than the entire
        # error range and pushed the inlier threshold below zero (no confident
        # inliers ever). Now expressed as a FRACTION of a robust error scale (the
        # median error) computed per forward pass, so it works in any coordinate
        # system. Set loss.min_threshold_separation (absolute, same units as the
        # errors) to override with a hard value; default -1 = use the fraction.
        self.min_separation_frac = conf.get_float('loss.min_threshold_separation_frac', default=0.25)
        self.min_separation_abs = conf.get_float('loss.min_threshold_separation', default=-1.0)
        self.warmup_epochs = conf.get_int('loss.warmup_epochs', default=10)

        # A2.5.1 (diagnosis task): contamination-linked outlier percentile.
        # The fixed outlier_percentile encodes a ~20% contamination prior; when
        # enabled, estimate the scene's outlier fraction per forward with a MAD
        # rule and label the top est-fraction as confident outliers instead.
        # Off by default — existing confs are unaffected.
        self.contamination_linked = conf.get_bool('loss.contamination_linked_percentile', default=False)
        self.contamination_mad_alpha = conf.get_float('loss.contamination_mad_alpha', default=2.0)
        
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
        outlier_percentile = self.outlier_percentile
        if self.contamination_linked:
            med = errors_flat.median()
            mad = (errors_flat - med).abs().median()
            est_frac = (errors_flat > med + self.contamination_mad_alpha * 1.4826 * mad).float().mean()
            outlier_percentile = float(torch.clamp(100.0 * (1.0 - est_frac), 55.0, 95.0))
        low_threshold = torch.quantile(errors_flat, self.inlier_percentile / 100.0)
        high_threshold = torch.quantile(errors_flat, outlier_percentile / 100.0)

        # Ensure minimum separation (handle degenerate/peaked distributions).
        # Scale-aware: unless an absolute override is given, the floor is a fraction
        # of the median error, so it tracks the (normalized) coordinate scale and
        # can never exceed the error range. clamp(min=0) keeps a non-negative inlier
        # threshold even under a hard absolute override.
        if self.min_separation_abs >= 0.0:
            min_sep = self.min_separation_abs
        else:
            min_sep = self.min_separation_frac * torch.median(errors_flat)
        if high_threshold - low_threshold < min_sep:
            mid = (high_threshold + low_threshold) / 2.0
            low_threshold = torch.clamp(mid - min_sep / 2.0, min=0.0)
            high_threshold = mid + min_sep / 2.0
        
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

        # Diagnostics for the TTT collapse-check figure (SPEC_cvpr_experiments R2.2)
        with torch.no_grad():
            self.last_stats = {
                'pred_outlier_frac': float((pred_outliers.squeeze() > 0.5).float().mean()),
                'low_threshold': float(low_threshold),
                'high_threshold': float(high_threshold),
                'outlier_percentile_used': float(outlier_percentile),
                'confident_frac': float(confident_mask.float().mean()),
            }
        
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



class CombinedLossDualSupervision(nn.Module):
    """
    Dual-supervision combined loss: the RESfM supervised recipe and the
    SS-RESfM self-supervised recipe carrying their classification targets
    TOGETHER rather than as alternatives.

        loss = alpha * outlier-weighted reprojection
             + beta  * BCE(pred_outliers, GT labels)            [supervised]
             + gamma * adaptive confident-pseudo-label loss      [self-supervised]

    beta and gamma split the SAME classification budget the single-target
    recipes use (loss.classification_loss_weight), so a run differs from the
    supervised baseline only in WHICH targets the classification term follows,
    not in how much classification is applied. The split is set by
    loss.gt_label_fraction in [0, 1]:
        beta  = classification_loss_weight * gt_label_fraction
        gamma = classification_loss_weight * (1 - gt_label_fraction)
    """
    def __init__(self, conf):
        super().__init__()
        self.outliers_loss = OutliersLoss(conf)
        self.adaptive_loss = AdaptiveConfidenceWeightedOutliersLoss(conf)
        self.weighted_ESFM_loss = ESFMLoss_weighted(conf)
        self.alpha = conf.get_float('loss.reproj_loss_weight')
        cls_weight = conf.get_float('loss.classification_loss_weight')
        gt_fraction = conf.get_float('loss.gt_label_fraction', default=0.5)
        self.beta = cls_weight * gt_fraction
        self.gamma = cls_weight * (1.0 - gt_fraction)

    def forward(self, pred_cam, pred_outliers, data, epoch=None):
        device = pred_outliers.device
        reproj_loss = torch.tensor([0.0], device=device)
        gt_loss = torch.tensor([0.0], device=device)
        ss_loss = torch.tensor([0.0], device=device)

        if self.alpha:
            reproj_loss = self.weighted_ESFM_loss(pred_cam, pred_outliers, data)
        if self.beta:
            gt_loss = self.outliers_loss(pred_outliers, data)
        if self.gamma:
            ss_loss = self.adaptive_loss(pred_cam, pred_outliers, data, epoch)

        return self.alpha * reproj_loss + self.beta * gt_loss + self.gamma * ss_loss


class RobustScaleReprojLoss(nn.Module):
    """
    Learned heavy-tailed reprojection loss (design idea 1).

    The head emits a per-observation log-scale s (model.predict_obs_scale);
    residuals are scored under a Cauchy negative log-likelihood

        NLL = log(s) + log(1 + (r/s)^2)

    so the network *learns* how much each observation should be trusted instead
    of having a removal/reweighting mechanism chosen per dataset: large learned
    s downweights an observation smoothly (the soft-weighting limit) and very
    large s makes its gradient vanish (the removal limit). log(s) is the
    penalty that stops the trivial "everything is uncertain" solution.
    Geometry (Ps, pts3D, hinge handling) is computed exactly as in
    ESFMLoss_weighted so the two are directly comparable.
    """
    def __init__(self, conf):
        super().__init__()
        self.infinity_pts_margin = conf.get_float("loss.infinity_pts_margin")
        self.normalize_grad = conf.get_bool("loss.normalize_grad")
        self.hinge_loss = conf.get_bool("loss.hinge_loss")
        self.hinge_loss_weight = conf.get_float("loss.hinge_loss_weight") if self.hinge_loss else 0
        # scale is parameterised as s = softplus(raw) + min_scale, keeping s > 0
        self.min_scale = conf.get_float('loss.robust_min_scale', default=1e-3)

    def forward(self, pred_cam, pred_outliers, data, epoch=None):
        Ps = pred_cam["Ps_norm"]
        pts_2d = Ps @ pred_cam["pts3D"]
        if self.normalize_grad:
            pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / data.valid_pts.sum())
        if self.hinge_loss:
            projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)
        else:
            projected_points = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight
        pts_2d = (pts_2d / torch.where(projected_points, pts_2d[:, 2, :],
                                       torch.ones_like(projected_points).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)
        projected_points = projected_points[data.valid_pts]
        reproj_err = reproj_err[data.valid_pts]
        hinge_loss = hinge_loss[data.valid_pts]

        # pred_outliers: [nnz, 2] = (outlier score, raw log-scale)
        raw_scale = pred_outliers[:, 1] if pred_outliers.dim() > 1 and pred_outliers.shape[1] > 1 \
            else torch.zeros_like(reproj_err)
        scale = F.softplus(raw_scale) + self.min_scale
        nll = torch.log(scale) + torch.log1p((reproj_err / scale) ** 2)
        robust = torch.where(projected_points, nll, hinge_loss)
        return robust.mean()


class CombinedLossRobustScale(nn.Module):
    """
    Self-supervised recipe with the learned robust reprojection loss in place of
    the outlier-weighted one: alpha * RobustScaleReprojLoss + beta * adaptive
    confident-pseudo-label loss (the pseudo-label term keeps the score channel
    meaningful so the existing test-time mechanisms still apply).
    """
    def __init__(self, conf):
        super().__init__()
        self.robust_loss = RobustScaleReprojLoss(conf)
        self.adaptive_loss = AdaptiveConfidenceWeightedOutliersLoss(conf)
        self.alpha = conf.get_float('loss.reproj_loss_weight')
        self.beta = conf.get_float('loss.classification_loss_weight')

    def forward(self, pred_cam, pred_outliers, data, epoch=None):
        reproj = self.robust_loss(pred_cam, pred_outliers, data, epoch)
        cls = self.adaptive_loss(pred_cam, pred_outliers[:, :1], data, epoch) if self.beta \
            else torch.tensor([0.0], device=pred_outliers.device)
        return self.alpha * reproj + self.beta * cls
