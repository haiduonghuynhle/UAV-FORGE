"""
Local Geometric Invariant (LGI) Filter.

RESEARCH CONTRIBUTION (Proposal §3.2):
Filters false-positive matches caused by repetitive structures
(housing arrays, solar fields, agricultural rows) using local
geometric consistency via k-NN graph analysis.

Key insight:
    True matches preserve local geometric relationships (scale,
    relative position) while false matches from repeated patterns
    violate these constraints.

Algorithm:
    1. Build k-NN graph over query keypoint positions
    2. For each keypoint, check if its k neighbors maintain
       consistent scale ratio with the global transformation
    3. Reject matches where local scale deviates from global
       scale by more than τ_scale
    4. Optionally verify angular consistency of neighbor vectors

Reference:
    - Bian et al., "GMS: Grid-based Motion Statistics" (CVPR 2017)
    - Ma et al., "Locality Preserving Matching" (IJCV 2019)
    - Proposal §3.2: Geometric Consensus Filtering
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LGIFilter:
    """
    Local Geometric Invariant Filter for robust match filtering.
    
    Removes false matches from repetitive structures by enforcing
    local geometric consistency using k-NN graph analysis.
    """
    
    def __init__(self,
                 k_neighbors: int = 5,
                 scale_threshold: float = 0.3,
                 angle_threshold_deg: float = 30.0,
                 min_matches: int = 10,
                 use_angular_check: bool = True):
        """
        Initialize LGI filter.
        
        Args:
            k_neighbors: Number of nearest neighbors for consistency check
            scale_threshold: Maximum allowed scale deviation ratio (0-1).
                            0.3 means allow ±30% scale variation.
            angle_threshold_deg: Maximum angular deviation for neighbor vectors
            min_matches: Minimum matches to attempt filtering
            use_angular_check: Also check angular consistency (slower but robust)
        """
        self._k = k_neighbors
        self._scale_threshold = scale_threshold
        self._angle_threshold = np.radians(angle_threshold_deg)
        self._min_matches = min_matches
        self._use_angular = use_angular_check
    
    def _build_knn_graph(self, points: np.ndarray, k: int) -> np.ndarray:
        """
        Build k-NN graph from 2D point positions.
        
        Args:
            points: Nx2 array of point coordinates
            k: Number of neighbors
            
        Returns:
            Nx(k) array of neighbor indices
        """
        N = len(points)
        k = min(k, N - 1)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # NxNx2
        dists = np.linalg.norm(diff, axis=2)  # NxN
        
        # Set self-distance to infinity
        np.fill_diagonal(dists, np.inf)
        
        # Get k nearest neighbors
        neighbors = np.argsort(dists, axis=1)[:, :k]  # Nx(k)
        
        return neighbors
    
    def _compute_local_scale(self,
                              query_pts: np.ndarray,
                              ref_pts: np.ndarray,
                              neighbors: np.ndarray) -> np.ndarray:
        """
        Compute local scale ratio for each match.
        
        For each match i, compute the mean distance to its k neighbors
        in both query and reference space. Scale ratio = ref_dist / query_dist.
        
        Args:
            query_pts: Nx2 query keypoints
            ref_pts: Nx2 reference keypoints
            neighbors: Nxk neighbor indices
            
        Returns:
            N array of local scale ratios
        """
        N = len(query_pts)
        k = neighbors.shape[1]
        
        local_scales = np.zeros(N)
        
        for i in range(N):
            nbrs = neighbors[i]
            
            # Distances in query space
            q_dists = np.linalg.norm(query_pts[nbrs] - query_pts[i], axis=1)
            
            # Distances in reference space
            r_dists = np.linalg.norm(ref_pts[nbrs] - ref_pts[i], axis=1)
            
            # Scale ratio (handle zero distances)
            valid = q_dists > 1.0  # Minimum 1 pixel
            if valid.sum() > 0:
                scales = r_dists[valid] / q_dists[valid]
                local_scales[i] = np.median(scales)
            else:
                local_scales[i] = 1.0  # Default
        
        return local_scales
    
    def _compute_angular_consistency(self,
                                      query_pts: np.ndarray,
                                      ref_pts: np.ndarray,
                                      neighbors: np.ndarray) -> np.ndarray:
        """
        Compute angular consistency of neighbor vectors.
        
        For each match i and each neighbor j, compute the angle of the
        vector i→j in both query and reference space. True matches
        should have consistent relative angles.
        
        Args:
            query_pts: Nx2 query points
            ref_pts: Nx2 reference points
            neighbors: Nxk neighbor indices
            
        Returns:
            N array of angular consistency scores [0, 1] (1 = consistent)
        """
        N = len(query_pts)
        k = neighbors.shape[1]
        
        angular_scores = np.ones(N)
        
        for i in range(N):
            nbrs = neighbors[i]
            
            # Angles in query space
            q_vecs = query_pts[nbrs] - query_pts[i]
            q_angles = np.arctan2(q_vecs[:, 1], q_vecs[:, 0])
            
            # Angles in reference space
            r_vecs = ref_pts[nbrs] - ref_pts[i]
            r_angles = np.arctan2(r_vecs[:, 1], r_vecs[:, 0])
            
            # Angular differences (should be constant = global rotation)
            angle_diffs = r_angles - q_angles
            # Wrap to [-π, π]
            angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
            
            # Consistency = low variance of angle differences
            if len(angle_diffs) > 1:
                std_angle = np.std(angle_diffs)
                # Normalize: 0 variance → score 1, high variance → score 0
                angular_scores[i] = np.exp(-std_angle / self._angle_threshold)
            else:
                angular_scores[i] = 1.0
        
        return angular_scores
    
    def filter_matches(self,
                        query_keypoints: np.ndarray,
                        ref_keypoints: np.ndarray,
                        confidence: np.ndarray = None,
                        inlier_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter matches using Local Geometric Invariant analysis.
        
        Args:
            query_keypoints: Nx2 query keypoints
            ref_keypoints: Nx2 reference keypoints
            confidence: N array of match confidence (optional)
            inlier_mask: N boolean mask from RANSAC (optional, applied first)
            
        Returns:
            Tuple of:
                - filtered_query: Mx2 filtered query keypoints
                - filtered_ref: Mx2 filtered reference keypoints
                - lgi_mask: N boolean mask (True = kept)
        """
        N = len(query_keypoints)
        
        if N < self._min_matches:
            logger.debug(f"Too few matches ({N}) for LGI filtering, skipping")
            return query_keypoints, ref_keypoints, np.ones(N, dtype=bool)
        
        # Apply RANSAC inlier mask first
        working_mask = np.ones(N, dtype=bool)
        if inlier_mask is not None:
            working_mask = inlier_mask.copy()
        
        # Get working set
        q_pts = query_keypoints[working_mask]
        r_pts = ref_keypoints[working_mask]
        
        if len(q_pts) < self._min_matches:
            return query_keypoints[working_mask], ref_keypoints[working_mask], working_mask
        
        # Build k-NN graph on query keypoints
        k = min(self._k, len(q_pts) - 1)
        if k < 2:
            return q_pts, r_pts, working_mask
        
        neighbors = self._build_knn_graph(q_pts, k)
        
        # Compute local scales
        local_scales = self._compute_local_scale(q_pts, r_pts, neighbors)
        
        # Global scale (median of all local scales)
        global_scale = np.median(local_scales)
        
        if global_scale < 1e-6:
            logger.warning("Global scale near zero, skipping LGI filter")
            return q_pts, r_pts, working_mask
        
        # Scale consistency check
        scale_ratios = local_scales / global_scale
        scale_consistent = np.abs(scale_ratios - 1.0) < self._scale_threshold
        
        # Angular consistency check (optional)
        if self._use_angular and len(q_pts) > self._k + 1:
            angular_scores = self._compute_angular_consistency(q_pts, r_pts, neighbors)
            angular_consistent = angular_scores > 0.5  # Score threshold
            
            # Combined mask
            lgi_pass = scale_consistent & angular_consistent
        else:
            lgi_pass = scale_consistent
        
        # Update the full mask
        working_indices = np.where(working_mask)[0]
        lgi_mask = np.zeros(N, dtype=bool)
        lgi_mask[working_indices[lgi_pass]] = True
        
        n_filtered = working_mask.sum() - lgi_pass.sum()
        logger.info(
            f"LGI filter: {working_mask.sum()} → {lgi_pass.sum()} matches "
            f"(removed {n_filtered}, global_scale={global_scale:.2f})"
        )
        
        return query_keypoints[lgi_mask], ref_keypoints[lgi_mask], lgi_mask
    
    def filter_match_result(self, match_result: 'MatchResult') -> 'MatchResult':
        """
        Filter a MatchResult object in-place.
        
        Args:
            match_result: MatchResult from matcher
            
        Returns:
            New MatchResult with LGI-filtered matches
        """
        try:
            from core.types import MatchResult
        except ImportError:
            class MatchResult:  # fallback minimal contract
                def __init__(self, query_keypoints, ref_keypoints, confidence, homography=None, inlier_mask=None, metadata=None):
                    self.query_keypoints = np.asarray(query_keypoints, dtype=np.float64)
                    self.ref_keypoints = np.asarray(ref_keypoints, dtype=np.float64)
                    self.confidence = np.asarray(confidence, dtype=np.float64)
                    self.homography = homography
                    self.inlier_mask = inlier_mask
                    self.metadata = metadata or {}

                @property
                def num_inliers(self):
                    if self.inlier_mask is None:
                        return len(self.query_keypoints)
                    return int(np.sum(self.inlier_mask))
        
        filtered_q, filtered_r, lgi_mask = self.filter_matches(
            match_result.query_keypoints,
            match_result.ref_keypoints,
            match_result.confidence,
            match_result.inlier_mask
        )
        
        # Filter confidence
        conf = match_result.confidence[lgi_mask] if len(match_result.confidence) == len(lgi_mask) else match_result.confidence
        
        # Recompute homography on filtered matches
        homography = None
        inlier_mask = None
        if len(filtered_q) >= 4:
            try:
                import cv2
                homography, mask = cv2.findHomography(
                    filtered_q, filtered_r, cv2.RANSAC, 5.0
                )
                inlier_mask = mask.ravel().astype(bool) if mask is not None else None
            except Exception:
                pass
        
        return MatchResult(
            query_keypoints=filtered_q,
            ref_keypoints=filtered_r,
            confidence=conf,
            homography=homography,
            inlier_mask=inlier_mask,
            metadata={
                **(match_result.metadata or {}),
                'lgi_filtered': True,
                'lgi_removed': int((~lgi_mask).sum()),
            }
        )
