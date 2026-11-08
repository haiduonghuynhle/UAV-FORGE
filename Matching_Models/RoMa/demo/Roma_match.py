from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path

from Matching_Models.RoMa.roma import roma_outdoor
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform


def Roma_Init(roma_weights_path=None, dinov2_weights_path=None):
    device = 'cuda'

    repo_root = Path(__file__).resolve().parents[3]
    default_roma = repo_root / "Matching_Models" / "RoMa" / "weights" / "roma_outdoor.pth"
    default_dino = repo_root / "Matching_Models" / "RoMa" / "weights" / "dinov2_vitl14_pretrain.pth"

    if roma_weights_path is not None:
        roma_weights = Path(roma_weights_path)
    else:
        roma_weights = Path(os.environ.get("ROMA_WEIGHTS_PATH", str(default_roma)))

    if dinov2_weights_path is not None:
        dino_weights = Path(dinov2_weights_path)
    else:
        dino_weights = Path(os.environ.get("DINOV2_WEIGHTS_PATH", str(default_dino)))

    if roma_weights.exists() and dino_weights.exists():
        print(f"Using local RoMa weights: {roma_weights}")
        print(f"Using local DINOv2 weights: {dino_weights}")
        roma_model = roma_outdoor(
            device=device,
            weights=str(roma_weights),
            dinov2_weights=str(dino_weights),
        )
    else:
        print("Local RoMa/DINOv2 weights not found, fallback to torch.hub cache/download.")
        roma_model = roma_outdoor(device=device)

    return roma_model


def Roma_match(image0, image1, roma_model, save_path, ransac_name, need_ransac , show_matches):
    result_save_path = save_path + ransac_name
    image0_origin = image0
    image1_origin = image1
    device = 'cuda'
    W_A, H_A = image0.shape[1], image0.shape[0]
    W_B, H_B = image1.shape[1], image1.shape[0]
    image1_PIL = Image.fromarray(image0)
    image2_PIL = Image.fromarray(image1)
    warp, certainty = roma_model.match(image1_PIL, image2_PIL, device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty, num=3000)
    keypoints_left, keypoints_right = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    mkpts0, mkpts1 = keypoints_left.cpu().numpy(), keypoints_right.cpu().numpy()


    n_inliers1 = 0
    inliers = [None]

    if len(mkpts0)>5 and need_ransac:

        _, inliers = ransac(
            (mkpts0, mkpts1),
            ProjectiveTransform, min_samples=4,
            residual_threshold=4, max_trials=10000
        )
    # if len(mkpts0)>8:
    #     np.random.seed(0)
    #     _, inliers = ransac(
    #         (mkpts0, mkpts1),
    #         FundamentalMatrixTransform, min_samples=8,
    #         residual_threshold=4, max_trials=10000
    #     )
        n_inliers1 = np.sum(inliers)
        inlier_keys_left = [[point[0], point[1]] for point in mkpts0[inliers]]
        inlier_keys_right = [[point[0], point[1]] for point in mkpts1[inliers]]
    else:
        inlier_keys_left = [[point[0], point[1]] for point in mkpts0[:]]
        inlier_keys_right = [[point[0], point[1]] for point in mkpts1[:]]

    if show_matches and len(inliers)>5:


        inlier_keypoints_left1 = [cv2.KeyPoint(point[0], point[1], 1) for point in mkpts0[inliers]]
        inlier_keypoints_right1 = [cv2.KeyPoint(point[0], point[1], 1) for point in mkpts1[inliers]]
        placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers1)]

        image3 = cv2.drawMatches(image0_origin, inlier_keypoints_left1, image1_origin, inlier_keypoints_right1, placeholder_matches,
                                 None)
        cv2.imwrite(result_save_path, image3)

    return inlier_keys_left, inlier_keys_right


def draw_matches(im_A, kpts_A, im_B, kpts_B):
    kpts_A = [cv2.KeyPoint(x,y,1.) for x,y in kpts_A.cpu().numpy()]
    kpts_B = [cv2.KeyPoint(x,y,1.) for x,y in kpts_B.cpu().numpy()]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.) for idx in range(len(kpts_A))]
    im_A, im_B = np.array(im_A), np.array(im_B)
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B,
                    matches_A_to_B, None)
    return ret
