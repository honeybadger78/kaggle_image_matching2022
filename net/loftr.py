import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF


class LoFTRMatcher:
    def __init__(self, config):
        self.config = config["LoFTR"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loftr_matcher = KF.LoFTR(pretrained=None)
        self._loftr_matcher.load_state_dict(
            torch.load(self.config["weight"])["state_dict"]
        )
        self._loftr_matcher = self._loftr_matcher.to(self.device).eval()
        self.input_longside = config["input_longside"]
        self.conf_thresh = config["conf_thresh"]

    def prep_img(self, img):
        scale = self.input_longside / max(img.shape[0], img.shape[1])
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
        img_ts = K.image_to_tensor(img_resized, False).float() / 255.0
        img_ts = K.color.bgr_to_rgb(img_ts)
        img_ts = K.color.rgb_to_grayscale(img_ts)
        return img_resized, img_ts.to(self.device), scale

    def tta_rotation_preprocess(self, img_np, angle):
        rot_M = cv2.getRotationMatrix2D(
            (img_np.shape[1] / 2, img_np.shape[0] / 2), angle, 1
        )
        rot_M_inv = cv2.getRotationMatrix2D(
            (img_np.shape[1] / 2, img_np.shape[0] / 2), -angle, 1
        )
        rot_img = cv2.warpAffine(img_np, rot_M, (img_np.shape[1], img_np.shape[0]))
        rot_img_ts = K.image_to_tensor(rot_img, False).float() / 255.0
        rot_img_ts = K.color.bgr_to_rgb(rot_img_ts)
        rot_img_ts = K.color.rgb_to_grayscale(rot_img_ts)
        return rot_M, rot_img_ts.to(self.device), rot_M_inv

    def tta_rotation_postprocess(self, kpts, rot_M_inv):
        ones = np.ones(shape=(kpts.shape[0],), dtype=np.float32)[:, None]
        hom = np.concatenate([kpts, ones], 1)
        rot_kpts = rot_M_inv.dot(hom.T).T[:, :2]
        mask = (
            (rot_kpts[:, 0] >= 0)
            & (rot_kpts[:, 0] < img_np.shape[1])
            & (rot_kpts[:, 1] >= 0)
            & (rot_kpts[:, 1] < img_np.shape[0])
        )
        return rot_kpts, mask

    def __call__(self, img_np1, img_np2, tta=["orig"]):
        with torch.no_grad():
            img_np1, img_ts0, scale0 = self.prep_img(img_np1)
            img_np2, img_ts1, scale1 = self.prep_img(img_np2)
            images0, images1 = [], []

            for tta_elem in tta:
                if tta_elem == "orig":
                    img_ts0_aug, img_ts1_aug = img_ts0, img_ts1
                elif tta_elem == "flip_lr":
                    img_ts0_aug = torch.flip(
                        img_ts0,
                        [
                            3,
                        ],
                    )
                    img_ts1_aug = torch.flip(
                        img_ts1,
                        [
                            3,
                        ],
                    )
                elif tta_elem == "flip_ud":
                    img_ts0_aug = torch.flip(
                        img_ts0,
                        [
                            2,
                        ],
                    )
                    img_ts1_aug = torch.flip(
                        img_ts1,
                        [
                            2,
                        ],
                    )
                elif tta_elem == "rot_r10":
                    (
                        rot_r10_M0,
                        img_ts0_aug,
                        rot_r10_M0_inv,
                    ) = self.tta_rotation_preprocess(img_np1, 10)
                    (
                        rot_r10_M1,
                        img_ts1_aug,
                        rot_r10_M1_inv,
                    ) = self.tta_rotation_preprocess(img_np2, 10)
                elif tta_elem == "rot_l10":
                    (
                        rot_l10_M0,
                        img_ts0_aug,
                        rot_l10_M0_inv,
                    ) = self.tta_rotation_preprocess(img_np1, -10)
                    (
                        rot_l10_M1,
                        img_ts1_aug,
                        rot_l10_M1_inv,
                    ) = self.tta_rotation_preprocess(img_np2, -10)
                else:
                    raise ValueError("Unknown TTA method.")
                images0.append(img_ts0_aug)
                images1.append(img_ts1_aug)

            input_dict = {"image0": torch.cat(images0), "image1": torch.cat(images1)}
            correspondences = self.loftr_matcher(input_dict)
            matching_key_points0 = correspondences["keypoints0"].cpu().numpy()
            matching_key_points1 = correspondences["keypoints1"].cpu().numpy()
            batch_id = correspondences["batch_indexes"].cpu().numpy()
            confidence = correspondences["confidence"].cpu().numpy()

            for idx, tta_elem in enumerate(tta):
                batch_mask = batch_id == idx

                if tta_elem == "orig":
                    pass
                elif tta_elem == "flip_lr":
                    matching_key_points0[batch_mask, 0] = img_np1.shape[1] - matching_key_points0[batch_mask, 0]
                    matching_key_points1[batch_mask, 0] = img_np2.shape[1] - matching_key_points1[batch_mask, 0]
                elif tta_elem == "flip_ud":
                    matching_key_points0[batch_mask, 1] = img_np1.shape[0] - matching_key_points0[batch_mask, 1]
                    matching_key_points1[batch_mask, 1] = img_np2.shape[0] - matching_key_points1[batch_mask, 1]
                elif tta_elem == "rot_r10":
                    matching_key_points0[batch_mask], mask0 = self.tta_rotation_postprocess(
                        matching_key_points0[batch_mask], img_np1, rot_r10_M0_inv
                    )
                    matching_key_points1[batch_mask], mask1 = self.tta_rotation_postprocess(
                        matching_key_points1[batch_mask], img_np2, rot_r10_M1_inv
                    )
                    confidence[batch_mask] += (~(mask0 & mask1)).astype(
                        np.float32
                    ) * -10.0
                elif tta_elem == "rot_l10":
                    matching_key_points0[batch_mask], mask0 = self.tta_rotation_postprocess(
                        matching_key_points0[batch_mask], img_np1, rot_l10_M0_inv
                    )
                    matching_key_points1[batch_mask], mask1 = self.tta_rotation_postprocess(
                        matching_key_points1[batch_mask], img_np2, rot_l10_M1_inv
                    )
                    confidence[batch_mask] += (~(mask0 & mask1)).astype(
                        np.float32
                    ) * -10.0
                else:
                    raise ValueError("Unknown TTA method.")

            if self.config["match_threshold"] is not None:
                th_mask = confidence >= self.config["match_threshold"]
            else:
                th_mask = confidence >= 0.0
            matching_key_points0, matching_key_points1 = matching_key_points0[th_mask, :], matching_key_points1[th_mask, :]
