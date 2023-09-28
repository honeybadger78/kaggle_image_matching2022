import cv2
import numpy as np

import torch
import torch.nn as nn
from superpoint.superpoint import SuperPoint
from superpoint.superglue import SuperGlue


class SuperGlueCustomMatching(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.superpoint = SuperPoint(config["superpoint"]).to(self.device)
        self.superglue = SuperGlue(config["superglue"]).to(self.device)

        self.tta_map = {
            "orig": self.untta_none,
            "eqhist": self.untta_none,
            "clahe": self.untta_none,
            "flip_lr": self.untta_fliplr,
            "flip_ud": self.untta_flipud,
            "rot_r10": self.untta_rotr10,
            "rot_l10": self.untta_rotl10,
            "fliplr_rotr10": self.untta_fliplr_rotr10,
            "fliplr_rotl10": self.untta_fliplr_rotl10,
        }

    def forward_flat(
        self,
        data,
        ttas=[
            "orig",
        ],
        tta_groups=[["orig"]],
    ):

        pred = {}


        if "keypoints0" not in data:
            pred0 = self.superpoint({"image": data["image0"]})
            pred = {**pred, **{k + "0": v for k, v in pred0.items()}}
        if "keypoints1" not in data:
            pred1 = self.superpoint({"image": data["image1"]})
            pred = {**pred, **{k + "1": v for k, v in pred1.items()}}

        pred["scores0"] = list(pred["scores0"])
        pred["scores1"] = list(pred["scores1"])
        for i in range(len(pred["keypoints0"])):
            (
                pred["keypoints0"][i],
                pred["descriptors0"][i],
                pred["scores0"][i],
            ) = self.tta_map[ttas[i]](
                pred["keypoints0"][i],
                pred["descriptors0"][i],
                pred["scores0"][i],
                w=data["image0"].shape[3],
                h=data["image0"].shape[2],
                inplace=True,
                mask_illegal=True,
            )

            (
                pred["keypoints1"][i],
                pred["descriptors1"][i],
                pred["scores1"][i],
            ) = self.tta_map[ttas[i]](
                pred["keypoints1"][i],
                pred["descriptors1"][i],
                pred["scores1"][i],
                w=data["image1"].shape[3],
                h=data["image1"].shape[2],
                inplace=True,
                mask_illegal=True,
            )

        data = {**data, **pred}

        group_preds = []
        for tta_group in tta_groups:
            group_mask = torch.from_numpy(
                np.array([x in tta_group for x in ttas], dtype=np.bool)
            )
            group_data = {
                **{
                    f"keypoints{k}": [
                        data[f"keypoints{k}"][i]
                        for i in range(len(ttas))
                        if ttas[i] in tta_group
                    ]
                    for k in [0, 1]
                },
                **{
                    f"descriptors{k}": [
                        data[f"descriptors{k}"][i]
                        for i in range(len(ttas))
                        if ttas[i] in tta_group
                    ]
                    for k in [0, 1]
                },
                **{
                    f"scores{k}": [
                        data[f"scores{k}"][i]
                        for i in range(len(ttas))
                        if ttas[i] in tta_group
                    ]
                    for k in [0, 1]
                },
                **{f"image{k}": data[f"image{k}"][group_mask, ...] for k in [0, 1]},
            }
            for k, v in group_data.items():
                if isinstance(group_data[k], (list, tuple)):
                    if k.startswith("descriptor"):
                        group_data[k] = torch.cat(group_data[k], 1)[None, ...]
                    else:
                        group_data[k] = torch.cat(group_data[k])[None, ...]
                else:
                    group_data[k] = torch.flatten(group_data[k], 0, 1)[None, ...]
            group_pred = {
                **group_data,
                **self.superglue(group_data),
            }

            group_preds.append(group_pred)
        return group_preds

    def forward_cross(
        self,
        data,
        ttas=[
            "orig",
        ],
        tta_groups=[("orig", "orig")],
    ):
        pred = {}

        if "keypoints0" not in data:
            pred0 = self.superpoint({"image": data["image0"]})
            pred = {**pred, **{k + "0": v for k, v in pred0.items()}}
        if "keypoints1" not in data:
            pred1 = self.superpoint({"image": data["image1"]})
            pred = {**pred, **{k + "1": v for k, v in pred1.items()}}


        data = {**data, **pred}

        group_pred_list = []
        tta2id = {k: i for i, k in enumerate(ttas)}
        for tta_group in tta_groups:
            group_idx = tta2id[tta_group[0]], tta2id[tta_group[1]]
            group_data = {
                **{
                    f"image{i}": data[f"image{i}"][group_idx[i] : group_idx[i] + 1]
                    for i in [0, 1]
                },
                **{
                    f"keypoints{i}": data[f"keypoints{i}"][
                        group_idx[i] : group_idx[i] + 1
                    ]
                    for i in [0, 1]
                },
                **{
                    f"descriptors{i}": data[f"descriptors{i}"][
                        group_idx[i] : group_idx[i] + 1
                    ]
                    for i in [0, 1]
                },
                **{
                    f"scores{i}": data[f"scores{i}"][group_idx[i] : group_idx[i] + 1]
                    for i in [0, 1]
                },
            }

            for k in group_data:
                if isinstance(group_data[k], (list, tuple)):
                    group_data[k] = torch.stack(group_data[k])

            group_sg_pred = self.superglue(group_data)
            group_pred_list.append(group_sg_pred)

        data["scores0"] = list(data["scores0"])
        data["scores1"] = list(data["scores1"])
        for i in range(len(data["keypoints0"])):
            (
                data["keypoints0"][i],
                data["descriptors0"][i],
                data["scores0"][i],
            ) = self.tta_map[ttas[i]](
                data["keypoints0"][i],
                data["descriptors0"][i],
                data["scores0"][i],
                w=data["image0"].shape[3],
                h=data["image0"].shape[2],
                inplace=True,
                mask_illegal=False,
            )

            (
                data["keypoints1"][i],
                data["descriptors1"][i],
                data["scores1"][i],
            ) = self.tta_map[ttas[i]](
                data["keypoints1"][i],
                data["descriptors1"][i],
                data["scores1"][i],
                w=data["image1"].shape[3],
                h=data["image1"].shape[2],
                inplace=True,
                mask_illegal=False,
            )

        for group_pred, tta_group in zip(group_pred_list, tta_groups):
            group_idx = tta2id[tta_group[0]], tta2id[tta_group[1]]
            group_pred.update(
                {
                    **{
                        f"keypoints{i}": data[f"keypoints{i}"][
                            group_idx[i] : group_idx[i] + 1
                        ]
                        for i in [0, 1]
                    },
                    **{
                        f"scores{i}": data[f"scores{i}"][
                            group_idx[i] : group_idx[i] + 1
                        ]
                        for i in [0, 1]
                    },
                }
            )
        return group_pred_list

    def untta_none(
        self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True
    ):
        if not inplace:
            keypoints = keypoints.clone()
        return keypoints, descriptors, scores

    def untta_fliplr(
        self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True
    ):
        if not inplace:
            keypoints = keypoints.clone()
        keypoints[:, 0] = w - keypoints[:, 0] - 1.0
        return keypoints, descriptors, scores

    def untta_flipud(
        self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True
    ):
        if not inplace:
            keypoints = keypoints.clone()
        keypoints[:, 1] = h - keypoints[:, 1] - 1.0
        return keypoints, descriptors, scores

    def untta_rotr10(
        self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True
    ):
        rot_M_inv = (
            torch.from_numpy(cv2.getRotationMatrix2D((w / 2, h / 2), -15, 1))
            .to(torch.float32)
            .to(self.device)
        )
        ones = torch.ones_like(keypoints[:, 0])
        hom = torch.cat([keypoints, ones[:, None]], 1)
        rot_kpts = torch.matmul(rot_M_inv, hom.T).T[:, :2]
        if mask_illegal:
            mask = (
                (rot_kpts[:, 0] >= 0)
                & (rot_kpts[:, 0] < w)
                & (rot_kpts[:, 1] >= 0)
                & (rot_kpts[:, 1] < h)
            )
            return rot_kpts[mask], descriptors[:, mask], scores[mask]
        else:
            return rot_kpts, descriptors, scores

    def untta_rotl10(
        self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True
    ):
        rot_M_inv = (
            torch.from_numpy(cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1))
            .to(torch.float32)
            .to(self.device)
        )
        ones = torch.ones_like(keypoints[:, 0])
        hom = torch.cat([keypoints, ones[:, None]], 1)
        rot_kpts = torch.matmul(rot_M_inv, hom.T).T[:, :2]
        if mask_illegal:
            mask = (
                (rot_kpts[:, 0] >= 0)
                & (rot_kpts[:, 0] < w)
                & (rot_kpts[:, 1] >= 0)
                & (rot_kpts[:, 1] < h)
            )
            return rot_kpts[mask], descriptors[:, mask], scores[mask]
        else:
            return rot_kpts, descriptors, scores

    def untta_fliplr_rotr10(
        self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True
    ):
        rot_M_inv = (
            torch.from_numpy(cv2.getRotationMatrix2D((w / 2, h / 2), -15, 1))
            .to(torch.float32)
            .to(self.device)
        )
        ones = torch.ones_like(keypoints[:, 0])
        hom = torch.cat([keypoints, ones[:, None]], 1)
        rot_kpts = torch.matmul(rot_M_inv, hom.T).T[:, :2]
        rot_kpts[:, 0] = w - rot_kpts[:, 0] - 1.0
        if mask_illegal:
            mask = (
                (rot_kpts[:, 0] >= 0)
                & (rot_kpts[:, 0] < w)
                & (rot_kpts[:, 1] >= 0)
                & (rot_kpts[:, 1] < h)
            )
            return rot_kpts[mask], descriptors[:, mask], scores[mask]
        else:
            return rot_kpts, descriptors, scores

    def untta_fliplr_rotl10(
        self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True
    ):
        rot_M_inv = (
            torch.from_numpy(cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1))
            .to(torch.float32)
            .to(self.device)
        )
        ones = torch.ones_like(keypoints[:, 0])
        hom = torch.cat([keypoints, ones[:, None]], 1)
        rot_kpts = torch.matmul(rot_M_inv, hom.T).T[:, :2]
        rot_kpts[:, 0] = w - rot_kpts[:, 0] - 1.0
        if mask_illegal:
            mask = (
                (rot_kpts[:, 0] >= 0)
                & (rot_kpts[:, 0] < w)
                & (rot_kpts[:, 1] >= 0)
                & (rot_kpts[:, 1] < h)
            )
            return rot_kpts[mask], descriptors[:, mask], scores[mask]
        else:
            return rot_kpts, descriptors, scores


class SuperGlueMatcher:
    def __init__(self, config):
        self.config = config["SuperGlue"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._superglue_matcher = (
            SuperGlueCustomMatching(config=config, device=self.device)
            .eval()
            .to(self.device)
        )

    def prep_np_img(self, img, long_side=None):
        if long_side is not None:
            scale = long_side / max(img.shape[0], img.shape[1])
            w = int(img.shape[1] * scale)
            h = int(img.shape[0] * scale)
            img = cv2.resize(img, (w, h))
        else:
            scale = 1.0
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scale

    def frame2tensor(self, frame):
        return (torch.from_numpy(frame).float() / 255.0)[None, None].to(self.device)

    def tta_rotation_preprocess(self, img_np, angle):
        rot_M = cv2.getRotationMatrix2D(
            (img_np.shape[1] / 2, img_np.shape[0] / 2), angle, 1
        )
        rot_M_inv = cv2.getRotationMatrix2D(
            (img_np.shape[1] / 2, img_np.shape[0] / 2), -angle, 1
        )
        rot_img = self.frame2tensor(
            cv2.warpAffine(img_np, rot_M, (img_np.shape[1], img_np.shape[0]))
        )
        return rot_M, rot_img, rot_M_inv

    def tta_rotation_postprocess(self, kpts, img_np, rot_M_inv):
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

    def __call__(
        self,
        img_np0,
        img_np1,
        tta_groups=[["orig"]],
        forward_type="cross",
        input_longside=None,
    ):
        with torch.no_grad():
            img_np0, scale0 = self.prep_np_img(img_np0, input_longside)
            img_np1, scale1 = self.prep_np_img(img_np1, input_longside)

            img_ts0 = self.frame2tensor(img_np0)
            img_ts1 = self.frame2tensor(img_np1)
            images0, images1 = [], []

            tta = []
            for tta_g in tta_groups:
                tta += tta_g
            tta = list(set(tta))

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
                    ) = self.tta_rotation_preprocess(img_np0, 15)
                    (
                        rot_r10_M1,
                        img_ts1_aug,
                        rot_r10_M1_inv,
                    ) = self.tta_rotation_preprocess(img_np1, 15)
                elif tta_elem == "rot_l10":
                    (
                        rot_l10_M0,
                        img_ts0_aug,
                        rot_l10_M0_inv,
                    ) = self.tta_rotation_preprocess(img_np0, -15)
                    (
                        rot_l10_M1,
                        img_ts1_aug,
                        rot_l10_M1_inv,
                    ) = self.tta_rotation_preprocess(img_np1, -15)
                elif tta_elem == "fliplr_rotr10":
                    (
                        rot_r10_M0,
                        img_ts0_aug,
                        rot_r10_M0_inv,
                    ) = self.tta_rotation_preprocess(img_np0[:, ::-1], 15)
                    (
                        rot_r10_M1,
                        img_ts1_aug,
                        rot_r10_M1_inv,
                    ) = self.tta_rotation_preprocess(img_np1[:, ::-1], 15)
                elif tta_elem == "fliplr_rotl10":
                    (
                        rot_l10_M0,
                        img_ts0_aug,
                        rot_l10_M0_inv,
                    ) = self.tta_rotation_preprocess(img_np0[:, ::-1], -15)
                    (
                        rot_l10_M1,
                        img_ts1_aug,
                        rot_l10_M1_inv,
                    ) = self.tta_rotation_preprocess(img_np1[:, ::-1], -15)
                elif tta_elem == "eqhist":
                    img_ts0_aug = self.frame2tensor(cv2.equalizeHist(img_np0))
                    img_ts1_aug = self.frame2tensor(cv2.equalizeHist(img_np1))
                elif tta_elem == "clahe":
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_ts0_aug = self.frame2tensor(clahe.apply(img_np0))
                    img_ts1_aug = self.frame2tensor(clahe.apply(img_np1))
                else:
                    raise ValueError("Unknown TTA method.")

                images0.append(img_ts0_aug)
                images1.append(img_ts1_aug)

            if forward_type == "cross":
                pred = self._superglue_matcher.forward_cross(
                    data={"image0": torch.cat(images0), "image1": torch.cat(images1)},
                    ttas=tta,
                    tta_groups=tta_groups,
                )
            elif forward_type == "flat":
                pred = self._superglue_matcher.forward_flat(
                    data={"image0": torch.cat(images0), "image1": torch.cat(images1)},
                    ttas=tta,
                    tta_groups=tta_groups,
                )
            else:
                raise RuntimeError(f"Unknown forward_type {forward_type}")

            matching_key_points0, matching_key_points1, mconf = [], [], []
            for group_pred in pred:
                pred_aug = {
                    k: v[0].detach().cpu().numpy().squeeze()
                    for k, v in group_pred.items()
                }
                kpts0, kpts1 = pred_aug["keypoints0"], pred_aug["keypoints1"]
                matches, conf = pred_aug["matches0"], pred_aug["matching_scores0"]

                if self.config["match_threshold"] is None:
                    valid = matches > -1
                else:
                    valid = (matches > -1) & (conf >= self.config["match_threshold"])
                matching_key_points0.append(kpts0[valid])
                matching_key_points1.append(kpts1[matches[valid]])
                mconf.append(conf[valid])

            cat_matching_key_points0 = np.concatenate(matching_key_points0)
            cat_matching_key_points1 = np.concatenate(matching_key_points1)
            mask0 = (
                (cat_matching_key_points0[:, 0] >= 0)
                & (cat_matching_key_points0[:, 0] < img_np0.shape[1])
                & (cat_matching_key_points0[:, 1] >= 0)
                & (cat_matching_key_points0[:, 1] < img_np0.shape[0])
            )
            mask1 = (
                (cat_matching_key_points1[:, 0] >= 0)
                & (cat_matching_key_points1[:, 0] < img_np1.shape[1])
                & (cat_matching_key_points1[:, 1] >= 0)
                & (cat_matching_key_points1[:, 1] < img_np1.shape[0])
            )
            return (
                cat_matching_key_points0[mask0 & mask1] / scale0,
                cat_matching_key_points1[mask0 & mask1] / scale1,
            )
