import cv2
import time
import numpy as np
from functools import partial
from net.loftr import LoFTRMatcher
from net.superglue import SuperGlueMatcher

def match_images(config, sample_id, batch_id, image_1_id, image_2_id, ret_ims=False):
    img_np1 = cv2.imread(f'{config["Directory"]["image"]}/{batch_id}/{image_1_id}.png')
    img_np2 = cv2.imread(f'{config["Directory"]["image"]}/{batch_id}/{image_2_id}.png')

    loftr_matcher = LoFTRMatcher(config["LoFTR"])
    superglue_matcher = SuperGlueMatcher(config["SuperGlue"])
    matchers_cfg = config["Matcher"]
    max_name_len = 0
    matching_key_points0, matching_key_points1, runtime_str, kp_count_str = [], [], [], []

    for m_cfg in matchers_cfg:
        max_name_len = max(len(m_cfg["name"]), max_name_len)
        if m_cfg["name"] == "loftr":
            matcher_fn = partial(loftr_matcher, tta=m_cfg["tta"])
        elif m_cfg["name"] == "superglue":
            matcher_fn = partial(
                superglue_matcher,
                tta_groups=m_cfg["tta_groups"],
                forward_type=m_cfg["forward_type"],
                input_longside=m_cfg["input_longside"],
            )
        else:
            continue

        start_time = time.time()
        m_matching_key_points0, m_matching_key_points1 = matcher_fn(img_np1, img_np2)
        end_time = time.time()

        matching_key_points0.append(m_matching_key_points0)
        matching_key_points1.append(m_matching_key_points1)
        runtime_str.append(
            f'{m_cfg["name"].ljust(max_name_len)}: {end_time - start_time:04f}s'
        )
        kp_count_str.append(f'{m_cfg["name"]}={len(m_matching_key_points0)}')

    matching_key_points0 = np.concatenate(matching_key_points0)
    matching_key_points1 = np.concatenate(matching_key_points1)

    if ret_ims:
        return matching_key_points0, matching_key_points1, img_np1, img_np2
    else:
        return matching_key_points0, matching_key_points1