import cv2
import time
import numpy as np
from utils.matrix import FlattenMatrix

class SolutionHolder:
    def __init__(self):
        self.F_dict = {}

    @staticmethod
    def solve_keypoints(matching_key_points0, matching_key_points1, ret_inliers=False):
        findmat_st = time.time()

        if len(matching_key_points0) > 7:
            F, inliers = cv2.findFundamentalMat(
                matching_key_points0, matching_key_points1, cv2.USAC_MAGSAC, 0.2, 0.9999, 250000
            )
            inliers = inliers.ravel().astype(bool)
            assert F.shape == (3, 3), "Malformed F?"
        else:
            F = np.zeros((3, 3))

        print(f"  - Ransac time: {time.time() - findmat_st:.4f} s")

        return (F, inliers) if ret_inliers else F

    def add_solution(self, sample_id, matching_key_points0, matching_key_points1):
        self.F_dict[sample_id] = self.solve_keypoints(matching_key_points0, matching_key_points1)

    def dump(self, output_file):
        with open(output_file, "w") as f:
            f.write("sample_id,fundamental_matrix\n")
            for sample_id, F in self.F_dict.items():
                f.write(f"{sample_id},{FlattenMatrix(F)}\n")