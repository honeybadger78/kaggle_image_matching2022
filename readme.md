🔗 **Competition Link**: [Image Matching Challenge 2022](https://www.kaggle.com/competitions/image-matching-challenge-2022/overview)

### 🏆 Result: **🥈50/642🥈**
### 📝 Brief Summary

- **Ensemble of LoFTR, SuperGlue**: The framework employs an ensemble of LoFTR and SuperGlue with the resize calibration and appropriate number of keypoints for each model.  

### 🔄 Data Preprocessing

- **Resize Image**: Images are Resized by try and error.
- **TTA Transformations**: Augmentations like the original, horizontal flip, and rotation are applied to the images before matching.

### 🚀 Model Training Techniques

- **Usage of the Partial Function**: The `functools.partial` function is used to fix arguments required by each matching algorithm, ensuring flexibility and modularity.
- **Diverse Matching Algorithms**: Algorithms like LoFTR and SuperGlue are employed to match feature points between images.

### 🎯 Ensembling Approach

1. **Keypoint Aggregation**: Extract keypoints from all the models.
2. **Concatenation**: Merge the keypoints from different models.
3. **RANSAC Application**: Apply the RANSAC algorithm to the concatenated keypoints for robust matching.

**Note**: We also experimented with concatenating keypoints after multiple RANSAC runs. However, this approach was slower and resulted in a lower score.

### 🔑 Keypoints Number

The optimal selection of the number of keypoints is crucial for maximizing the performance of the ensemble. The default values for the number of keypoints can vary significantly among models, which might be why ensembles may not work effectively for some.

While not all models offer a strict parameter for the number of final keypoints, here's how to adjust them for better performance:
1. **LoFTR**: Adjust `max_keypoints` in the config from 1024 to 2048.
2. **SuperGlue**: Modify `max_keypoints` in the config from 1024 to 2048.



---
