import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray

def hog_feature(img_rgb_64: np.ndarray) -> np.ndarray:
    gray = rgb2gray(img_rgb_64)  # [64,64]
    feat = hog(
        gray,
        orientations=6,
        pixels_per_cell=(4,4),
        cells_per_block=(2,2),
        block_norm="L2-Hys",
        feature_vector=True
    )
    return feat.astype(np.float32)

def hog_batch(X_patches: np.ndarray) -> np.ndarray:
    feats = [hog_feature(x) for x in X_patches]
    return np.vstack(feats)
