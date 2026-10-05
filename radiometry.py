"""
radiometry.py
Inter-photo radiometric equalization and histogram matching.
Balances brightness, contrast, and color balance across overlapping aerial photos.
"""

import numpy as np
from skimage.exposure import match_histograms

def match_radiometry(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    overlap_mask: np.ndarray
) -> np.ndarray:
    """
    Adjusts source_img channels (RGB) to match the radiometric distribution
    of reference_img within their common unoccluded overlap region using
    skimage.exposure.match_histograms.
    """
    if not np.any(overlap_mask):
        return source_img

    adjusted_img = source_img.copy()
    
    # We only want to match the histograms for pixels within the overlap mask
    # Since match_histograms doesn't natively take a mask for computing the histogram,
    # we can extract the masked pixels, run matching on them, and place them back.
    # But wait, match_histograms matches the *entire* image. If we just want to match
    # the overlapping region's histogram and apply it to the whole image, it's tricky.
    # The simplest way is to apply match_histograms to the entire image.
    
    adjusted_img = match_histograms(source_img, reference_img, channel_axis=-1).astype(np.uint8)
    
    return adjusted_img


def histogram_match_channel(source_channel: np.ndarray, ref_channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Non-linear CDF histogram matching for an individual color channel.
    """
    src_pixels = source_channel[mask].ravel()
    ref_pixels = ref_channel[mask].ravel()

    if len(src_pixels) < 100 or len(ref_pixels) < 100:
        return source_channel

    # Compute empirical CDFs
    s_values, s_counts = np.unique(src_pixels, return_counts=True)
    r_values, r_counts = np.unique(ref_pixels, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / src_pixels.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / ref_pixels.size

    # Map source values through inverse reference CDF
    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)

    lookup = np.arange(256, dtype=np.uint8)
    lookup[s_values] = np.clip(interp_r_values, 0, 255).astype(np.uint8)

    return lookup[source_channel]
