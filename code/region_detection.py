import numpy as np
from skimage import feature
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, rectangle, diamond, disk, cube, octahedron, ball, octagon, star
from skimage.measure import label, regionprops,regionprops_table,inertia_tensor_eigvals,inertia_tensor,moments_central,moments_normalized,moments_hu,perimeter

class pos_neg_detection:
    def __init__(self):
        pass

    def identify_pos_neg_region(self, fits_map, pos_gauss = 100, neg_gauss= -100):
        pos_map = np.zeros(fits_map.data.shape)
        neg_map = np.zeros(fits_map.data.shape)

        np.warnings.filterwarnings('ignore')
        result_pos = np.where(fits_map.data >= pos_gauss)
        result_neg = np.where(fits_map.data <= neg_gauss)

        pos_map[result_pos[0],result_pos[1]] = 1
        neg_map[result_neg[0],result_neg[1]] = 1

        return pos_map, neg_map

    def edge_detection(self, binary_map):
        sig = 1

        # sigma: float, optional
        # Standard deviation of the Gaussian filter.

        # low_threshold: float, optional
        # Lower bound for hysteresis thresholding (linking edges). If None, low_threshold is set to 10% of dtype’s max.

        # high_threshold: float, optional
        # Upper bound for hysteresis thresholding (linking edges). If None, high_threshold is set to 20% of dtype’s max.

        # mask: array, dtype=bool, optional
        # Mask to limit the application of Canny to a certain area.

        edges = feature.canny(binary_map, sigma=sig)

        return edges

    def mask_img(self, img):
        return np.ma.masked_where(img.astype(float) == 0, img.astype(float))

    def mask_pil(self, img):
        return np.ma.masked_where(img.astype(bool).astype(float) == 0, img.astype(bool).astype(float))

    def buff_edge(self, edges, size=4):
        selem = square(size)

        dilated_edges = dilation(edges, selem)

        return dilated_edges



