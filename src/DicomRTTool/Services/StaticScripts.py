import numpy as np
import cv2


def add_to_mask(mask, z_value, r_value, c_value, mask_value=1):
    mask[int(np.floor(z_value)), int(np.floor(r_value)), int(np.floor(c_value))] = mask_value
    mask[int(np.floor(z_value)), int(np.ceil(r_value)), int(np.floor(c_value))] = mask_value
    mask[int(np.floor(z_value)), int(np.floor(r_value)), int(np.ceil(c_value))] = mask_value
    mask[int(np.floor(z_value)), int(np.ceil(r_value)), int(np.ceil(c_value))] = mask_value
    mask[int(np.ceil(z_value)), int(np.floor(r_value)), int(np.floor(c_value))] = mask_value
    mask[int(np.ceil(z_value)), int(np.ceil(r_value)), int(np.floor(c_value))] = mask_value
    mask[int(np.ceil(z_value)), int(np.floor(r_value)), int(np.ceil(c_value))] = mask_value
    mask[int(np.ceil(z_value)), int(np.ceil(r_value)), int(np.ceil(c_value))] = mask_value
    return None


def poly2mask(vertex_row_coords: np.array, vertex_col_coords: np.array,
              shape: tuple) -> np.array:
    """[converts polygon coordinates to filled boolean mask]

    Args:
        vertex_row_coords (np.array): [row image coordinates]
        vertex_col_coords (np.array): [column image coordinates]
        shape (tuple): [image dimensions]

    Returns:
        [np.array]: [filled boolean polygon mask with vertices at
                     (row, col) coordinates]
    """
    xy_coords = np.array([vertex_col_coords, vertex_row_coords])
    coords = np.expand_dims(xy_coords.T, 0)
    mask = np.zeros(shape)
    cv2.fillPoly(mask, coords, 1)
    return np.array(mask, dtype=bool)


if __name__ == '__main__':
    pass
