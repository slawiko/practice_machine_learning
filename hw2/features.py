import numpy as np

def sum_feature(a):
    """Sum of intensities of all pixels: digits with more used pixels will have higher value."""
    return np.sum(a)

def right_sum_feature(a):
    """Sum of intensities of pixels in right half."""
    a = np.reshape(a, (28, 28))
    return np.sum(a[:, 14:])

def left_sum_feature(a):
    """Sum of intensities of pixels in left half."""
    a = np.reshape(a, (28, 28))
    return np.sum(a[:, :14])

def top_sum_feature(a):
    """Sum of intensities of pixels in top half."""
    a = np.reshape(a, (28, 28))
    return np.sum(a[14:, :])

def bottom_sum_feature(a):
    """Sum of intensities of pixels in bottom half."""
    a = np.reshape(a, (28, 28))
    return np.sum(a[:14, :])

def right_top_sum_feature(a):
    """Sum of intensities of pixels in right top quarter."""
    a = np.reshape(a, (28, 28))
    return np.sum(a[14:, 14:])

def right_bottom_sum_feature(a):
    """Sum of intensities of pixels in right bottom quarter."""
    a = np.reshape(a, (28, 28))
    return np.sum(a[:14, 14:])

def left_top_sum_feature(a):
    """Sum of intensities of pixels in left top quarter."""
    a = np.reshape(a, (28, 28))
    return np.sum(a[14:, :14])

def left_bottom_sum_feature(a):
    """Sum of intensities of pixels in left bottom quarter."""
    a = np.reshape(a, (28, 28))
    return np.sum(a[:14, :14])

def intersection_count(a):
    counter = 0
    started = False
    for e in a:
        if started:
            if e < 80:
                counter += 1
                started = False
        else:
            if e > 80:
                started = True
    return counter

def center_sum_feature(a):
    """Sum of intensities of pixels in center (4x4) square."""
    a = np.reshape(a, (28, 28))
    return np.sum(a[12:16, 12:16])

def empty_feature(a):
    return 0

def digital_clock_feature(a):
    """Intersections of digit with digital_clock_pattern count.
        digital clock pattern is something like:
        ______
        |    |
        |    |
        ------
        |    |
        |    |
        ------
    """
    a = np.reshape(a, (28, 28))
    col1 = a[:, 9]
    col2 = a[:, 17]
    row1 = a[4, :]
    row2 = a[10, :]
    row3 = a[16, :]

    return intersection_count(col1) + intersection_count(col2) + intersection_count(row1) + intersection_count(row2) + intersection_count(row3)

def bottom_padding_feature(a):
    """Digit padding from bottom."""
    a = np.reshape(a, (28, 28))
    for i in range(27, 0, -1):
        if len(np.where(a[i, :] > 80)[0]) > 0:
            return 28 - i
    return 0

def top_padding_feature(a):
    """Digit padding from top."""
    a = np.reshape(a, (28, 28))
    for i in range(0, 28):
        if len(np.where(a[i, :] > 80)[0]) > 0:
            return i
    return 0

def right_padding_feature(a):
    """Digit padding from right."""
    a = np.reshape(a, (28, 28))
    for i in range(27, 0, -1):
        if len(np.where(a[:, i] > 80)[0]) > 0:
            return 28 - i
    return 0

def left_padding_feature(a):
    """Digit padding from left."""
    a = np.reshape(a, (28, 28))
    for i in range(0, 28):
        if len(np.where(a[:, i] > 80)[0]) > 0:
            return i
    return 0

FEATURES = {
    (0, 1) : (empty_feature, digital_clock_feature), # 98.6
    (0, 2) : (top_sum_feature, left_sum_feature), # 85.3
    (0, 3) : (top_sum_feature, right_sum_feature), # 78.8
    (0, 4) : (empty_feature, digital_clock_feature), # 86.3
    (0, 5) : (empty_feature, digital_clock_feature), # 77.3
    (0, 6) : (top_sum_feature, left_sum_feature), # 79.8
    (0, 7) : (empty_feature, digital_clock_feature), # 90.0
    (0, 8) : (empty_feature, center_sum_feature), # 94.4
    (0, 9) : (empty_feature, digital_clock_feature), # 82.3
    (1, 2) : (empty_feature, digital_clock_feature), # 94.7
    (1, 3) : (empty_feature, digital_clock_feature), # 92.9
    (1, 4) : (empty_feature, digital_clock_feature), # 85.5
    (1, 5) : (empty_feature, digital_clock_feature), # 89.5
    (1, 6) : (empty_feature, digital_clock_feature), # 96.2
    (1, 7) : (empty_feature, digital_clock_feature), # 75.3
    (1, 8) : (empty_feature, digital_clock_feature), # 94.7
    (1, 9) : (empty_feature, digital_clock_feature), # 84.1
    (2, 3) : (top_sum_feature, right_sum_feature), # 83.4
    (2, 4) : (digital_clock_feature, right_bottom_sum_feature), # 76.3
    (2, 5) : (top_sum_feature, left_sum_feature), # 80.9
    (2, 6) : (empty_feature, left_padding_feature), # 75.8
    (2, 7) : (empty_feature, digital_clock_feature), # 76.9
    (2, 8) : (top_sum_feature, left_sum_feature), # 80.3
    (2, 9) : (top_sum_feature, right_sum_feature), # 82.6
    (3, 4) : (right_bottom_sum_feature, left_top_sum_feature), # 76.0
    (3, 5) : (left_sum_feature, sum_feature), # 75.9
    (3, 6) : (top_sum_feature, right_sum_feature), # 91.7
    (3, 7) : (top_padding_feature, left_padding_feature), # 93.0
    (3, 8) : (right_padding_feature, left_padding_feature), # 83.8
    (3, 9) : (top_padding_feature, left_padding_feature), # 90.0
    (4, 5) : (bottom_sum_feature, right_sum_feature), # 75.7
    (4, 6) : (empty_feature, digital_clock_feature), # 76.7
    (4, 7) : (top_sum_feature, right_sum_feature), # 81.8
    (4, 8) : (center_sum_feature, bottom_sum_feature), # 78.3
    (4, 9) : (top_padding_feature, left_padding_feature), # 82.9
    (5, 6) : (top_sum_feature, left_sum_feature), # 75.4
    (5, 7) : (top_sum_feature, right_sum_feature), # 79.8
    (5, 8) : (center_sum_feature, right_padding_feature), # 75.3
    (5, 9) : (empty_feature, right_padding_feature), # 79.5
    (6, 7) : (empty_feature, digital_clock_feature), # 81.8
    (6, 8) : (top_sum_feature, right_sum_feature), # 83.7
    (6, 9) : (top_sum_feature, right_sum_feature), # 90.8
    (7, 8) : (empty_feature, digital_clock_feature), # 78.1
    (7, 9) : (digital_clock_feature, center_sum_feature), # 83.5
    (8, 9) : (digital_clock_feature, center_sum_feature), # 79.3
}
