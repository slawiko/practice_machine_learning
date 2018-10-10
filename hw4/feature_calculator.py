"""
    - Type 0      - Type 1      - Type 2       - Type 3
    AAAAAAAA      AAAABBBB      AAAACCCC       BBBBAAAACCCC
    AAAAAAAA      AAAABBBB      AAAACCCC       BBBBAAAACCCC
    BBBBBBBB      AAAABBBB      BBBBDDDD       BBBBAAAACCCC
    BBBBBBBB      AAAABBBB      BBBBDDDD       BBBBAAAACCCC

     A - B         A - B      A + D - B - C     A - B - C
"""

FEATURE_TO_W_STEP_MAP = {
    0: 1,
    1: 2,
    2: 2,
    3: 3
}

FEATURE_TO_H_STEP_MAP = {
    0: 2,
    1: 1,
    2: 2,
    3: 1
}

FEATURE_TO_START_MAP = {
    0: 2,
    1: 2,
    2: 2,
    3: 3
}

def calculate_0_feature(img, x, y, width, height):
    if height % FEATURE_TO_H_STEP_MAP[0] > 0:
        return None
    h = height // FEATURE_TO_H_STEP_MAP[0]

    if x == 0 and y == 0:
        A = img[h - 1, width - 1]
        B = img[height - 1, width - 1] - img[h - 1, width - 1]
    elif x == 0:
        A = img[h - 1, y + width - 1] - img[h - 1, y - 1]
        B = img[height - 1, y + width - 1] - img[h - 1, y + width - 1] - img[height - 1, y - 1] + img[h - 1, y - 1]
    elif y == 0:
        A = img[x + h - 1, width - 1] - img[x - 1, width - 1]
        B = img[x + height - 1, width - 1] - img[x + h - 1, width - 1]
    else:
        A = img[x + h - 1, y + width - 1] - img[x - 1, y + width - 1] - img[x + h - 1, y - 1] + img[x - 1, y - 1]
        B = img[x + height - 1, y + width - 1] - img[x + h - 1, y + width - 1] - img[x + height - 1, y - 1] + img[x + h - 1, y - 1]
    return A - B

def calculate_1_feature(img, x, y, width, height):
    if width % FEATURE_TO_W_STEP_MAP[1] > 0:
        return None
    w = width // FEATURE_TO_W_STEP_MAP[1]
    
    if x == 0 and y == 0:
        A = img[height - 1, w - 1]
        B = img[height - 1, width - 1] - img[height - 1, w - 1]
    elif x == 0:
        A = img[height - 1, y + w - 1] - img[height - 1, y - 1]
        B = img[height - 1, y + width - 1] - img[height - 1, y + w - 1]
    elif y == 0:
        A = img[x + height - 1, w - 1] - img[x - 1, w - 1]
        B = img[x + height - 1, width - 1] - img[x + height - 1, w - 1] - img[x - 1, width - 1] + img[x - 1, w - 1]
    else:
        A = img[x + height - 1, y + w - 1] - img[x + height - 1, y - 1] - img[x - 1, y + w - 1] + img[x - 1, y - 1]
        B = img[x + height - 1, y + width - 1] - img[x + height - 1, y + w - 1] - img[x - 1, y + width - 1] + img[x - 1, y + w - 1]
    return A - B

def calculate_2_feature(img, x, y, width, height):
    if width % FEATURE_TO_W_STEP_MAP[2] > 0 or height % FEATURE_TO_H_STEP_MAP[2] > 0:
        return None
    w = width // FEATURE_TO_W_STEP_MAP[2]
    h = height // FEATURE_TO_H_STEP_MAP[2]

    if x == 0 and y == 0:
        A = img[h - 1, w - 1]
        B = img[height - 1, w - 1] - img[h - 1, w - 1]
        C = img[h - 1, width - 1] - img[h - 1, w - 1]
        D = img[height - 1, width - 1] - img[h - 1, width - 1] - img[height - 1, w - 1] + img[h - 1, w - 1]
    elif x == 0:
        A = img[h - 1, y + w - 1] - img[h - 1, y - 1]
        B = img[height - 1, y + w - 1] - img[height - 1, y - 1] - img[h - 1, y + w - 1] + img[h - 1, y - 1]
        C = img[h - 1, y + width - 1] - img[h - 1, y + w - 1]
        D = img[height - 1, y + width - 1] - img[height - 1, y + w - 1] - img[h - 1, y + width - 1] + img[h - 1, y + w - 1]
    elif y == 0:
        A = img[x + h - 1, w - 1] - img[x - 1, w - 1]
        B = img[x + height - 1, w - 1] - img[x + h - 1, w - 1]
        C = img[x + h - 1, width - 1] - img[x + h - 1, w - 1] - img[x - 1, width - 1] + img[x - 1, w - 1]
        D = img[x + height - 1, width - 1] - img[x + height - 1, w - 1] - img[x + h - 1, width - 1] + img[x + h - 1, w - 1]
    else:
        A = img[x + h - 1, y + w - 1] - img[x + h - 1, y - 1] - img[x - 1, y + w - 1] + img[x - 1, y - 1]
        B = img[x + height - 1, y + w - 1] - img[x + height - 1, y - 1] - img[x + h - 1, y + w - 1] + img[x + h - 1, y - 1]
        C = img[x + h - 1, y + width - 1] - img[x + h - 1, y + w - 1] - img[x - 1, y + width - 1] + img[x - 1, y + w - 1]
        D = img[x + height - 1, y + width - 1] - img[x + height - 1, y + w - 1] - img[x + h - 1, y + width - 1] + img[x + h - 1, y + w - 1]
    return A + D - B - C

def calculate_3_feature(img, x, y, width, height):
    if width % FEATURE_TO_W_STEP_MAP[3] > 0:
        return None
    w = width // FEATURE_TO_W_STEP_MAP[3]

    if x == 0 and y == 0:
        A = img[height - 1, w + w - 1] - img[height - 1, w - 1]
        B = img[height - 1, w - 1]
        C = img[height - 1, width - 1] - img[height - 1, w + w - 1]
    elif x == 0:
        A = img[height - 1, y + w + w - 1] - img[height - 1, y + w - 1]
        B = img[height - 1, y + w - 1] - img[height - 1, y - 1]
        C = img[height - 1, y + width - 1] - img[height - 1, y + w + w - 1]
    elif y == 0:
        A = img[x + height - 1, w + w - 1] - img[x + height - 1, w - 1] - img[x - 1, w + w - 1] + img[x - 1, w - 1]
        B = img[x + height - 1, w - 1] - img[x - 1, w - 1]
        C = img[x + height - 1, width - 1] - img[x + height - 1, w + w - 1] - img[x - 1, width - 1] + img[x - 1, w + w - 1]
    else:
        A = img[x + height - 1, y + w + w - 1] - img[x + height - 1, y + w - 1] - img[x - 1, y + w + w - 1] + img[x - 1, y + w - 1]
        B = img[x + height - 1, y + w - 1] - img[x + height - 1, y - 1] - img[x - 1, y + w - 1] + img[x - 1, y - 1]
        C = img[x + height - 1, y + width - 1] - img[x + height - 1, y + w + w - 1] - img[x - 1, y + width - 1] + img[x - 1, y + w + w - 1]
    return A - B - C

FEATURE_TO_FUNC_MAP = {
    0: calculate_0_feature,
    1: calculate_1_feature,
    2: calculate_2_feature,
    3: calculate_3_feature
}

def calculate_feature(img, x, y, width, height, t, polarity):
  return polarity * FEATURE_TO_FUNC_MAP[t](img, x, y, width, height)
