import os
import time
from typing import Union, Tuple, List, Optional, Dict
import cv2
import numpy as np
import matplotlib.pyplot as plt


MIN_AREA = 80
MIN_WIDTH = 2
MIN_HEIGHT = 8
MIN_RATIO = 0.25
MAX_RATIO = 1.0
MAX_DIAG_MULTIPLIER = 5
MAX_ANGLE_DIFF = 12.0
MAX_AREA_DIFF = 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3
PLATE_WIDTH_PADDING = 1.3
PLATE_HEIGHT_PADDING = 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10


def resize(
    img: np.ndarray,
    size: Union[Tuple[int, int], int],
    is_upsample: Optional[bool] = False,
) -> np.ndarray:
    height, width, channel = img.shape

    if isinstance(size, int):
        ratio = height / width if height > width else width / height
        dsize = (
            (int(size * ratio), size) if width > height else (size, int(size * ratio))
        )

    elif isinstance(size, tuple):
        if len(size) != 2:
            raise ValueError
        dsize = size

    return (
        cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_AREA)
        if not is_upsample
        else cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    )


def get_blurred_img(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Morphological Transformation (https://dsbook.tistory.com/203)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)


def get_thresh_img(img: np.ndarray, mode: Optional[Union[int, str]] = 1) -> np.ndarray:
    if mode not in [0, 1, "normal", "adaptive"]:
        raise ValueError

    if mode in ["normal", 0]:
        # mode 1. normal threshold
        # ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ret, binary = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
        img_thresh = cv2.bitwise_not(binary)

    elif mode in ["adaptive", 1]:
        # mode 2. adaptive threshold
        img_thresh = cv2.adaptiveThreshold(
            img,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=9,
        )

    return img_thresh


def get_black_and_white_img(img: np.ndarray, output_inverse: bool = True) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 7)
    img = cv2.Laplacian(img, cv2.CV_8U, 5)
    _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_not(img) if output_inverse else img


def find_roi(img_thresh: np.ndarray) -> List[Dict[str, int]]:
    contours, _ = cv2.findContours(
        img_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = img_thresh.shape[:2]
    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contours_dict.append(
            {
                "contour": contour,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + (w / 2),
                "cy": y + (h / 2),
            }
        )

    possible_contours = []
    cnt = 0

    for d in contours_dict:
        area = d["w"] * d["h"]
        ratio = d["w"] / d["h"]

        if (
            area > MIN_AREA
            and d["w"] > MIN_WIDTH
            and d["h"] > MIN_HEIGHT
            and MIN_RATIO < ratio < MAX_RATIO
        ):
            d["idx"] = cnt
            cnt += 1
            possible_contours.append(d)

    def find_chars(contour_list):
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1["idx"] == d2["idx"]:
                    continue

                dx = abs(d1["cx"] - d2["cx"])
                dy = abs(d1["cy"] - d2["cy"])

                diagonal_length1 = np.sqrt(d1["w"] ** 2 + d1["h"] ** 2)

                distance = np.linalg.norm(
                    np.array([d1["cx"], d1["cy"]]) - np.array([d2["cx"], d2["cy"]])
                )
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1["w"] * d1["h"] - d2["w"] * d2["h"]) / (
                    d1["w"] * d1["h"]
                )
                width_diff = abs(d1["w"] - d2["w"]) / d1["w"]
                height_diff = abs(d1["h"] - d2["h"]) / d1["h"]

                if (
                    distance < diagonal_length1 * MAX_DIAG_MULTIPLIER
                    and angle_diff < MAX_ANGLE_DIFF
                    and area_diff < MAX_AREA_DIFF
                    and width_diff < MAX_WIDTH_DIFF
                    and height_diff < MAX_HEIGHT_DIFF
                ):
                    matched_contours_idx.append(d2["idx"])

            matched_contours_idx.append(d1["idx"])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4["idx"] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4["idx"])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    result_idx = find_chars(possible_contours)
    matched_result = []

    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x["cx"])

        plate_cx = (sorted_chars[0]["cx"] + sorted_chars[-1]["cx"]) / 2
        plate_cy = (sorted_chars[0]["cy"] + sorted_chars[-1]["cy"]) / 2

        plate_width = (
            sorted_chars[-1]["x"] + sorted_chars[-1]["w"] - sorted_chars[0]["x"]
        ) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d["h"]

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        plate_infos.append(
            {
                "x": int(plate_cx - plate_width / 2),
                "y": int(plate_cy - plate_height / 2),
                "w": int(plate_width),
                "h": int(plate_height),
            }
        )

    return plate_infos


def convert_contour(
    contours: List[Dict],
    imgsz: Tuple[int, int],
    target_imgsz: Tuple[int, int],
) -> List[Dict[str, int]]:

    ratio_height, ratio_width = (
        target_imgsz[1] / imgsz[1]
        if target_imgsz[1] > imgsz[1]
        else imgsz[1] / target_imgsz[1],
        target_imgsz[0] / imgsz[0]
        if target_imgsz[0] > imgsz[0]
        else imgsz[0] / target_imgsz[0],
    )

    for contour in contours:
        contour["x"] = int(contour["x"] * ratio_width)
        contour["y"] = int(contour["y"] * ratio_height)
        contour["w"] = int(contour["w"] * ratio_width)
        contour["h"] = int(contour["h"] * ratio_height)

    return contours


DEBUG_OPT: bool = True

if __name__ == "__main__":
    img_dir = "../images"
    # img_dir = "../regions"
    img_list = os.listdir(img_dir)

    for fname in img_list:
        if fname[0] == ".":
            continue

        start = time.time()
        img_ori = cv2.imread(f"{img_dir}/{fname}")
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        height_ori, width_ori = img_ori.shape[:2]
        img = resize(img_ori, 640)
        height, width = img.shape[:2]
        # img1 = get_blurred_img(img)
        # img1 = get_thresh_img(img1, mode=1)
        # img1 = get_black_and_white_img(img, False)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)

        threshold1 = 100
        threshold2 = 500
        # img1 = cv2.Canny(blurred, threshold1, threshold2)
        img1 = cv2.Canny(img_gray, threshold1, threshold2)

        if DEBUG_OPT:
            plt.imshow(img1)
            plt.show()

        contours = find_roi(img1)
        contours = convert_contour(
            contours,
            imgsz=(width, height),
            target_imgsz=(width_ori, height_ori),
        )

        print(f"processing time for {fname}: {time.time() - start}, imgsz: {width_ori}*{height_ori}")

        if contours:
            for contour in contours:
                top_left: Tuple[int, int] = (contour["x"], contour["y"])
                bottom_right: Tuple[int, int] = (
                    contour["x"] + contour["w"],
                    contour["y"] + contour["h"],
                )
                img_ori = cv2.rectangle(
                    img_ori, top_left, bottom_right, (255, 0, 0), 10
                )
                break  # TODO: add routine for ROI ratio comparison and exception
        fig = plt.figure()
        plt.imshow(img_ori)
        # plt.show()
        fig.savefig(f"../outputs/{fname}")
