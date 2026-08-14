"""
omr_scanner.py
--------------
OMR sheet er chobi theke answer ber korar shob logic ekhane.

Kaj 2 ta step e hoy:
  1. detect_and_warp() -> chobi te sheet er 4 kona khuje ber kore, seta
     shoja kore ("perspective warp") ekta fixed size canvas e bosay.
  2. read_answers() -> calibration data use kore proti question er
     4 ta bubble (A/B/C/D) er jaygay pixel darkness check kore
     kon ta bhora ache seta ber kore.

Calibration = mentor ekbar "Calibration" page e blank OMR sheet upload
kore 4 ta point click korbe:
   1) Question 1 - option A এর কেন্দ্র
   2) Question 1 - option D এর কেন্দ্র
   3) Question 25 - option A এর কেন্দ্র  (same block, last row)
   4) Question 26 - option A এর কেন্দ্র  (next block, first row)
Eita theke pura 100 ta question er shob bubble position hisab kora hoy,
karon printed sheet e shob shomoy uniform grid thake.
"""

import cv2
import numpy as np

WARP_WIDTH = 1200
WARP_HEIGHT = 1600
TOTAL_QUESTIONS = 100
QUESTIONS_PER_BLOCK = 25
NUM_BLOCKS = 4
OPTIONS = ["A", "B", "C", "D"]
BUBBLE_SAMPLE_RADIUS = 12  # pixel, warped image er upor nirbhor kore proyojon hole change korte hobe


def _order_points(pts):
    """4 ta point ke top-left, top-right, bottom-right, bottom-left order e sajay."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_and_warp(image_bgr):
    """
    Chobi theke shob theke boro 4-konar contour (sheet er border) khuje
    seta ke shoja kore warp kore fixed size e ferot dey.
    Jodi valo contour na pawa jay, image ke shudhu resize kore ferot dey
    (user ke tokhon nijer hate straight chobi tulte bola hobe).
    """
    orig = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    sheet_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            sheet_contour = approx
            break

    if sheet_contour is None:
        # fallback: just resize, warp korte parlo na
        resized = cv2.resize(orig, (WARP_WIDTH, WARP_HEIGHT))
        return resized, False

    pts = sheet_contour.reshape(4, 2).astype("float32")
    rect = _order_points(pts)
    dst = np.array(
        [[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (WARP_WIDTH, WARP_HEIGHT))
    return warped, True


def build_grid(calibration):
    """
    calibration dict e 4 ta clicked point thake (x,y):
      q1_a, q1_d, q25_a, q26_a
    Eita theke প্রতিটা question(1-100) er 4 bubble center hisab kore
    ekta dict banay: { question_no: {"A": (x,y), "B": (x,y), ...} }
    """
    q1_a = np.array(calibration["q1_a"], dtype=float)
    q1_d = np.array(calibration["q1_d"], dtype=float)
    q25_a = np.array(calibration["q25_a"], dtype=float)
    q26_a = np.array(calibration["q26_a"], dtype=float)

    option_step = (q1_d - q1_a) / (len(OPTIONS) - 1)  # A->D dorotto, 3 bhag e ভাগ
    row_step = (q25_a - q1_a) / (QUESTIONS_PER_BLOCK - 1)  # row to row dorotto
    block_step = q26_a - q1_a  # ekta block theke porerta te x/y shift

    grid = {}
    q_no = 1
    for block in range(NUM_BLOCKS):
        block_origin = q1_a + block * block_step
        for row in range(QUESTIONS_PER_BLOCK):
            row_origin = block_origin + row * row_step
            options = {}
            for i, opt in enumerate(OPTIONS):
                center = row_origin + i * option_step
                options[opt] = (int(round(center[0])), int(round(center[1])))
            grid[q_no] = options
            q_no += 1
    return grid


def _bubble_darkness(gray_img, center, radius=BUBBLE_SAMPLE_RADIUS):
    x, y = center
    h, w = gray_img.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius)
    y0, y1 = max(0, y - radius), min(h, y + radius)
    if x1 <= x0 or y1 <= y0:
        return 255.0
    patch = gray_img[y0:y1, x0:x1]
    return float(np.mean(patch))


def read_answers(warped_bgr, grid, dark_threshold=150, min_gap=15):
    """
    Proti question er jonno 4 ta bubble er darkness compare kore
    kon ta মার্ক kora ache seta বের kore.

    dark_threshold: eर niche gore mane bubble ta bhora (kalo)
    min_gap: sobcheye kalo r porerta kalor moddhe eto pathok na thakle
             seta "unclear / multiple marked" dhore newa hoy

    Return: dict {question_no: 'A'/'B'/'C'/'D'/None}
            None mane blank ba bujha jacche na
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    # adaptive threshold diye bubble er kalo dag ke aro clear kora
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    answers = {}
    for q_no, options in grid.items():
        darkness = {opt: _bubble_darkness(gray, center) for opt, center in options.items()}
        sorted_opts = sorted(darkness.items(), key=lambda kv: kv[1])
        darkest_opt, darkest_val = sorted_opts[0]
        second_val = sorted_opts[1][1]

        if darkest_val > dark_threshold:
            answers[q_no] = None  # kono bubble e mark nai (blank)
        elif (second_val - darkest_val) < min_gap:
            answers[q_no] = None  # duita bubble kacha-kachi kalo -> unclear/multi-mark
        else:
            answers[q_no] = darkest_opt
    return answers


def score_answers(student_answers, key_string):
    """
    key_string: 'ABCD...' dhoroner 100 character er string (index 0 = Q1)
    Return: (score, total, wrong_question_numbers_list)
    """
    total = len(key_string)
    score = 0
    wrong = []
    for i in range(total):
        q_no = i + 1
        correct = key_string[i].upper()
        given = student_answers.get(q_no)
        if given == correct:
            score += 1
        else:
            wrong.append(q_no)
    return score, total, wrong
