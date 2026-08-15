"""
omr_scanner.py
--------------
All logic for extracting answers from an OMR sheet photo lives here.

There are 2 main steps:
1. detect_and_warp() -> finds the sheet's 4 corners in the photo and
   straightens it ("perspective warp") onto a fixed-size canvas.
2. read_answers() -> using the calibration data, checks the pixel
   darkness at each question's 4 bubble positions (A/B/C/D) to find
   which one is filled in.

Calibration = the mentor uploads a blank OMR sheet once on the
"Calibration" page and clicks 4 points:
  1) Question 1  - center of option A
  2) Question 1  - center of option D
  3) Question 25 - center of option A (same block, last row)
  4) Question 26 - center of option A (next block, first row)

From these 4 points, all 100 questions' bubble positions are computed,
since a printed sheet always has a uniform grid.
"""

import cv2
import numpy as np

WARP_WIDTH = 1200
WARP_HEIGHT = 1600
TOTAL_QUESTIONS = 100
QUESTIONS_PER_BLOCK = 25
NUM_BLOCKS = 4
OPTIONS = ["A", "B", "C", "D"]
BUBBLE_SAMPLE_RADIUS = 12  # pixels; may need adjusting depending on the warped image


def _order_points(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
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
    Finds the largest 4-cornered contour (the sheet's border) in the photo
    and warps it flat onto a fixed size canvas.

    If no good contour is found, just resizes the image and returns it
    (the user should then be asked to retake a straighter photo).
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
        # fallback: couldn't warp, just resize
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
    calibration dict contains 4 clicked points (x, y):
      q1_a, q1_d, q25_a, q26_a

    From these, computes each question's (1-100) 4 bubble centers and
    returns a dict: { question_no: {"A": (x,y), "B": (x,y), ...} }
    """
    q1_a = np.array(calibration["q1_a"], dtype=float)
    q1_d = np.array(calibration["q1_d"], dtype=float)
    q25_a = np.array(calibration["q25_a"], dtype=float)
    q26_a = np.array(calibration["q26_a"], dtype=float)

    option_step = (q1_d - q1_a) / (len(OPTIONS) - 1)       # distance A->D, split into 3
    row_step = (q25_a - q1_a) / (QUESTIONS_PER_BLOCK - 1)  # distance row to row
    block_step = q26_a - q1_a                               # x/y shift from one block to the next

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
    For each question, compares the darkness of its 4 bubbles to find
    which one is marked.

    dark_threshold: below this, the bubble is considered filled (dark)
    min_gap: if the darkest and second-darkest bubbles are too close in
             darkness, it's treated as "unclear / multiple marked"

    Returns: dict {question_no: 'A'/'B'/'C'/'D'/None}
             None means blank or unclear
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    answers = {}
    for q_no, options in grid.items():
        darkness = {opt: _bubble_darkness(gray, center) for opt, center in options.items()}
        sorted_opts = sorted(darkness.items(), key=lambda kv: kv[1])
        darkest_opt, darkest_val = sorted_opts[0]
        second_val = sorted_opts[1][1]

        if darkest_val > dark_threshold:
            answers[q_no] = None  # no bubble marked (blank)
        elif (second_val - darkest_val) < min_gap:
            answers[q_no] = None  # two bubbles nearly equally dark -> unclear/multi-mark
        else:
            answers[q_no] = darkest_opt

    return answers


def score_answers(student_answers, key_string, negative_marking=False, negative_value=0.0):
    """
    key_string: a string like 'ABCD...' (index 0 = Q1)
    negative_marking: if True, `negative_value` marks are deducted for every
                       WRONG (attempted-but-incorrect) answer. Skipped/blank
                       questions are never penalized.

    Returns a dict:
      total          - total number of questions
      answered       - how many the student attempted (non-blank)
      skipped        - how many were left blank
      correct        - how many were correct
      wrong_count    - how many were attempted but wrong
      wrong          - list of question numbers that were wrong
      wrong_details  - {question_no: {"given": "B", "correct": "C"}}
      accuracy       - correct / answered * 100 (0 if nothing answered)
      marks          - correct - (wrong_count * negative_value), if enabled
      negative_marking / negative_value - echoed back for display purposes
    """
    total = len(key_string)
    correct = 0
    answered = 0
    wrong = []
    wrong_details = {}

    for i in range(total):
        q_no = i + 1
        correct_ans = key_string[i].upper()
        given = student_answers.get(q_no)

        if given is None:
            continue

        answered += 1
        if given == correct_ans:
            correct += 1
        else:
            wrong.append(q_no)
            wrong_details[q_no] = {"given": given, "correct": correct_ans}

    skipped = total - answered
    wrong_count = len(wrong)
    penalty = (wrong_count * negative_value) if negative_marking else 0
    marks = round(correct - penalty, 2)
    accuracy = round((correct / answered) * 100, 2) if answered else 0.0

    return {
        "total": total,
        "answered": answered,
        "skipped": skipped,
        "correct": correct,
        "wrong_count": wrong_count,
        "wrong": wrong,
        "wrong_details": wrong_details,
        "accuracy": accuracy,
        "marks": marks,
        "negative_marking": negative_marking,
        "negative_value": negative_value,
    }


# ---------------- Visual answer-key input (clickable sheet for the mentor) ----------------

def render_sheet_image(grid, total_questions=100, answers=None):
    """
    Uses the calibration grid to draw a blank/filled bubble-sheet image
    (a PIL Image) that can be shown in the Mentor Panel and clicked on -
    just like a real OMR sheet, for selecting answers.

    answers: {question_no: 'A'/'B'/'C'/'D'} - filled bubbles are shown solid black.
    """
    from PIL import Image as PILImage, ImageDraw

    answers = answers or {}
    img = PILImage.new("RGB", (WARP_WIDTH, WARP_HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    r = BUBBLE_SAMPLE_RADIUS + 6
    for q in range(1, total_questions + 1):
        opts = grid.get(q)
        if not opts:
            continue
        for opt in OPTIONS:
            x, y = opts[opt]
            filled = answers.get(q) == opt
            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                outline=(30, 30, 30),
                width=2,
                fill=(20, 20, 20) if filled else (255, 255, 255),
            )
            draw.text((x - 4, y - 6), opt, fill=(255, 255, 255) if filled else (30, 30, 30))
        ax, ay = opts["A"]
        draw.text((ax - 44, ay - 7), str(q), fill=(0, 0, 0))

    return img


def find_clicked_bubble(grid, total_questions, x, y, radius=None):
    """
    Finds the nearest bubble (question, option) to a click position (x, y).
    Returns None if nothing is within the radius.
    """
    if radius is None:
        radius = BUBBLE_SAMPLE_RADIUS + 10

    best = None
    best_d = radius
    for q in range(1, total_questions + 1):
        opts = grid.get(q)
        if not opts:
            continue
        for opt in OPTIONS:
            bx, by = opts[opt]
            d = ((x - bx) ** 2 + (y - by) ** 2) ** 0.5
            if d <= best_d:
                best_d = d
                best = (q, opt)
    return best
