"""
omr_scanner.py
--------------
All logic for extracting answers from an OMR sheet photo lives here.

There are 2 main steps:
  1. detect_and_warp() -> finds the sheet's 4 corners in the photo and
     straightens it ("perspective warp") onto a fixed-size canvas.
  2. read_answers()    -> using the calibration data, checks the pixel
     darkness at each question's 4 bubble positions (A/B/C/D) to find
     which one(s) are filled in.

Calibration = the mentor uploads a blank OMR sheet once (per sheet type:
100Q or 40Q - since the two layouts are printed differently) and clicks
4 points:
  1) Question 1              - center of option A
  2) Question 1               - center of option D
  3) Last question of block 1 - center of option A  (Q25 for 100Q, Q20 for 40Q)
  4) First question of block 2 - center of option A (Q26 for 100Q, Q21 for 40Q)

From these 4 points + the sheet's layout meta (questions_per_block,
num_blocks), every question's bubble positions are computed, since a
printed sheet always has a uniform grid.

Each calibration dict now carries its own layout meta so build_grid()
works for ANY layout (100Q: 4 blocks of 25, 40Q: 2 blocks of 20, or
anything else) without hardcoding assumptions:

    {
        "q1_a":  (x, y),
        "q1_d":  (x, y),
        "qlast_a": (x, y),   # last question of block 1, option A
        "qnext_a": (x, y),   # first question of block 2, option A
        "total_questions": 100,
        "questions_per_block": 25,
        "num_blocks": 4,
    }
"""

import cv2
import numpy as np

WARP_WIDTH = 1200
WARP_HEIGHT = 1600

OPTIONS = ["A", "B", "C", "D"]
BUBBLE_SAMPLE_RADIUS = 12  # pixels; may need adjusting depending on the warped image

# Layout presets used by the Mentor Calibration UI.
# key = total_questions the mentor selected in the Answer Key / Calibration step
LAYOUTS = {
    100: {"questions_per_block": 25, "num_blocks": 4,
          "block_labels": ["Q1-A", "Q1-D", "Q25-A", "Q26-A"]},
    40: {"questions_per_block": 20, "num_blocks": 2,
         "block_labels": ["Q1-A", "Q1-D", "Q20-A", "Q21-A"]},
}


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
    calibration dict must contain:
        q1_a, q1_d, qlast_a, qnext_a  (clicked points)
        questions_per_block, num_blocks  (layout meta - set at save time)

    From these, computes every question's (1..total) 4 bubble centers and
    returns a dict: { question_no: {"A": (x,y), "B": (x,y), ...} }

    Works for ANY layout (100Q / 4x25, 40Q / 2x20, or others) since the
    block/row geometry is derived purely from the calibration points and
    meta - nothing is hardcoded here anymore.
    """
    q1_a = np.array(calibration["q1_a"], dtype=float)
    q1_d = np.array(calibration["q1_d"], dtype=float)
    qlast_a = np.array(calibration["qlast_a"], dtype=float)
    qnext_a = np.array(calibration["qnext_a"], dtype=float)

    questions_per_block = int(calibration["questions_per_block"])
    num_blocks = int(calibration["num_blocks"])

    option_step = (q1_d - q1_a) / (len(OPTIONS) - 1)          # A -> D, split into 3
    row_step = (qlast_a - q1_a) / (questions_per_block - 1)    # row to row
    block_step = qnext_a - q1_a                                 # block to block

    grid = {}
    q_no = 1
    for block in range(num_blocks):
        block_origin = q1_a + block * block_step
        for row in range(questions_per_block):
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
    which one is marked - and also detects "double touch" (more than one
    bubble filled in for the same question).

    dark_threshold: below this, a bubble is considered filled (dark)
    min_gap: if the darkest and second-darkest bubbles are too close in
             darkness, it's treated as an unclear double-touch too
             (a light smudge on a 2nd bubble often reads this way)

    Returns a tuple (answers, marks_detail):
        answers:      {question_no: 'A'/'B'/'C'/'D'/None}
                       None means blank OR double-touch (ambiguous/invalid)
        marks_detail: {question_no: [list of option letters that were
                       actually marked/filled]}
                       - []              -> genuinely blank (skipped)
                       - ['B']           -> single clean mark
                       - ['A', 'C']      -> double touch (2+ bubbles filled)
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    answers = {}
    marks_detail = {}

    for q_no, options in grid.items():
        darkness = {opt: _bubble_darkness(gray, center) for opt, center in options.items()}
        sorted_opts = sorted(darkness.items(), key=lambda kv: kv[1])
        darkest_opt, darkest_val = sorted_opts[0]
        second_opt, second_val = sorted_opts[1]

        filled = [opt for opt, val in darkness.items() if val <= dark_threshold]

        if darkest_val > dark_threshold:
            # nothing dark enough at all -> blank / skipped
            answers[q_no] = None
            marks_detail[q_no] = []
        elif len(filled) >= 2:
            # 2+ bubbles clearly filled -> double touch, invalid
            answers[q_no] = None
            marks_detail[q_no] = filled
        elif (second_val - darkest_val) < min_gap:
            # top 2 bubbles too close in darkness to call confidently ->
            # treat conservatively as a double touch between those two
            answers[q_no] = None
            marks_detail[q_no] = [darkest_opt, second_opt]
        else:
            answers[q_no] = darkest_opt
            marks_detail[q_no] = [darkest_opt]

    return answers, marks_detail


def score_answers(student_answers, key_string, marks_detail=None,
                   negative_marking=False, negative_value=0.0):
    """
    key_string: a string like 'ABCD...' (index 0 = Q1)
    marks_detail: output from read_answers() - used to tell a genuine
        blank apart from a double-touch, and to know exactly which
        option(s) the student marked (for the Solution view).

    Rules:
      - blank (nothing marked)         -> counted as SKIPPED, never penalized
      - single clean mark, correct     -> counted as CORRECT
      - single clean mark, wrong       -> counted as WRONG
      - double touch (2+ bubbles)      -> counted as WRONG (an attempt was
                                           made, it's just invalid), and both
                                           the negative-marking penalty (if on)
                                           applies to it like any other wrong answer

    Returns a dict:
        total, answered, skipped, correct, wrong_count
        wrong        - list of wrongly-answered question numbers
                       (includes double-touch questions)
        wrong_details   - {q_no: {"given": [opts marked], "correct": "C"}}
        skipped_list - list of genuinely blank question numbers
        skipped_details - {q_no: {"given": [], "correct": "C"}}
        accuracy, marks, negative_marking, negative_value
    """
    marks_detail = marks_detail or {}
    total = len(key_string)
    correct = 0
    answered = 0
    wrong = []
    wrong_details = {}
    skipped_list = []
    skipped_details = {}

    for i in range(total):
        q_no = i + 1
        correct_ans = key_string[i].upper()
        given = student_answers.get(q_no)
        marked = marks_detail.get(q_no, [] if given is None else [given])

        if given is None:
            if len(marked) >= 2:
                # double touch -> treated as an attempted wrong answer
                answered += 1
                wrong.append(q_no)
                wrong_details[q_no] = {"given": marked, "correct": correct_ans}
            else:
                # genuinely blank -> skipped, never penalized
                skipped_list.append(q_no)
                skipped_details[q_no] = {"given": [], "correct": correct_ans}
            continue

        answered += 1
        if given == correct_ans:
            correct += 1
        else:
            wrong.append(q_no)
            wrong_details[q_no] = {"given": marked, "correct": correct_ans}

    skipped = len(skipped_list)
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
        "skipped_list": skipped_list,
        "skipped_details": skipped_details,
        "accuracy": accuracy,
        "marks": marks,
        "negative_marking": negative_marking,
        "negative_value": negative_value,
    }


# ---------------- Visual answer-key input (clickable sheet for the mentor) ----------------
# NOTE: kept for backwards-compatibility / optional future use - the current
# Mentor "Answer Key" tab uses a native bubble-grid (st.radio) instead of this
# image-clicking approach, so these two helpers aren't on the main app flow.

def render_sheet_image(grid, total_questions=100, answers=None):
    """
    Uses a calibration grid to draw a blank/filled bubble-sheet image
    (a PIL Image) - filled bubbles are shown solid black.
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
