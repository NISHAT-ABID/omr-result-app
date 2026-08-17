"""
omr_scanner.py
--------------
All logic for extracting answers from an OMR sheet photo lives here.

There are 2 supported sheet layouts:
  - 100 questions -> 4 printed blocks of 25 questions each (Q1-25, 26-50, 51-75, 76-100)
  - 40  questions -> 2 printed blocks of 20 questions each (Q1-20, 21-40)

Two separate calibration flows use the SAME 4-click pattern, just with
different question numbers plugged in depending on the layout:
  1) Question 1        - center of option A
  2) Question 1        - center of option D
  3) Question <N>       - center of option A (last row of the first block)
  4) Question <N + 1>   - center of option A (first row of the next block)
where N = 25 for the 100-question layout and N = 20 for the 40-question layout.
See get_layout() / calibration_points_info() below.

There are two places this 4-click calibration happens:
  - Mentor "OMR Sheet Setup" page: mentor clicks the 4 points once on a
    BLANK sheet, per layout (100q / 40q). This is kept mainly as a
    reference/setup-status record.
  - Student submission: EVERY student clicks the 4 points on their OWN
    uploaded photo, right before submitting. This is what's actually used
    to build the reading grid for that submission - since each photo can
    have a slightly different angle/crop from the phone camera, having the
    grid anchored to points clicked on that exact photo is far more
    reliable than trying to perspective-warp every photo to match one
    fixed reference grid.

Main steps for scoring one submission:
1. validate_omr_image() -> rejects photos that are too small, blurry, or
   too dark/bright, so the student gets a clear message instead of a
   silently wrong result. Runs on the full-resolution photo.
2. resize_max_dim() -> shrinks (never enlarges) the photo to a manageable
   size for the on-screen calibration click + fast pixel sampling.
3. build_grid() -> using the 4 clicked points (on that resized photo) and
   the exam's layout (100 or 40), computes every question's 4 bubble
   positions.
4. read_answers() -> checks pixel darkness at each bubble position to find
   which option (if any) was filled in.
"""

import cv2
import numpy as np

WARP_WIDTH = 1200
WARP_HEIGHT = 1600
TOTAL_QUESTIONS = 100
OPTIONS = ["A", "B", "C", "D"]
BUBBLE_SAMPLE_RADIUS = 12  # pixels; used as a default when no adaptive radius is given

# Student-facing photos are capped to this many pixels on the longer side
# before calibration/reading - keeps the click UI responsive and pixel
# sampling fast, without affecting the (separate) quality validation below,
# which always runs on the original full-resolution photo first.
STUDENT_DISPLAY_MAX_DIM = 1300

# ---- image-quality thresholds (tune if real-world photos misbehave) ----
MIN_WIDTH = 500
MIN_HEIGHT = 700
BLUR_VARIANCE_THRESHOLD = 60.0   # below this = too blurry (Laplacian variance)
DARK_MEAN_THRESHOLD = 40.0       # below this = too dark
BRIGHT_MEAN_THRESHOLD = 240.0    # above this = too bright / overexposed / blown out

# ---- sheet layouts: total_questions -> (questions_per_block, num_blocks) ----
LAYOUT_PRESETS = {
    100: (25, 4),
    40: (20, 2),
}


def get_layout(total_questions):
    """
    Returns (questions_per_block, num_blocks) for a given sheet size.
    100 -> 4 printed blocks of 25. 40 -> 2 printed blocks of 20.
    Falls back to a generic 2-block split for any other total (shouldn't
    normally happen, since the app only offers 100q / 40q exams).
    """
    if total_questions in LAYOUT_PRESETS:
        return LAYOUT_PRESETS[total_questions]
    blocks = 2
    per_block = -(-total_questions // blocks)  # ceil division
    return per_block, blocks


def calibration_points_info(total_questions):
    """
    Returns the ordered list of calibration points to click for this
    layout, as [{"key", "short", "full", "block", "role"}, ...].

    Unlike before (only 4 points total for the WHOLE sheet), this now asks
    for a top-A and bottom-A point for EVERY printed block, plus one Q1-D
    point (for A-to-D option spacing). This is what makes the reading
    grid resilient to a photo where the sheet isn't perfectly flat - each
    block gets its OWN row spacing instead of assuming every block is
    identically spaced from the first one, which is what caused later
    blocks to drift off (and read wrong/skipped) on curved or
    perspective-tilted photos.

    "block" is the 0-indexed block number, "role" is "top" / "bottom" /
    "optd" (the Q1-D option-spacing reference point).
    """
    per_block, blocks = get_layout(total_questions)
    points = []
    for b in range(blocks):
        block_start_q = b * per_block + 1
        block_end_q = min(block_start_q + per_block - 1, total_questions)

        points.append({
            "key": f"p{len(points) + 1}",
            "short": f"Q{block_start_q}-A",
            "full": f"Question {block_start_q} - center of bubble A",
            "block": b,
            "role": "top",
        })
        if b == 0:
            points.append({
                "key": f"p{len(points) + 1}",
                "short": f"Q{block_start_q}-D",
                "full": f"Question {block_start_q} - center of bubble D",
                "block": b,
                "role": "optd",
            })
        points.append({
            "key": f"p{len(points) + 1}",
            "short": f"Q{block_end_q}-A",
            "full": f"Question {block_end_q} - center of bubble A",
            "block": b,
            "role": "bottom",
        })
    return points

def validate_omr_image(image_bgr):
    """
    Runs quick, cheap checks on an uploaded photo before we try to score it.
    Always run this on the ORIGINAL full-resolution photo (before any
    resizing), since the blur/brightness thresholds were tuned for that.

    Returns: (ok: bool, errors: list[str], warnings: list[str])
      - errors   -> blocking problems; the submission should be refused
      - warnings -> non-blocking issues; submission can proceed but the
                    student should be told the result might be inaccurate
    """
    errors = []
    warnings = []

    if image_bgr is None or image_bgr.size == 0:
        return False, ["The uploaded file could not be read as an image."], []

    h, w = image_bgr.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        errors.append(
            f"Image resolution is too low ({w}x{h}). Please retake the photo with a "
            f"higher resolution camera, at least {MIN_WIDTH}x{MIN_HEIGHT}."
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(np.mean(gray))
    if mean_brightness < DARK_MEAN_THRESHOLD:
        errors.append("The photo is too dark to read. Please retake it in better lighting.")
    elif mean_brightness > BRIGHT_MEAN_THRESHOLD:
        warnings.append("The photo looks overexposed / very bright - results may be inaccurate.")

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_variance < BLUR_VARIANCE_THRESHOLD:
        errors.append("The photo looks blurry. Please hold the camera steady and retake it.")

    return (len(errors) == 0), errors, warnings


def resize_max_dim(image_bgr, max_dim=STUDENT_DISPLAY_MAX_DIM):
    """
    Shrinks (never enlarges) an image so its longer side is at most
    max_dim, keeping aspect ratio. Used to give a fast, responsive
    calibration/reading image without affecting quality validation
    (which should always run on the original photo, before this).
    """
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image_bgr
    scale = max_dim / float(longest)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_bubble_radius(image_bgr):
    """
    Adaptive bubble-sampling radius based on the image's actual size -
    needed because, unlike the fixed-size mentor calibration canvas,
    each student's resized photo can be a slightly different size.
    """
    h, w = image_bgr.shape[:2]
    return max(9, int(round(min(h, w) * 0.010)))


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
    and warps it flat onto a fixed size canvas. Used by the MENTOR's blank
    -sheet calibration page only - student submissions skip this and use
    their own clicked points directly on their (un-warped) photo instead.

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


def build_grid(calibration, total_questions=TOTAL_QUESTIONS):
    """
    calibration dict now holds ONE top-A and bottom-A point PER BLOCK
    (plus one Q1-D point for option spacing), keyed as produced by
    calibration_points_info() above.

    Each block's rows are interpolated using ONLY that block's own top/
    bottom points - so a fold, curve, or perspective tilt in the photo
    only affects the block it's actually in, instead of accumulating
    error into every block that comes after it (which is what happened
    with the old single-block_step-for-the-whole-sheet approach).
    """
    per_block, blocks = get_layout(total_questions)
    points_info = calibration_points_info(total_questions)

    q1_a = None
    q1_d = None
    block_tops = {}
    block_bottoms = {}
    for info in points_info:
        pt = np.array(calibration[info["key"]], dtype=float)
        b = info["block"]
        if info["role"] == "top":
            block_tops[b] = pt
            if b == 0:
                q1_a = pt
        elif info["role"] == "bottom":
            block_bottoms[b] = pt
        elif info["role"] == "optd":
            q1_d = pt

    option_step = (q1_d - q1_a) / (len(OPTIONS) - 1)

    grid = {}
    q_no = 1
    for b in range(blocks):
        top = block_tops[b]
        bottom = block_bottoms[b]
        rows_in_block = min(per_block, total_questions - b * per_block)
        if rows_in_block <= 0:
            break
        if rows_in_block > 1:
            row_step = (bottom - top) / (rows_in_block - 1)
        else:
            row_step = np.array([0.0, 0.0])
        for row in range(rows_in_block):
            if q_no > total_questions:
                return grid
            row_origin = top + row * row_step
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


def read_answers(warped_bgr, grid, dark_threshold=150, min_gap=15, radius=None):
    """
    For each question, compares the darkness of its 4 bubbles.

    Returns: dict {question_no: 'A'/'B'/'C'/'D'/None/'MULTI'}
      None   -> blank / nothing marked
      'MULTI'-> two (or more) bubbles marked close together - ambiguous,
                treated as a WRONG answer by score_answers() (not skipped),
                since a real exam would count a double-touch as invalid.
    """
    if radius is None:
        radius = BUBBLE_SAMPLE_RADIUS

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    answers = {}
    for q_no, options in grid.items():
        darkness = {opt: _bubble_darkness(gray, center, radius) for opt, center in options.items()}
        sorted_opts = sorted(darkness.items(), key=lambda kv: kv[1])
        darkest_opt, darkest_val = sorted_opts[0]
        second_val = sorted_opts[1][1]

        if darkest_val > dark_threshold:
            answers[q_no] = None  # nothing marked
        elif (second_val - darkest_val) < min_gap:
            answers[q_no] = "MULTI"  # two bubbles nearly equally dark
        else:
            answers[q_no] = darkest_opt
    return answers

def score_answers(student_answers, key_string, negative_marking=False, negative_value=0.0):
    """
    key_string: a string like 'ABCD...' (index 0 = Q1)

    A 'MULTI' (double-touched) answer now counts as ATTEMPTED and WRONG -
    it will be penalized under negative marking, same as a normal wrong
    answer, instead of silently being treated as skipped.
    """
    total = len(key_string)
    correct = 0
    answered = 0
    wrong = []
    wrong_details = {}
    skipped_questions = []

    for i in range(total):
        q_no = i + 1
        correct_ans = key_string[i].upper()
        given = student_answers.get(q_no)
        if given is None:
            skipped_questions.append(q_no)
            continue
        answered += 1
        if given == "MULTI":
            wrong.append(q_no)
            wrong_details[q_no] = {"given": "Multiple", "correct": correct_ans}
        elif given == correct_ans:
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
        "skipped_questions": skipped_questions,
        "accuracy": accuracy,
        "marks": marks,
        "negative_marking": negative_marking,
        "negative_value": negative_value,
    }

def build_review_rows(student_answers, key_string):
    """
    status is one of "correct", "wrong", "skipped". A 'MULTI' answer shows
    up as "wrong" with given="Multiple" (the review UI just won't
    highlight a specific wrong bubble for it, since two were touched).
    """
    rows = []
    for i, correct_ans in enumerate(key_string):
        q_no = i + 1
        given = student_answers.get(q_no)
        correct_ans = correct_ans.upper()
        if given is None:
            status = "skipped"
        elif given == "MULTI":
            status = "wrong"
        elif given == correct_ans:
            status = "correct"
        else:
            status = "wrong"
        rows.append({"q": q_no, "given": given, "correct": correct_ans, "status": status})
    return rows


# ---------------- Visual answer-key input (clickable sheet for the mentor) ----------------

def render_sheet_image(grid, total_questions=100, answers=None):
    """
    Uses a calibration grid to draw a blank/filled bubble-sheet image
    (a PIL Image) that can be shown and clicked on - just like a real OMR
    sheet, for selecting answers.

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
