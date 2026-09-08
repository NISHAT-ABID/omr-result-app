"""OMR scanner for 40/50/100-question sheets.
Student calibration is performed on the exact uploaded photo. Reading uses
local bubble-center contrast + ink density instead of a single whole-patch
mean, which is much less likely to mistake printed bubble outlines/letters
for filled answers.
Important detection fix:
Do NOT use grayscale luminance for bubble darkness.
Printed pink/magenta OMR graphics can look dark after grayscale conversion
because the green/blue channels are low.
Real pen/pencil marks are dark in all RGB channels.
Therefore bubble darkness is measured from max(R, G, B), i.e. the HSV
"Value" channel. A printed pink mark remains bright in this channel while
a genuinely dark pen mark remains dark.
"""
import cv2
import numpy as np

WARP_WIDTH = 1000
WARP_HEIGHT = 1600
TOTAL_QUESTIONS = 100
OPTIONS = ["A", "B", "C", "D"]
BUBBLE_SAMPLE_RADIUS = 12
STUDENT_DISPLAY_MAX_DIM = 1300
MIN_WIDTH = 500
MIN_HEIGHT = 700
BLUR_VARIANCE_THRESHOLD = 60.0
DARK_MEAN_THRESHOLD = 40.0
BRIGHT_MEAN_THRESHOLD = 240.0
LAYOUT_PRESETS = {100: (25, 4), 50: (25, 2), 40: (25, 2)}

# IMPROVED: More conservative thresholds for better accuracy
FILL_SCORE_THRESHOLD = 10.0  # Increased from 9.0 for fewer false positives
MIN_STRONG_INK_FRACTION = 0.04  # Increased from 0.035
MULTI_MIN_CONTRAST = 24.0  # Increased from 22.0 for stricter multi-detection
MULTI_MIN_INK_FRACTION = 0.08  # Increased from 0.075
MULTI_RATIO_LIMIT = 0.70  # Decreased from 0.72 for stricter multi-detection
CLEAR_WINNER_MARGIN = 13.0  # Increased from 12.0

MARGIN_EROSION_PX = 2
DARK_PIXEL_THRESHOLD = 145


def get_layout(total_questions):
    """Return the physical OMR geometry for an exam."""
    total_questions = int(total_questions)
    if total_questions in (40, 50):
        return 25, 2
    if total_questions == 100:
        return 25, 4
    if total_questions > 50:
        blocks = 4
        per_block = 25
    else:
        blocks = 2
        per_block = 25
    return per_block, blocks


def calibration_points_info(total_questions):
    """Return calibration points for the physical sheet geometry."""
    total_questions = int(total_questions)
    physical_total = 50 if total_questions in (40, 50) else total_questions
    per_block, blocks = get_layout(physical_total)
    points = []
    for b in range(blocks):
        start = b * per_block + 1
        end = min(start + per_block - 1, physical_total)
        points.append({
            "key": f"p{len(points)+1}",
            "short": f"Q{start}-A",
            "full": f"Question {start} - center of bubble A",
            "block": b,
            "role": "top",
        })
        if b == 0:
            points.append({
                "key": f"p{len(points)+1}",
                "short": f"Q{start}-D",
                "full": f"Question {start} - center of bubble D",
                "block": b,
                "role": "optd",
            })
        points.append({
            "key": f"p{len(points)+1}",
            "short": f"Q{end}-A",
            "full": f"Question {end} - center of bubble A",
            "block": b,
            "role": "bottom",
        })
    return points


def validate_omr_image(image_bgr):
    errors, warnings = [], []
    if image_bgr is None or image_bgr.size == 0:
        return False, ["The uploaded file could not be read as an image."], []
    h, w = image_bgr.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        errors.append(
            f"Image resolution is too low ({w}x{h}). "
            f"Please retake the photo with a higher resolution camera, "
            f"at least {MIN_WIDTH}x{MIN_HEIGHT}."
        )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    if mean < DARK_MEAN_THRESHOLD:
        errors.append(
            "The photo is too dark to read. Please retake it in better lighting."
        )
    elif mean > BRIGHT_MEAN_THRESHOLD:
        warnings.append(
            "The photo looks overexposed / very bright - results may be inaccurate."
        )
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < BLUR_VARIANCE_THRESHOLD:
        errors.append(
            "The photo looks blurry. Please hold the camera steady and retake it."
        )
    return len(errors) == 0, errors, warnings


def resize_max_dim(image_bgr, max_dim=STUDENT_DISPLAY_MAX_DIM):
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image_bgr
    scale = max_dim / float(longest)
    return cv2.resize(
        image_bgr,
        (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


def compute_bubble_radius(image_bgr):
    h, w = image_bgr.shape[:2]
    return max(9, int(round(min(h, w) * 0.010)))


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect


def detect_and_warp(image_bgr):
    orig = image_bgr.copy()
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        50,
        150,
    )
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    sheet = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            sheet = approx
            break
    if sheet is None:
        return cv2.resize(
            orig,
            (WARP_WIDTH, WARP_HEIGHT),
        ), False
    rect = _order_points(
        sheet.reshape(4, 2).astype("float32")
    )
    dst = np.array(
        [
            [0, 0],
            [WARP_WIDTH - 1, 0],
            [WARP_WIDTH - 1, WARP_HEIGHT - 1],
            [0, WARP_HEIGHT - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        orig,
        matrix,
        (WARP_WIDTH, WARP_HEIGHT),
    ), True


def build_grid(calibration, total_questions=TOTAL_QUESTIONS):
    """Build bubble centers from calibration and return only requested Qs."""
    requested = int(total_questions)
    physical_total = 50 if requested in (40, 50) else requested
    per_block, blocks = get_layout(physical_total)
    info = calibration_points_info(requested)
    q1_a = q1_d = None
    tops, bottoms = {}, {}
    for item in info:
        if item["key"] not in calibration:
            raise ValueError(f"Calibration is missing point {item['key']} ({item['short']}).")
        pt = np.asarray(calibration[item["key"]], dtype=float)
        b = item["block"]
        if item["role"] == "top":
            tops[b] = pt
            if b == 0:
                q1_a = pt
        elif item["role"] == "bottom":
            bottoms[b] = pt
        else:
            q1_d = pt
    if q1_a is None or q1_d is None:
        raise ValueError("Calibration is missing the Q1 A/D spacing points.")
    option_step = (q1_d - q1_a) / 3.0
    grid, q_no = {}, 1
    for b in range(blocks):
        if b not in tops or b not in bottoms:
            raise ValueError(f"Calibration is missing block {b + 1} top/bottom points.")
        rows = per_block
        row_step = (bottoms[b] - tops[b]) / (rows - 1) if rows > 1 else np.array([0.0, 0.0])
        for r in range(rows):
            if q_no > requested:
                break
            origin = tops[b] + r * row_step
            grid[q_no] = {
                opt: (
                    int(round((origin + i * option_step)[0])),
                    int(round((origin + i * option_step)[1])),
                )
                for i, opt in enumerate(OPTIONS)
            }
            q_no += 1
    return grid


def _refine_bubble_center(image_bgr, center, search=32):
    """Snap one expected bubble center to the strongest circular printed ring nearby."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    x, y = int(round(center[0])), int(round(center[1]))
    h, w = gray.shape[:2]
    x0, x1 = max(0, x - search), min(w, x + search + 1)
    y0, y1 = max(0, y - search), min(h, y + search + 1)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return np.asarray(center, dtype=np.float32), 0.0
    circles = cv2.HoughCircles(
        crop, cv2.HOUGH_GRADIENT, dp=1.0, minDist=10,
        param1=70, param2=10, minRadius=7, maxRadius=24,
    )
    if circles is None:
        return np.asarray(center, dtype=np.float32), 0.0
    best = None
    for cx, cy, r in circles[0]:
        px, py = x0 + float(cx), y0 + float(cy)
        dist = float(np.hypot(px - x, py - y))
        if dist > search * 0.9:
            continue
        score = (search - dist) + min(float(r), 20.0) * 0.15
        if best is None or score > best[0]:
            best = (score, px, py)
    if best is None:
        return np.asarray(center, dtype=np.float32), 0.0
    return np.asarray([best[1], best[2]], dtype=np.float32), float(best[0])


def align_to_master_grid(image_bgr, grid, min_matches=30):
    """Locally refine the mentor grid before reading answers."""
    if image_bgr is None or image_bgr.size == 0 or not grid:
        return image_bgr, {"applied": False, "reason": "empty image or grid"}
    refined = {}
    moved = []
    for q_no, options in grid.items():
        refined[q_no] = {}
        for opt in OPTIONS:
            original = np.asarray(options[opt], dtype=np.float32)
            point, confidence = _refine_bubble_center(image_bgr, original, search=32)
            refined[q_no][opt] = (int(round(point[0])), int(round(point[1])))
            if confidence > 0:
                moved.append(float(np.linalg.norm(point - original)))
    if len(moved) < min_matches:
        return image_bgr, {"applied": False, "reason": f"only {len(moved)} bubble centers could be refined"}
    avg_move = float(np.mean(moved)) if moved else 0.0
    return image_bgr, {
        "applied": True,
        "matches": len(moved),
        "inliers": len(moved),
        "average_center_adjustment_px": round(avg_move, 2),
        "grid": refined,
    }


def preprocess_omr_image(image_bgr):
    """Prepare a raw OMR image without photographic enhancement."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty OMR image.")
    value = np.max(image_bgr, axis=2).astype(np.uint8)
    value = cv2.GaussianBlur(value, (3, 3), 0)
    return value


def _bubble_metrics(image_red, center, radius):
    """Return robust local evidence for ink inside one bubble."""
    x, y = int(round(center[0])), int(round(center[1]))
    h, w = image_red.shape[:2]
    r = max(7, int(radius))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    d2 = xx * xx + yy * yy
    core_r = max(3.0, r * 0.42 - MARGIN_EROSION_PX)
    ring_inner = r * 0.62
    ring_outer = r * 0.88
    core_mask = d2 <= core_r * core_r
    ring_mask = (d2 >= ring_inner * ring_inner) & (d2 <= ring_outer * ring_outer)
    x0, x1 = max(0, x-r), min(w, x+r+1)
    y0, y1 = max(0, y-r), min(h, y+r+1)
    patch = image_red[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0, 255.0, 255.0
    mh, mw = patch.shape[:2]
    cm = core_mask[:mh, :mw]
    rm = ring_mask[:mh, :mw]
    core = patch[cm]
    ring = patch[rm]
    if core.size == 0:
        return 0.0, 0.0, 255.0, 255.0
    core_mean = float(np.mean(core))
    ring_mean = float(np.mean(ring)) if ring.size else core_mean
    contrast = max(0.0, ring_mean - core_mean)
    local_cut = min(150.0, ring_mean - 18.0)
    ink_fraction = float(np.mean(core < local_cut))
    score = contrast * 0.72 + (ink_fraction * 100.0) * 0.28
    return float(score), ink_fraction, core_mean, ring_mean


def read_answers(warped_bgr, grid, dark_threshold=DARK_PIXEL_THRESHOLD, min_gap=15, radius=None):
    """Read answers from raw perspective-corrected OMR pixels."""
    del dark_threshold, min_gap
    radius = BUBBLE_SAMPLE_RADIUS if radius is None else int(radius)
    red = preprocess_omr_image(warped_bgr)
    answers = {}
    for q_no, options in grid.items():
        metrics = {
            opt: _bubble_metrics(red, center, radius)
            for opt, center in options.items()
        }
        scores = {opt: metrics[opt][0] for opt in OPTIONS}
        inks = {opt: metrics[opt][1] for opt in OPTIONS}
        ordered = sorted(OPTIONS, key=lambda o: scores[o], reverse=True)
        best, second = ordered[0], ordered[1]
        max1, max2 = scores[best], scores[second]
        strength2 = 0.0 if max1 <= 0.001 else max2 / max1
        best_contrast = max(0.0, metrics[best][3] - metrics[best][2])
        second_contrast = max(0.0, metrics[second][3] - metrics[second][2])
        
        # IMPROVED: Stricter multi-touch detection
        is_multi = (
            best_contrast >= MULTI_MIN_CONTRAST
            and second_contrast >= MULTI_MIN_CONTRAST
            and inks[best] >= MULTI_MIN_INK_FRACTION
            and inks[second] >= MULTI_MIN_INK_FRACTION
            and strength2 >= MULTI_RATIO_LIMIT
        )
        if is_multi:
            answers[q_no] = "MULTI"
            continue
        if (
            max1 < FILL_SCORE_THRESHOLD
            or best_contrast < 14.0
            or inks[best] < MIN_STRONG_INK_FRACTION
        ):
            answers[q_no] = None
            continue
        margin = max1 - max2
        if margin >= CLEAR_WINNER_MARGIN or strength2 < 0.55:
            answers[q_no] = best
        else:
            answers[q_no] = None
    return answers


def score_answers(
    student_answers,
    key_string,
    negative_marking=False,
    negative_value=0.0,
):
    total = len(key_string)
    correct = 0
    answered = 0
    wrong = []
    wrong_details = {}
    skipped_questions = []
    for i in range(total):
        q = i + 1
        correct_ans = key_string[i].upper()
        given = student_answers.get(q)
        if given is None:
            skipped_questions.append(q)
            continue
        answered += 1
        if given == "MULTI":
            wrong.append(q)
            wrong_details[q] = {
                "given": "Multiple",
                "correct": correct_ans,
            }
        elif given == correct_ans:
            correct += 1
        else:
            wrong.append(q)
            wrong_details[q] = {
                "given": given,
                "correct": correct_ans,
            }
    wrong_count = len(wrong)
    penalty = (
        wrong_count * negative_value
        if negative_marking
        else 0
    )
    marks = round(
        correct - penalty,
        2,
    )
    accuracy = (
        round(correct / answered * 100, 2)
        if answered
        else 0.0
    )
    return {
        "total": total,
        "answered": answered,
        "skipped": total - answered,
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
    rows = []
    for i, correct_ans in enumerate(key_string):
        q = i + 1
        given = student_answers.get(q)
        ca = correct_ans.upper()
        status = (
            "skipped"
            if given is None
            else (
                "wrong"
                if given == "MULTI" or given != ca
                else "correct"
            )
        )
        rows.append({
            "q": q,
            "given": given,
            "correct": ca,
            "status": status,
        })
    return rows


def render_sheet_image(
    grid,
    total_questions=100,
    answers=None,
):
    from PIL import Image as PILImage, ImageDraw
    answers = answers or {}
    img = PILImage.new(
        "RGB",
        (WARP_WIDTH, WARP_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(img)
    r = BUBBLE_SAMPLE_RADIUS + 6
    for q in range(
        1,
        total_questions + 1,
    ):
        opts = grid.get(q)
        if not opts:
            continue
        for opt in OPTIONS:
            x, y = opts[opt]
            filled = answers.get(q) == opt
            draw.ellipse(
                [
                    x - r,
                    y - r,
                    x + r,
                    y + r,
                ],
                outline=(30, 30, 30),
                width=2,
                fill=(
                    (20, 20, 20)
                    if filled
                    else (255, 255, 255)
                ),
            )
            draw.text(
                (x - 4, y - 6),
                opt,
                fill=(
                    (255, 255, 255)
                    if filled
                    else (30, 30, 30)
                ),
            )
        ax, ay = opts["A"]
        draw.text(
            (ax - 44, ay - 7),
            str(q),
            fill=(0, 0, 0),
        )
    return img


def find_clicked_bubble(
    grid,
    total_questions,
    x,
    y,
    radius=None,
):
    radius = (
        BUBBLE_SAMPLE_RADIUS + 10
        if radius is None
        else radius
    )
    best = None
    best_d = radius
    for q in range(
        1,
        total_questions + 1,
    ):
        opts = grid.get(q)
        if not opts:
            continue
        for opt, (bx, by) in opts.items():
            d = (
                (x - bx) ** 2
                + (y - by) ** 2
            ) ** 0.5
            if d <= best_d:
                best_d = d
                best = (q, opt)
    return best
