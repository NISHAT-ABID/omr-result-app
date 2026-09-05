"""OMR scanner for 40/50/100-question sheets.

Student calibration is performed on the exact uploaded photo. Reading uses
local bubble-center contrast + ink density instead of a single whole-patch
mean, which is much less likely to mistake printed bubble outlines/letters
for filled answers.

Important detection fix:
- Do NOT use grayscale luminance for bubble darkness.
- Printed pink/magenta OMR graphics can look dark after grayscale conversion
  because the green/blue channels are low.
- Real pen/pencil marks are dark in all RGB channels.
- Therefore bubble darkness is measured from max(R, G, B), i.e. the HSV
  "Value" channel. A printed pink mark remains bright in this channel while
  a genuinely dark pen mark remains dark.
"""

import cv2
import numpy as np


WARP_WIDTH = 1200
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


# Detection tuning.
# The algorithm combines center darkness and local center-vs-ring contrast,
# then validates candidates against the other bubbles in the same question.
FILL_SCORE_THRESHOLD = 16.0
STRONG_FILL_SCORE = 24.0
MULTI_SECOND_SCORE = 25.0
MIN_STRONG_INK_FRACTION = 0.055
MULTI_MIN_INK_FRACTION = 0.18

# Value-channel threshold used after the printed pink/red ink is removed.
# It is intentionally moderate; the decision is also based on local contrast,
# ink coverage and the gap between the best and second-best option.
DARK_PIXEL_THRESHOLD = 155



def get_layout(total_questions):
    """Return the physical OMR geometry for an exam.

    40-question and 50-question exams intentionally share the same physical
    50-question sheet: two blocks of 25.  A 40-question exam simply reads
    Q1-Q40 and silently ignores the unused Q41-Q50 area.
    """
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
    """Return calibration points for the physical sheet geometry.

    The 40/50 layout is the same printed sheet, so both use Q1/Q25/Q26/Q50
    reference points.  This keeps mentor and student calibration consistent
    and avoids maintaining two almost-identical sheet geometries.
    """
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


def _bubble_metrics(bgr, center, radius, dark_threshold=DARK_PIXEL_THRESHOLD):
    """Return robust evidence for one bubble.

    The sheet print is pink/red while student marks are normally neutral black,
    grey, blue or dark pencil.  We therefore suppress red-dominant print first,
    then inspect the bubble core rather than the whole circle.  The returned
    score deliberately combines several independent signals so one bad pixel
    threshold cannot turn a real mark into a skip.
    """
    x, y = center
    h, w = bgr.shape[:2]
    r = max(6, int(radius))

    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    d2 = xx * xx + yy * yy

    # The centre is where student ink is most useful.  The wider ring is a
    # local paper reference and is kept away from the printed bubble outline.
    core_mask = d2 <= (r * 0.52) ** 2
    inner_mask = d2 <= (r * 0.64) ** 2
    ring_mask = (d2 >= (r * 0.78) ** 2) & (d2 <= r * r)

    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0, 255.0, 255.0

    mh, mw = patch.shape[:2]
    core = core_mask[:mh, :mw]
    inner = inner_mask[:mh, :mw]
    ring = ring_mask[:mh, :mw]

    b_chan = patch[:, :, 0].astype(np.int16)
    g_chan = patch[:, :, 1].astype(np.int16)
    r_chan = patch[:, :, 2].astype(np.int16)
    max_rgb = np.maximum(np.maximum(b_chan, g_chan), r_chan)
    min_rgb = np.minimum(np.minimum(b_chan, g_chan), r_chan)
    red_dominance = r_chan - np.maximum(b_chan, g_chan)
    chroma = max_rgb - min_rgb

    printed_red = (
        (red_dominance >= 16) &
        (chroma >= 20) &
        (r_chan >= 75)
    )
    very_dark_neutral = (
        (max_rgb <= 100) &
        (np.abs(r_chan - g_chan) <= 30) &
        (np.abs(g_chan - b_chan) <= 30)
    )
    printed_red &= ~very_dark_neutral

    value = max_rgb.astype(np.float32)

    # Use the centre core for ink coverage.  A lower threshold is paired with
    # colour suppression and relative comparison, so faint real marks survive.
    dark = (value < float(dark_threshold)) & (~printed_red)
    core_value = value[core]
    core_dark = dark[core]
    inner_value = value[inner]
    inner_dark = dark[inner]

    if core_value.size == 0:
        return 0.0, 0.0, 255.0, 255.0

    clean_core = core_value[~printed_red[core]]
    if clean_core.size:
        center_mean = float(np.median(clean_core))
    else:
        center_mean = float(np.median(core_value))

    # Dark coverage is more stable than a raw mean when a pen stroke covers
    # only part of the bubble.  Include a softer darkness percentile as a
    # second signal for very light pencil marks.
    ink_fraction = float(np.mean(core_dark))
    inner_ink_fraction = float(np.mean(inner_dark)) if inner_dark.size else ink_fraction

    darkness = np.clip(255.0 - core_value.astype(np.float32), 0.0, 255.0)
    if clean_core.size:
        clean_darkness = np.clip(255.0 - clean_core.astype(np.float32), 0.0, 255.0)
        p65_dark = float(np.percentile(clean_darkness, 65))
    else:
        p65_dark = float(np.percentile(darkness, 65))

    ring_value = value[ring]
    ring_red = printed_red[ring]
    clean_ring = ring_value[~ring_red]
    ring_mean = float(np.median(clean_ring)) if clean_ring.size else 255.0

    contrast = max(0.0, ring_mean - center_mean)
    # Saturate the darkness contribution so one tiny black artifact cannot
    # dominate the decision.
    darkness_signal = min(45.0, p65_dark)

    score = (
        0.50 * contrast
        + 42.0 * ink_fraction
        + 0.18 * darkness_signal
        + 14.0 * inner_ink_fraction
    )

    return score, ink_fraction, center_mean, ring_mean


def read_answers(
    warped_bgr,
    grid,
    dark_threshold=DARK_PIXEL_THRESHOLD,
    min_gap=15,
    radius=None,
):
    """Read answers using adaptive, within-question evidence.

    40-question exams use the first 40 positions of the physical 50-question
    sheet because the supplied grid already contains only the requested Qs.
    """
    radius = BUBBLE_SAMPLE_RADIUS if radius is None else int(radius)
    smoothed = cv2.GaussianBlur(warped_bgr, (3, 3), 0)
    answers = {}

    for q_no, options in grid.items():
        metrics = {
            opt: _bubble_metrics(smoothed, center, radius, dark_threshold=dark_threshold)
            for opt, center in options.items()
        }
        scores = {opt: metrics[opt][0] for opt in OPTIONS}
        inks = {opt: metrics[opt][1] for opt in OPTIONS}
        ordered = sorted(OPTIONS, key=lambda o: scores[o], reverse=True)
        best, second = ordered[0], ordered[1]
        best_score, second_score = scores[best], scores[second]
        margin = best_score - second_score

        # Adaptive baseline: compare the best bubble with the other three.
        # This is much more stable across bright/dark phone photos than one
        # global score cutoff.
        other_scores = [scores[o] for o in OPTIONS if o != best]
        baseline = float(np.median(other_scores))
        relative_gain = best_score - baseline

        candidates = [
            o for o in OPTIONS
            if scores[o] >= FILL_SCORE_THRESHOLD and inks[o] >= MIN_STRONG_INK_FRACTION
        ]

        # Double touch: two bubbles must each have real centre ink.  Merely
        # having similar scores is not enough; both need independent evidence.
        strong = [
            o for o in OPTIONS
            if scores[o] >= MULTI_SECOND_SCORE and inks[o] >= MULTI_MIN_INK_FRACTION
        ]
        if len(strong) >= 2:
            # Require the two strongest marks to be reasonably close.  This
            # prevents a single genuine fill plus a printed/noisy artefact from
            # becoming MULTI.
            second_strong = ordered[1]
            if (
                best_score >= STRONG_FILL_SCORE
                and second_strong in strong
                and second_score >= best_score * 0.70
            ):
                answers[q_no] = "MULTI"
                continue

        if not candidates:
            answers[q_no] = None
            continue

        # Strong winner -> answer.  Medium winner -> answer only when it has a
        # meaningful advantage over the other three bubbles.  Otherwise leave
        # it reviewable instead of confidently guessing.
        if best_score >= STRONG_FILL_SCORE:
            answers[q_no] = best
        elif (
            best_score >= FILL_SCORE_THRESHOLD
            and relative_gain >= max(6.0, float(min_gap) * 0.38)
            and margin >= max(4.5, float(min_gap) * 0.28)
        ):
            answers[q_no] = best
        else:
            # If there is measurable ink but the evidence is ambiguous, keep
            # it as MULTI only when two options really qualify; otherwise blank.
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
