"""OMR scanner for 40/100-question sheets.

Student calibration is performed on the exact uploaded photo.  Reading uses
local bubble-center contrast + ink density instead of a single whole-patch
mean, which is much less likely to mistake printed bubble outlines/letters for
filled answers.

NOTE (fix): darkness is measured from the max(R,G,B) "value" channel instead
of grayscale luminance (0.299R+0.587G+0.114B). Luminance under-weights the Red
channel, so bright printed pink/magenta bubble outlines and A/B/C/D letters
(high-R, low-G, low-B) were being scored as "dark ink" and almost every
question came out as MULTI (double-touch). Actual pen/pencil marks are dark in
every channel, so max(R,G,B) stays low for them but stays high for printed
pink - this alone separates the two without risking false negatives on
saturated blue pen ink (which a saturation-based filter would risk).
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
LAYOUT_PRESETS = {100: (25, 4), 40: (20, 2)}

# Detection tuning.  The algorithm combines center darkness and local
# center-vs-ring contrast, then validates candidates against their own row.
FILL_SCORE_THRESHOLD = 20.0
STRONG_FILL_SCORE = 30.0
MULTI_SECOND_SCORE = 17.0
MIN_STRONG_INK_FRACTION = 0.10
DARK_PIXEL_THRESHOLD = 175


def get_layout(total_questions):
    if total_questions in LAYOUT_PRESETS:
        return LAYOUT_PRESETS[total_questions]
    blocks = 2
    per_block = -(-total_questions // blocks)
    return per_block, blocks


def calibration_points_info(total_questions):
    per_block, blocks = get_layout(total_questions)
    points = []
    for b in range(blocks):
        start = b * per_block + 1
        end = min(start + per_block - 1, total_questions)
        points.append({"key": f"p{len(points)+1}", "short": f"Q{start}-A", "full": f"Question {start} - center of bubble A", "block": b, "role": "top"})
        if b == 0:
            points.append({"key": f"p{len(points)+1}", "short": f"Q{start}-D", "full": f"Question {start} - center of bubble D", "block": b, "role": "optd"})
        points.append({"key": f"p{len(points)+1}", "short": f"Q{end}-A", "full": f"Question {end} - center of bubble A", "block": b, "role": "bottom"})
    return points


def validate_omr_image(image_bgr):
    errors, warnings = [], []
    if image_bgr is None or image_bgr.size == 0:
        return False, ["The uploaded file could not be read as an image."], []
    h, w = image_bgr.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        errors.append(f"Image resolution is too low ({w}x{h}). Please retake the photo with a higher resolution camera, at least {MIN_WIDTH}x{MIN_HEIGHT}.")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    if mean < DARK_MEAN_THRESHOLD:
        errors.append("The photo is too dark to read. Please retake it in better lighting.")
    elif mean > BRIGHT_MEAN_THRESHOLD:
        warnings.append("The photo looks overexposed / very bright - results may be inaccurate.")
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < BLUR_VARIANCE_THRESHOLD:
        errors.append("The photo looks blurry. Please hold the camera steady and retake it.")
    return len(errors) == 0, errors, warnings


def resize_max_dim(image_bgr, max_dim=STUDENT_DISPLAY_MAX_DIM):
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image_bgr
    scale = max_dim / float(longest)
    return cv2.resize(image_bgr, (max(1, int(round(w*scale))), max(1, int(round(h*scale)))), interpolation=cv2.INTER_AREA)


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
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            sheet = approx
            break
    if sheet is None:
        return cv2.resize(orig, (WARP_WIDTH, WARP_HEIGHT)), False
    rect = _order_points(sheet.reshape(4,2).astype("float32"))
    dst = np.array([[0,0],[WARP_WIDTH-1,0],[WARP_WIDTH-1,WARP_HEIGHT-1],[0,WARP_HEIGHT-1]], dtype="float32")
    return cv2.warpPerspective(orig, cv2.getPerspectiveTransform(rect,dst), (WARP_WIDTH,WARP_HEIGHT)), True


def build_grid(calibration, total_questions=TOTAL_QUESTIONS):
    per_block, blocks = get_layout(total_questions)
    info = calibration_points_info(total_questions)
    q1_a = q1_d = None
    tops, bottoms = {}, {}
    for item in info:
        pt = np.asarray(calibration[item["key"]], dtype=float)
        b = item["block"]
        if item["role"] == "top":
            tops[b] = pt
            if b == 0: q1_a = pt
        elif item["role"] == "bottom":
            bottoms[b] = pt
        else:
            q1_d = pt
    if q1_a is None or q1_d is None:
        raise ValueError("Calibration is missing the Q1 A/D spacing points.")
    option_step = (q1_d - q1_a) / 3.0
    grid, q_no = {}, 1
    for b in range(blocks):
        if b not in tops or b not in bottoms: continue
        rows = min(per_block, total_questions - b*per_block)
        row_step = (bottoms[b]-tops[b])/(rows-1) if rows > 1 else np.array([0.,0.])
        for r in range(rows):
            origin = tops[b] + r*row_step
            grid[q_no] = {opt: (int(round((origin+i*option_step)[0])), int(round((origin+i*option_step)[1]))) for i,opt in enumerate(OPTIONS)}
            q_no += 1
    return grid


def _bubble_metrics(bgr, center, radius):
    """Measures how "inked" a bubble is.

    `bgr` is a color (BGR) image/patch. Darkness is measured from the
    max(R,G,B) value channel instead of grayscale luminance, so bright
    printed pink/magenta outlines and letters (high-R, low-G/B) are not
    mistaken for dark pen/pencil ink (which is dark across all channels).
    """
    x, y = center
    h, w = bgr.shape[:2]
    r = max(5, int(radius))
    # Inner area avoids most printed circle outlines; ring estimates local paper brightness.
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    d2 = xx*xx + yy*yy
    inner_mask = d2 <= (r*0.58)**2
    ring_mask = (d2 >= (r*0.72)**2) & (d2 <= r*r)
    x0, x1 = max(0,x-r), min(w,x+r+1)
    y0, y1 = max(0,y-r), min(h,y+r+1)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0, 255.0, 255.0
    # "value" = max(R,G,B) per pixel, i.e. the HSV V channel. Robust to hue,
    # so it doesn't matter whether the printed ink is pink/magenta/red -
    # only genuinely dark (low in every channel) pixels score as ink.
    value = np.max(patch, axis=2).astype(np.float32)
    # Masks are cropped at image edges if necessary.
    mh, mw = value.shape[:2]
    im = inner_mask[:mh,:mw]
    rm = ring_mask[:mh,:mw]
    inner = value[im]
    ring = value[rm]
    if inner.size == 0: return 0.0, 0.0, 255.0, 255.0
    center_mean = float(np.mean(inner))
    ring_mean = float(np.median(ring)) if ring.size else 255.0
    ink_fraction = float(np.mean(inner < DARK_PIXEL_THRESHOLD))
    contrast = max(0.0, ring_mean - center_mean)
    # Contrast is strongest signal; ink fraction catches lighter pen/pencil marks.
    score = 0.72*contrast + 28.0*ink_fraction
    return score, ink_fraction, center_mean, ring_mean


def read_answers(warped_bgr, grid, dark_threshold=150, min_gap=15, radius=None):
    """Read bubbles with local adaptive scoring.

    None = blank, A-D = one confident mark, MULTI = two or more independently
    strong marks.  Printed bubble outlines/letters are suppressed by sampling
    the inner bubble area (via the max(R,G,B) value channel, not grayscale
    luminance) and comparing it with the surrounding paper ring.
    """
    radius = BUBBLE_SAMPLE_RADIUS if radius is None else int(radius)
    # Blur in color (not converted to grayscale) so the value-channel trick
    # in _bubble_metrics keeps working on real RGB information.
    smoothed = cv2.GaussianBlur(warped_bgr, (3,3), 0)
    answers = {}
    for q_no, options in grid.items():
        metrics = {opt: _bubble_metrics(smoothed, center, radius) for opt, center in options.items()}
        scores = {opt: metrics[opt][0] for opt in OPTIONS}
        inks = {opt: metrics[opt][1] for opt in OPTIONS}
        ordered = sorted(OPTIONS, key=lambda o: scores[o], reverse=True)
        best, second = ordered[0], ordered[1]
        best_score, second_score = scores[best], scores[second]

        # Candidate needs both meaningful local contrast and some ink. This
        # rejects rows where every bubble is only carrying printed text/outline.
        candidates = [o for o in OPTIONS if scores[o] >= FILL_SCORE_THRESHOLD and inks[o] >= MIN_STRONG_INK_FRACTION]

        if not candidates:
            answers[q_no] = None
        elif len(candidates) >= 2:
            # A second real mark must be reasonably strong. If all four are
            # nearly identical, treat the row as blank rather than MULTI.
            strong = [o for o in candidates if scores[o] >= MULTI_SECOND_SCORE]
            if len(strong) >= 2 and best_score >= STRONG_FILL_SCORE:
                answers[q_no] = "MULTI"
            else:
                answers[q_no] = best
        else:
            # One candidate: require either strong evidence or a clear margin
            # over the next option. This helps with very light marks.
            margin = best_score - second_score
            if best_score >= STRONG_FILL_SCORE or (best_score >= FILL_SCORE_THRESHOLD and margin >= 7.0):
                answers[q_no] = best
            else:
                answers[q_no] = None
    return answers


def score_answers(student_answers, key_string, negative_marking=False, negative_value=0.0):
    total = len(key_string)
    correct = answered = 0
    wrong, wrong_details, skipped_questions = [], {}, []
    for i in range(total):
        q = i+1
        correct_ans = key_string[i].upper()
        given = student_answers.get(q)
        if given is None:
            skipped_questions.append(q); continue
        answered += 1
        if given == "MULTI":
            wrong.append(q); wrong_details[q] = {"given":"Multiple","correct":correct_ans}
        elif given == correct_ans:
            correct += 1
        else:
            wrong.append(q); wrong_details[q] = {"given":given,"correct":correct_ans}
    wrong_count = len(wrong)
    penalty = wrong_count*negative_value if negative_marking else 0
    marks = round(correct-penalty,2)
    accuracy = round(correct/answered*100,2) if answered else 0.0
    return {"total":total,"answered":answered,"skipped":total-answered,"correct":correct,"wrong_count":wrong_count,"wrong":wrong,"wrong_details":wrong_details,"skipped_questions":skipped_questions,"accuracy":accuracy,"marks":marks,"negative_marking":negative_marking,"negative_value":negative_value}


def build_review_rows(student_answers, key_string):
    rows=[]
    for i, correct_ans in enumerate(key_string):
        q=i+1; given=student_answers.get(q); ca=correct_ans.upper()
        status="skipped" if given is None else ("wrong" if given=="MULTI" or given!=ca else "correct")
        rows.append({"q":q,"given":given,"correct":ca,"status":status})
    return rows


def render_sheet_image(grid, total_questions=100, answers=None):
    from PIL import Image as PILImage, ImageDraw
    answers=answers or {}; img=PILImage.new("RGB",(WARP_WIDTH,WARP_HEIGHT),"white"); draw=ImageDraw.Draw(img); r=BUBBLE_SAMPLE_RADIUS+6
    for q in range(1,total_questions+1):
        opts=grid.get(q)
        if not opts: continue
        for opt in OPTIONS:
            x,y=opts[opt]; filled=answers.get(q)==opt
            draw.ellipse([x-r,y-r,x+r,y+r],outline=(30,30,30),width=2,fill=(20,20,20) if filled else (255,255,255))
            draw.text((x-4,y-6),opt,fill=(255,255,255) if filled else (30,30,30))
        ax,ay=opts["A"]; draw.text((ax-44,ay-7),str(q),fill=(0,0,0))
    return img


def find_clicked_bubble(grid,total_questions,x,y,radius=None):
    radius=BUBBLE_SAMPLE_RADIUS+10 if radius is None else radius
    best=None; best_d=radius
    for q in range(1,total_questions+1):
        opts=grid.get(q)
        if not opts: continue
        for opt,(bx,by) in opts.items():
            d=((x-bx)**2+(y-by)**2)**0.5
            if d<=best_d: best_d=d; best=(q,opt)
    return best
