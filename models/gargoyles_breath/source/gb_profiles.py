"""Reference measurements for The Gargoyle's Breath flintlock.

All points were measured on the right-side (lock-side) orthographic product
view of the character reference guide (page 12) and are stored in that
image's pixel space ("ref px", 1920x1080 page).  `to_mm` converts them into
model space:

    X  forward along the bore (+X = muzzle)
    Y  lateral (+Y = left side of the gun, -Y = lock side)
    Z  up
    origin on the bore axis at the breech face, units = millimetres.

Scale: 0.73 mm per ref px, which gives a 450 mm overall length.
"""
import numpy as np

K = 0.73          # mm per reference pixel
X0 = 415.0        # breech face (ref px)
Z0 = 468.5        # bore axis (ref px)


def to_mm(pts):
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    return np.stack([(p[:, 0] - X0) * K, (Z0 - p[:, 1]) * K], axis=1)


def from_mm(pts):
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    return np.stack([p[:, 0] / K + X0, Z0 - p[:, 1] / K], axis=1)


def left_view_to_right(pts):
    """Convert points measured on the left-side view (p4) into p3 ref px."""
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    return np.stack([1150.0 - p[:, 0], p[:, 1] - 164.0], axis=1)


def catmull_rom(pts, samples_per_seg=8, closed=False):
    """Centripetal Catmull-Rom spline through pts (N,2|3)."""
    P = np.asarray(pts, dtype=float)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]

        def tj(ti, a, b):
            return ti + max(np.linalg.norm(b - a), 1e-6) ** 0.5
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        ts = np.linspace(t1, t2, samples_per_seg, endpoint=False)[:, None]
        a1 = (t1 - ts) / (t1 - t0) * p0 + (ts - t0) / (t1 - t0) * p1
        a2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        a3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        b1 = (t2 - ts) / (t2 - t0) * a1 + (ts - t0) / (t2 - t0) * a2
        b2 = (t3 - ts) / (t3 - t1) * a2 + (ts - t1) / (t3 - t1) * a3
        out.append((t2 - ts) / (t2 - t1) * b1 + (ts - t1) / (t2 - t1) * b2)
    out = np.vstack(out)
    if not closed:
        out = np.vstack([out, P[-2]])
    return out


# ---------------------------------------------------------------- barrel ---
# Octagonal barrel, across-flats 31.4 mm, breech X=0 -> ring -> bell muzzle.
BARREL_INRADIUS = 15.7
# (X mm, outer radius mm) lathe profile for the ring + bell + muzzle mouldings
MUZZLE_PROFILE = [
    (194.0, 16.1), (195.0, 17.4), (196.2, 17.9), (197.6, 17.9), (198.2, 17.2),
    (198.8, 17.2), (199.4, 17.9), (201.0, 17.9), (201.8, 17.4),
    (203.0, 17.6), (212.0, 18.4), (222.0, 19.5), (232.0, 20.9), (242.0, 22.4),
    (250.0, 23.4), (252.0, 23.6), (252.6, 24.4), (255.0, 24.5), (255.6, 23.8),
    (256.4, 23.8), (257.0, 24.5), (259.4, 24.5), (260.0, 24.0),
    (261.0, 24.2), (264.0, 24.6), (266.5, 24.4), (267.6, 23.6), (268.0, 22.4),
]
BORE_RADIUS_BREECH = 9.6
BORE_RADIUS_MUZZLE = 19.2   # trumpet bore at the mouth
MUZZLE_X = 268.0

# ---------------------------------------------------------------- stock ----
# Side silhouette of the walnut stock (ref px), traced clockwise starting at
# the top front corner of the fore-end.  The butt end continues under the cap.
STOCK_TOP_FOREEND = [(715.5, 469.6), (415.5, 469.6)]
STOCK_WRIST_TOP = [
    (415.5, 447.0), (400, 447.0), (380, 447.4), (365, 449.4), (350, 452.0),
    (335, 455.8), (320, 460.2), (308, 464.4), (300, 467.4), (290, 472.4),
    (280, 478.2), (270, 484.6), (260, 492.0), (250, 500.6), (240, 509.8),
    (232, 517.6), (225, 525.2), (218, 532.8), (212, 540.2), (206, 548.5),
    (201, 557.0),
]
STOCK_BUTT_END = [(201, 557.0), (214, 580.0), (228, 600.0), (243, 616.0),
                  (256, 626.0)]
STOCK_GRIP_FRONT = [
    (256, 626.0), (259.5, 619.0), (263.5, 607.0), (267.5, 595.0),
    (271.5, 583.0), (276, 571.0), (281, 560.5), (287, 550.5), (295, 541.4),
    (305, 534.2), (320, 529.2), (335, 526.2), (370, 524.0), (405, 522.0),
]
STOCK_BOTTOM = [
    (405, 522.0), (450, 520.6), (500, 519.6), (556, 519.0), (566, 518.2),
    (576, 516.0), (584, 512.6), (590, 509.4), (596, 506.6), (604, 505.2),
    (650, 505.0), (698, 505.0), (706, 503.0), (711.5, 498.5), (714.8, 491.0),
    (716.0, 481.0), (716.0, 474.0), (715.5, 469.6),
]

# half-width control points (X mm, Z mm, half width mm, edge radius mm)
STOCK_WIDTH_CTRL = [
    # fore-end (flat sided, rounded underside)
    (215, -12, 17.9, 4.0), (190, -12, 18.1, 4.5), (150, -12, 18.3, 5.0),
    (125, -20, 18.8, 5.0),
    # lock panels
    (100, -18, 20.4, 3.2), (60, -18, 20.6, 3.0), (20, -18, 20.6, 3.0),
    (-20, -16, 20.4, 3.2), (-45, -16, 19.6, 4.0),
    # wrist
    (-35, 6, 16.4, 7.0), (-65, -10, 16.2, 9.5), (-60, -32, 17.0, 9.0),
    (-85, -25, 16.6, 11.5), (-90, -45, 17.4, 11.0),
    # grip / butt
    (-105, -50, 18.4, 12.5), (-115, -70, 19.6, 13.5), (-125, -85, 20.6, 14.0),
    (-135, -95, 21.2, 14.5), (-140, -70, 20.4, 13.0),
]

# ------------------------------------------------------------- butt cap ----
# Visible boundary between wood and the silver gargoyle cap (ref px).
CAP_BOUNDARY = [(213, 531), (222, 553), (231, 573), (241, 589), (252, 604),
                (258, 612)]
# Silhouette of the gargoyle cap (ref px), clockwise from the rear boundary.
CAP_OUTLINE = [
    (213, 531), (205, 537), (196, 545), (187, 553), (180, 562), (174, 572),
    (170, 583), (167, 595), (165, 607), (165, 620), (167, 632), (171, 645),
    (178, 657), (186, 667), (195, 676), (204, 684), (212, 689), (222, 690.5),
    (230, 688), (236, 683), (238.5, 676), (246, 670.5), (238, 667),
    (246, 665.5), (255, 660), (266, 655), (262, 650), (255, 646.5),
    (249, 640), (246.5, 630), (247.5, 620), (252, 610), (258, 612),
]
CAP_EYE = (207.0, 615.0)
CAP_EAR_TIP = (225.0, 582.0)
CAP_SNOUT_TIP = (222.0, 690.5)
CAP_JAW_TIP = (266.0, 655.0)
CAP_FANG_TIP = (246.0, 670.5)

# --------------------------------------------------------------- lock ------
LOCK_PLATE = [
    (322, 493.0), (330, 486.5), (345, 478.5), (365, 472.5), (390, 469.6),
    (440, 468.6), (500, 469.8), (530, 474.0), (548, 482.0), (556.5, 491.5),
    (557, 497.0), (551, 503.0), (530, 507.4), (480, 509.0), (420, 510.0),
    (370, 508.0), (345, 503.0), (330, 498.0),
]
LOCK_SCREW_REAR = (339.0, 489.0, 7.6)        # x, y, diameter (ref px)
FRIZZEN_SPRING_SCREW = (495.0, 497.0, 9.0)
FRIZZEN_PIVOT = (519.0, 473.0, 6.5)
COCK_PIVOT = (397.0, 483.0, 22.0)

# Cock (hammer) side profile pieces, ref px.
COCK_BODY = [
    (397, 456), (388, 458), (382, 466), (379.5, 476), (380, 489), (385, 499),
    (393, 505), (405, 507), (417, 504), (425, 498), (430, 489), (431, 478),
    (428, 467), (422, 459), (414, 455), (405, 454.5),
]
COCK_NECK_SPINE = [  # centreline of the swan neck, from body up to lower jaw
    (418, 462), (414, 452), (407, 446), (398, 441.5), (389, 436.5),
    (382, 430), (380, 422), (382, 415.5), (386, 412.5),
]
COCK_NECK_WIDTH = [8.5, 9.0, 9.0, 8.8, 8.6, 8.4, 8.2, 8.0, 7.6]   # ref px
COCK_LOWER_JAW = [(377, 409.5), (420.5, 411.2), (421.0, 415.6), (377, 414.8)]
COCK_TOP_JAW = [(379, 384.0), (402, 389.6), (425.5, 396.6), (425.0, 399.2),
                (379, 398.6)]
COCK_COMB = [(378.5, 380.0), (379.5, 409.0)]     # post centreline
COCK_COMB_BALL = (377.6, 378.6, 7.4)
COCK_JAW_SCREW = [(394.5, 361.0), (394.5, 428.0)]   # top of finial -> tail
FLINT = [(402, 400.0), (433.5, 406.8), (402, 410.5)]

# Frizzen (shown open): vertical pan-cover leg + forward-swept face.
FRIZZEN_SPINE = [(515.5, 471.0), (513.0, 452.0), (511.0, 432.0), (510.5, 422.0),
                 (513.5, 417.4), (522.0, 416.0), (535.0, 413.4), (548.0, 410.0),
                 (557.0, 407.4)]
FRIZZEN_THICK = [6.6, 6.0, 5.4, 5.0, 4.6, 4.0, 3.8, 3.6, 3.4]          # ref px

# Pan with fence (ref px, side profile)
PAN = [(447, 466.0), (450, 459.0), (458, 456.6), (505, 458.6), (511, 462.0),
       (511, 470.0), (447, 470.0)]

# ---------------------------------------------------------- trigger guard --
GUARD_BOW = [  # centreline of the bow, front attach -> bottom -> rear attach
    (407.5, 522.5), (408.5, 532.0), (406.0, 546.0), (399.0, 560.0),
    (390.0, 571.0), (378.0, 577.0), (365.0, 576.0), (352.0, 568.0),
    (343.0, 555.0), (338.0, 541.0), (336.5, 528.0),
]
GUARD_WIRE = 4.6   # ref px, visible thickness of the bow
TRIGGER = [(370.5, 523.5), (368.6, 536.0), (368.0, 549.0), (369.2, 560.0),
           (372.0, 568.0), (374.6, 571.0)]
TRIGGER_WIDTH = [5.6, 5.2, 4.8, 4.4, 4.0, 3.2]
GUARD_FRONT_TANG = [(407.5, 522.5), (450, 521.2), (500, 520.4), (548, 519.8),
                    (557, 516.5)]
GRIP_STRAP = [(336.5, 527.0), (320, 530.0), (305, 535.0), (295, 542.0),
              (287, 551.0), (281, 561.0), (276, 572.0), (271.5, 584.0),
              (267.5, 596.0), (263.5, 608.0), (259.5, 620.0), (253.5, 631.0)]

# ---------------------------------------------------------------- ramrod ---
RAMROD_Z_PX = 512.6
RAMROD_DIAM_PX = 7.4
RAMROD_TIP_X_PX = 740.0
RAMROD_HEAD = [(722, 3.7), (730, 4.6), (735, 6.6), (738.5, 8.8), (740, 8.6)]
RAMROD_PIPES = [(589.5, 619.5), (667.0, 700.5)]   # x ranges (ref px)
PIPE_DIAM_PX = 12.6

# ------------------------------------------------------------ side plate ---
# Left side: dragon side plate (measured on the left view, converted).
SIDE_SCREW_FRONT = left_view_to_right([(614.0, 668.0)])[0]
SIDE_SCREW_REAR = left_view_to_right([(770.0, 647.5)])[0]   # in the loop
