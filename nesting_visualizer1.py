import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Point, MultiPolygon, GeometryCollection
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box
from shapely.ops import unary_union


# ============================================================
# CONFIG
# ============================================================
FAST_PAIR_GRID_STEPS = 9
FAST_CIRCLE_PHASE_SAMPLES = 10
FAST_MAX_CANDIDATES = 12
FAST_LOOKAHEAD_DEPTH = 4
FAST_MAX_ROUNDS = 8
EPS = 1e-6


# ============================================================
# EXCEPTIONS
# ============================================================
class NestingFailed(Exception):
    def __init__(self, message, best_layout=None, best_count=0, expected_count=0):
        super().__init__(message)
        self.best_layout = best_layout or []
        self.best_count = best_count
        self.expected_count = expected_count


# ============================================================
# JSON / GEOMETRY
# ============================================================
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def part_to_polygon(part):
    pts = part["contours"][0]["points"]

    # circle
    if len(pts) == 1 and pts[0].get("radius", 0) > 0:
        r = pts[0]["radius"]
        cx, cy = pts[0]["x"], pts[0]["y"]
        angle = np.linspace(0, 2 * np.pi, 60)
        return ShapelyPolygon([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angle])

    return ShapelyPolygon([(p["x"], p["y"]) for p in pts])


def poly_size(poly):
    minx, miny, maxx, maxy = poly.bounds
    return maxx - minx, maxy - miny


def bounding_area(poly):
    minx, miny, maxx, maxy = poly.bounds
    return (maxx - minx) * (maxy - miny)


def normalize_poly(poly):
    minx, miny, _, _ = poly.bounds
    return translate(poly, xoff=-minx, yoff=-miny)


def rotate_normalize(poly, angle):
    return normalize_poly(rotate(poly, angle, origin="centroid", use_radians=False))


def get_rotated_variant(item, angle):
    cache = item.setdefault("_rot_cache", {})
    if angle not in cache:
        cache[angle] = rotate_normalize(item["poly"], angle)
    return cache[angle]


# ============================================================
# SHAPE DETECTION
# ============================================================
def edge_vector(p1, p2):
    return p2[0] - p1[0], p2[1] - p1[1]


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def is_parallel(v1, v2, tol=1e-4):
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])

    if n1 < 1e-12 or n2 < 1e-12:
        return False

    return abs(cross) / (n1 * n2) < tol


def is_circle_json_part(part):
    pts = part["contours"][0]["points"]
    return len(pts) == 1 and pts[0].get("radius", 0) > 0


def is_rectangle(poly, tol=1e-6):
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) != 4:
        return False

    for i in range(4):
        p0 = coords[i]
        p1 = coords[(i + 1) % 4]
        p2 = coords[(i + 2) % 4]
        v1 = edge_vector(p0, p1)
        v2 = edge_vector(p1, p2)
        if abs(dot(v1, v2)) > tol:
            return False

    return True


def is_chamfered_rectangle(poly, area_ratio_tol=0.90):
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) not in (6, 8):
        return False

    mrr = poly.minimum_rotated_rectangle
    if mrr.area <= 1e-9:
        return False

    ratio = poly.area / mrr.area
    return ratio >= area_ratio_tol


def is_trapezoid(poly, tol=1e-4):
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) != 4:
        return False

    if is_rectangle(poly, tol):
        return False

    v0 = edge_vector(coords[0], coords[1])
    v1 = edge_vector(coords[1], coords[2])
    v2 = edge_vector(coords[2], coords[3])
    v3 = edge_vector(coords[3], coords[0])

    return is_parallel(v0, v2, tol) or is_parallel(v1, v3, tol)


def detect_shape_type(part, poly):
    if is_circle_json_part(part):
        return "circle"

    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)

    if is_rectangle(poly) or is_chamfered_rectangle(poly):
        return "rectangle"
    if is_trapezoid(poly):
        return "trapezoid"
    if n == 5:
        return "pentagon"
    if n == 6:
        return "hexagon"
    return f"polygon_{n}"


def shape_priority(shape_type):
    order = {
        "rectangle": 0,
        "trapezoid": 1,
        "pentagon": 2,
        "hexagon": 3,
        "circle": 99,
    }
    return order.get(shape_type, 50)


# ============================================================
# GENERAL HELPERS
# ============================================================
def expected_part_count(parts):
    return sum(int(p.get("quantity", 1)) for p in parts)


def placed_part_count(placed_parts):
    return sum(int(p.get("unit_count", 1)) for p in placed_parts)


def get_part_key(p):
    return p.get("base_id", p.get("id", "unknown"))


def placed_base_counts(placed_parts):
    counts = {}
    for p in placed_parts:
        key = get_part_key(p)
        counts[key] = counts.get(key, 0) + int(p.get("unit_count", 1))
    return counts


def group_placed_parts_by_base(placed_parts):
    groups = {}
    for p in placed_parts:
        groups.setdefault(get_part_key(p), []).append(p)
    return groups


def cluster_bounds(placed_parts, fallback_poly):
    if not placed_parts:
        return fallback_poly.bounds

    minx = min(p["poly"].bounds[0] for p in placed_parts)
    miny = min(p["poly"].bounds[1] for p in placed_parts)
    maxx = max(p["poly"].bounds[2] for p in placed_parts)
    maxy = max(p["poly"].bounds[3] for p in placed_parts)
    return minx, miny, maxx, maxy


def overall_bbox_area(placed_parts):
    if not placed_parts:
        return 0.0

    u = unary_union([p["poly"] for p in placed_parts]).buffer(0)
    if u.is_empty:
        return 0.0

    minx, miny, maxx, maxy = u.bounds
    return (maxx - minx) * (maxy - miny)


def layout_fill_ratio(placed_parts):
    if not placed_parts:
        return 0.0

    u = unary_union([p["poly"] for p in placed_parts]).buffer(0)
    if u.is_empty:
        return 0.0

    minx, miny, maxx, maxy = u.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    if bbox_area <= 1e-9:
        return 0.0

    return u.area / bbox_area


def layout_rank_key(placed_parts):
    return (
        placed_part_count(placed_parts),
        layout_fill_ratio(placed_parts),
        -overall_bbox_area(placed_parts),
    )


def _clean_geom(g):
    if g.is_empty:
        return g
    return g.buffer(0)


def _iter_polygons(g):
    if g.is_empty:
        return []

    if isinstance(g, ShapelyPolygon):
        return [g]

    if isinstance(g, MultiPolygon):
        return [p for p in g.geoms if not p.is_empty]

    if isinstance(g, GeometryCollection):
        polys = []
        for sub in g.geoms:
            polys.extend(_iter_polygons(sub))
        return polys

    return []


def get_polygon_vertices(poly):
    if isinstance(poly, MultiPolygon):
        verts = []
        for g in poly.geoms:
            verts.extend(list(g.exterior.coords)[:-1])
        return verts

    if isinstance(poly, ShapelyPolygon):
        return list(poly.exterior.coords)[:-1]

    return []


# ============================================================
# TRAPEZOID PAIRING
# ============================================================
def combine_trapezoids(poly):
    """
    Build a compact pair from 2 trapezoids.
    Only a heuristic.
    """
    best_area = float("inf")
    best_union = None

    rotations = [0, 180, 90, 270]
    w, h = poly_size(normalize_poly(poly))
    x_vals = np.linspace(-2 * w, 2 * w, FAST_PAIR_GRID_STEPS)
    y_vals = np.linspace(-2 * h, 2 * h, FAST_PAIR_GRID_STEPS)

    for angle1 in rotations:
        for angle2 in rotations:
            p1 = rotate(poly, angle1, origin=(0, 0), use_radians=False)
            p2 = rotate(poly, angle2, origin=(0, 0), use_radians=False)

            for dx in x_vals:
                for dy in y_vals:
                    p2_moved = translate(p2, xoff=float(dx), yoff=float(dy))
                    if p1.intersection(p2_moved).area > EPS:
                        continue

                    combined = p1.union(p2_moved)
                    area = bounding_area(combined)
                    if area < best_area:
                        best_area = area
                        best_union = combined

    return best_union


# ============================================================
# PART LIST BUILDING
# ============================================================
def build_parts_list(parts):
    parts_list = []

    all_non_circles = []
    trap_base_ids = set()

    for part in parts:
        poly = part_to_polygon(part)
        shape_type = detect_shape_type(part, poly)

        if shape_type != "circle":
            all_non_circles.append(shape_type)
        if shape_type == "trapezoid":
            trap_base_ids.add(str(part["id"]))

    pure_trapezoid_strip_case = (
        len(all_non_circles) > 0
        and all(s == "trapezoid" for s in all_non_circles)
        and len(trap_base_ids) == 1
    )

    for part in parts:
        poly = part_to_polygon(part)
        qty = int(part.get("quantity", 1))
        pid = str(part["id"])
        shape_type = detect_shape_type(part, poly)

        if shape_type != "trapezoid":
            for _ in range(qty):
                parts_list.append(
                    {
                        "id": pid,
                        "base_id": pid,
                        "poly": poly,
                        "shape": shape_type,
                        "priority": shape_priority(shape_type),
                        "area": bounding_area(poly),
                        "unit_count": 1,
                    }
                )
            continue

        # all-trapezoid-strip case -> keep singles
        if pure_trapezoid_strip_case:
            for _ in range(qty):
                parts_list.append(
                    {
                        "id": pid,
                        "base_id": pid,
                        "poly": poly,
                        "shape": "trapezoid",
                        "priority": shape_priority("trapezoid"),
                        "area": bounding_area(poly),
                        "unit_count": 1,
                    }
                )
            continue

        # generic case -> pair qty//2 + single remainder
        pair_poly = combine_trapezoids(poly)
        pair_count = qty // 2
        remainder = qty % 2

        if pair_poly is not None:
            for _ in range(pair_count):
                parts_list.append(
                    {
                        "id": pid + "_pair",
                        "base_id": pid,
                        "poly": pair_poly,
                        "shape": "trapezoid",
                        "priority": shape_priority("trapezoid"),
                        "area": bounding_area(pair_poly),
                        "unit_count": 2,
                    }
                )

            if remainder == 1:
                parts_list.append(
                    {
                        "id": pid,
                        "base_id": pid,
                        "poly": poly,
                        "shape": "trapezoid",
                        "priority": shape_priority("trapezoid"),
                        "area": bounding_area(poly),
                        "unit_count": 1,
                    }
                )
        else:
            for _ in range(qty):
                parts_list.append(
                    {
                        "id": pid,
                        "base_id": pid,
                        "poly": poly,
                        "shape": "trapezoid",
                        "priority": shape_priority("trapezoid"),
                        "area": bounding_area(poly),
                        "unit_count": 1,
                    }
                )

    parts_list.sort(key=lambda p: (p["priority"], -p["area"]))
    return parts_list


def build_groups_by_total_square(parts_list):
    groups = {}
    circles = []

    for item in parts_list:
        if item["shape"] == "circle":
            circles.append(item)
            continue

        key = item["base_id"]
        if key not in groups:
            groups[key] = {
                "base_id": key,
                "shape": item["shape"],
                "items": [],
                "total_square": 0.0,
            }

        groups[key]["items"].append(item)
        groups[key]["total_square"] += item["area"]

    grouped = list(groups.values())
    grouped.sort(key=lambda g: -g["total_square"])
    return grouped, circles


# ============================================================
# COLLISION / VALIDITY
# ============================================================
def has_real_overlap(candidate, placed_parts, eps=EPS):
    for p in placed_parts:
        if candidate.intersection(p["poly"]).area > eps:
            return True
    return False


def valid_candidate(candidate, placed_parts, plate_poly):
    return plate_poly.covers(candidate) and not has_real_overlap(candidate, placed_parts)


# ============================================================
# CANDIDATE GENERATION
# ============================================================
def generate_candidate_positions_for_part(rotated_poly, placed_parts, plate_poly):
    rw, rh = poly_size(rotated_poly)
    minx, miny, maxx, maxy = plate_poly.bounds

    positions = {
        (minx, miny),
        (minx, maxy - rh),
        (maxx - rw, miny),
        (maxx - rw, maxy - rh),
    }

    # bbox-touch positions from placed parts
    for p in placed_parts:
        pxmin, pymin, pxmax, pymax = p["poly"].bounds

        positions.add((pxmax, pymin))
        positions.add((pxmax, pymax - rh))

        positions.add((pxmin - rw, pymin))
        positions.add((pxmin - rw, pymax - rh))

        positions.add((pxmin, pymax))
        positions.add((pxmax - rw, pymax))

        positions.add((pxmin, pymin - rh))
        positions.add((pxmax - rw, pymin - rh))

    # vertex-align positions for irregular shapes
    cand_vertices = get_polygon_vertices(rotated_poly)
    plate_pts = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]

    for ax, ay in plate_pts:
        for vx, vy in cand_vertices:
            positions.add((ax - vx, ay - vy))

    for p in placed_parts:
        for ex, ey in get_polygon_vertices(p["poly"]):
            for vx, vy in cand_vertices:
                positions.add((ex - vx, ey - vy))

    # free-region bounds corners
    if placed_parts:
        occupied = unary_union([p["poly"] for p in placed_parts]).buffer(0)
        free = _clean_geom(plate_poly.difference(occupied))
        for rg in _iter_polygons(free):
            fminx, fminy, fmaxx, fmaxy = rg.bounds
            positions.add((fminx, fminy))
            positions.add((fminx, fmaxy - rh))
            positions.add((fmaxx - rw, fminy))
            positions.add((fmaxx - rw, fmaxy - rh))

    out = sorted({(round(x, 6), round(y, 6)) for x, y in positions})
    return out


def orientation_profile(item, target="horizontal"):
    best = None

    for angle in [0, 90, 180, 270]:
        rotated = get_rotated_variant(item, angle)
        w, h = poly_size(rotated)

        if target == "horizontal":
            score = (w / max(h, 1e-9), w, -h)
        else:
            score = (h / max(w, 1e-9), h, -w)

        if best is None or score > best["score"]:
            best = {
                "angle": angle,
                "poly": rotated,
                "w": w,
                "h": h,
                "score": score,
            }

    return best


def enumerate_candidate_placements(
    item,
    placed_parts,
    plate_poly,
    mode="bottom_left",
    preferred_angles=None,
    ref_bounds=None,
    max_candidates=FAST_MAX_CANDIDATES,
):
    if preferred_angles is None:
        preferred_angles = [0, 90, 180, 270]

    _, _, _, plate_top = plate_poly.bounds
    ref_right = ref_bounds[2] if ref_bounds is not None else None

    candidates = []
    seen = set()

    for angle in preferred_angles:
        rotated = get_rotated_variant(item, angle)
        rw, rh = poly_size(rotated)

        for x, y in generate_candidate_positions_for_part(rotated, placed_parts, plate_poly):
            key = (round(x, 6), round(y, 6), angle)
            if key in seen:
                continue
            seen.add(key)

            candidate_poly = translate(rotated, xoff=x, yoff=y)
            if not valid_candidate(candidate_poly, placed_parts, plate_poly):
                continue

            minx, miny, maxx, maxy = candidate_poly.bounds

            if mode == "top_left":
                local_score = (plate_top - maxy, minx, rw * rh)
            elif mode == "near_vertical":
                penalty_left_of_ref = 0 if ref_right is None or minx >= ref_right - 1e-6 else 1
                local_score = (
                    penalty_left_of_ref,
                    abs(minx - ref_right) if ref_right is not None else 0.0,
                    miny,
                    rw * rh,
                )
            else:
                local_score = (miny, minx, rw * rh)

            candidates.append(
                {
                    "id": item["id"],
                    "base_id": item.get("base_id", item["id"]),
                    "unit_count": item.get("unit_count", 1),
                    "poly": candidate_poly,
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                    "local_score": local_score,
                    "mode": mode,
                }
            )

    candidates.sort(key=lambda c: c["local_score"])
    return candidates[:max_candidates]


def candidate_modes_for_item(item, placed_parts, plate_poly, strategy=None):
    strategy = strategy or {}
    ref_bounds = cluster_bounds(placed_parts, plate_poly)

    if item["shape"] == "rectangle":
        horiz = orientation_profile(item, "horizontal")["angle"]
        vert = orientation_profile(item, "vertical")["angle"]

        if strategy.get("rect_mode") == "vertical_first":
            return [
                ("bottom_left", [vert, horiz, 0, 90, 180, 270], ref_bounds),
                ("near_vertical", [vert, 90, 270, 0, 180], ref_bounds),
                ("top_left", [horiz, 0, 180, 90, 270], ref_bounds),
            ]

        return [
            ("top_left", [horiz, 0, 180, 90, 270], ref_bounds),
            ("near_vertical", [vert, 90, 270, 0, 180], ref_bounds),
            ("bottom_left", [vert, horiz, 0, 90, 180, 270], ref_bounds),
        ]

    if item["shape"] == "trapezoid":
        return [
            ("bottom_left", [0, 180, 90, 270], ref_bounds),
        ]

    return [
        ("bottom_left", [0, 90, 180, 270], ref_bounds),
        ("top_left", [0, 90, 180, 270], ref_bounds),
    ]


def place_item_bottom_left(item, placed_parts, plate_poly, preferred_angles=None):
    if preferred_angles is None:
        preferred_angles = [0, 90, 180, 270]

    cands = enumerate_candidate_placements(
        item=item,
        placed_parts=placed_parts,
        plate_poly=plate_poly,
        mode="bottom_left",
        preferred_angles=preferred_angles,
        max_candidates=1,
    )
    return cands[0] if cands else None


def place_item_top_left(item, placed_parts, plate_poly, preferred_angles=None):
    if preferred_angles is None:
        preferred_angles = [0, 90, 180, 270]

    cands = enumerate_candidate_placements(
        item=item,
        placed_parts=placed_parts,
        plate_poly=plate_poly,
        mode="top_left",
        preferred_angles=preferred_angles,
        max_candidates=1,
    )
    return cands[0] if cands else None


def place_item_near_reference_vertical(item, placed_parts, plate_poly, ref_bounds):
    cands = enumerate_candidate_placements(
        item=item,
        placed_parts=placed_parts,
        plate_poly=plate_poly,
        mode="near_vertical",
        preferred_angles=[90, 270, 0, 180],
        ref_bounds=ref_bounds,
        max_candidates=1,
    )
    return cands[0] if cands else None


def place_item_generic(item, placed_parts, plate_poly):
    if item["shape"] == "trapezoid":
        return place_item_bottom_left(item, placed_parts, plate_poly)
    return place_item_top_left(item, placed_parts, plate_poly)


# ============================================================
# CIRCLE PACKING
# ============================================================
def get_circle_template(parts):
    for part in parts:
        if is_circle_json_part(part):
            poly = part_to_polygon(part)
            minx, miny, maxx, maxy = poly.bounds
            diameter = max(maxx - minx, maxy - miny)
            radius = diameter / 2.0
            poly_norm = normalize_poly(poly)
            return poly_norm, radius, diameter, str(part["id"])
    return None, None, None, None


def make_circle_at(circle_poly_norm, radius, cx, cy):
    return translate(circle_poly_norm, xoff=cx - radius, yoff=cy - radius)


def sort_centers_by_anchor(centers, anchor):
    if anchor == "bl":
        return sorted(centers, key=lambda c: (c[1], c[0]))
    if anchor == "br":
        return sorted(centers, key=lambda c: (c[1], -c[0]))
    if anchor == "tl":
        return sorted(centers, key=lambda c: (-c[1], c[0]))
    if anchor == "tr":
        return sorted(centers, key=lambda c: (-c[1], -c[0]))
    return sorted(centers, key=lambda c: (c[1], c[0]))


def get_circle_center_region(plate_poly, placed_parts, radius, eps=1e-7):
    usable_plate = _clean_geom(plate_poly.buffer(-(radius - eps)))
    if usable_plate.is_empty:
        return usable_plate

    if not placed_parts:
        return usable_plate

    occupied = unary_union([p["poly"] for p in placed_parts])
    occupied = _clean_geom(occupied)

    blocked = _clean_geom(occupied.buffer(radius - eps))
    center_region = _clean_geom(usable_plate.difference(blocked))
    return center_region


def generate_circle_centers_in_region(region_poly, radius, mode="hex", anchor="bl", x_phase=0.0, y_phase=0.0):
    minx, miny, maxx, maxy = region_poly.bounds
    centers = []
    eps = 1e-9

    if mode == "hex":
        step_x = 2.0 * radius
        step_y = math.sqrt(3.0) * radius

        row = 0
        y = miny + y_phase
        while y <= maxy + eps:
            row_shift = radius if row % 2 == 1 else 0.0
            x = minx + x_phase + row_shift

            while x <= maxx + eps:
                pt = Point(x, y)
                if region_poly.covers(pt):
                    centers.append((x, y))
                x += step_x

            y += step_y
            row += 1

    else:
        step = 2.0 * radius
        y = miny + y_phase
        while y <= maxy + eps:
            x = minx + x_phase
            while x <= maxx + eps:
                pt = Point(x, y)
                if region_poly.covers(pt):
                    centers.append((x, y))
                x += step
            y += step

    return sort_centers_by_anchor(centers, anchor)


def place_circles_best_pattern(parts, plate_poly, placed_parts, circle_count):
    circle_poly_norm, radius, diameter, circle_pid = get_circle_template(parts)
    if circle_poly_norm is None or circle_count <= 0:
        return placed_parts, 0

    center_region = get_circle_center_region(plate_poly, placed_parts, radius)
    if center_region.is_empty:
        return placed_parts, 0

    region_polys = sorted(_iter_polygons(center_region), key=lambda p: p.area, reverse=True)
    if not region_polys:
        return placed_parts, 0

    best_layout = list(placed_parts)
    best_count = 0
    best_score = None

    def phase_values(step, n):
        return np.linspace(0.0, step, n, endpoint=False)

    trials = [
        ("hex", "tl"),
        ("hex", "tr"),
        ("hex", "bl"),
        ("hex", "br"),
        ("grid", "tl"),
        ("grid", "tr"),
        ("grid", "bl"),
        ("grid", "br"),
    ]

    samples = FAST_CIRCLE_PHASE_SAMPLES

    for mode, anchor in trials:
        if mode == "hex":
            step_x = 2.0 * radius
            step_y = math.sqrt(3.0) * radius
            x_phases = phase_values(step_x, samples)
            y_phases = phase_values(step_y, samples)
        else:
            step_x = 2.0 * radius
            step_y = 2.0 * radius
            x_phases = phase_values(step_x, samples)
            y_phases = phase_values(step_y, samples)

        for x_phase in x_phases:
            for y_phase in y_phases:
                temp_parts = list(placed_parts)
                placed_now = 0
                placed_centers = []

                for region_poly in region_polys:
                    centers = generate_circle_centers_in_region(
                        region_poly=region_poly,
                        radius=radius,
                        mode=mode,
                        anchor=anchor,
                        x_phase=float(x_phase),
                        y_phase=float(y_phase),
                    )

                    for cx, cy in centers:
                        candidate = make_circle_at(circle_poly_norm, radius, cx, cy)

                        if not plate_poly.covers(candidate):
                            continue
                        if has_real_overlap(candidate, temp_parts):
                            continue

                        temp_parts.append(
                            {
                                "id": circle_pid,
                                "base_id": circle_pid,
                                "unit_count": 1,
                                "poly": candidate,
                                "x": cx - radius,
                                "y": cy - radius,
                                "angle": 0,
                                "shape": "circle",
                            }
                        )
                        placed_centers.append((cx, cy))
                        placed_now += 1

                        if placed_now >= circle_count:
                            break

                    if placed_now >= circle_count:
                        break

                score = (placed_now, len(placed_centers))
                if best_score is None or score > best_score:
                    best_score = score
                    best_count = placed_now
                    best_layout = temp_parts

                if best_count >= circle_count:
                    break

            if best_count >= circle_count:
                break

        if best_count >= circle_count:
            break

    return best_layout, best_count


def get_circle_count(parts):
    return sum(int(p.get("quantity", 1)) for p in parts if is_circle_json_part(p))


# ============================================================
# QUICK SIMULATION / HARD FEASIBILITY
# ============================================================
def quick_place_for_simulation(item, placed_parts, plate_poly):
    ref_bounds = cluster_bounds(placed_parts, plate_poly)
    best = None
    best_local = None

    for mode, angles, rb in candidate_modes_for_item(item, placed_parts, plate_poly):
        rb = ref_bounds if mode == "near_vertical" else rb
        cands = enumerate_candidate_placements(
            item=item,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            mode=mode,
            preferred_angles=angles,
            ref_bounds=rb,
            max_candidates=4,
        )
        for cand in cands:
            if best is None or cand["local_score"] < best_local:
                best = cand
                best_local = cand["local_score"]

    return best


def simulate_remaining_noncircle_fit(placed_parts, remaining_items, plate_poly, max_depth=FAST_LOOKAHEAD_DEPTH):
    temp_parts = list(placed_parts)
    fit_count = 0
    fit_area = 0.0

    test_items = sorted(remaining_items, key=lambda p: -p["area"])[:max_depth]
    for item in test_items:
        cand = quick_place_for_simulation(item, temp_parts, plate_poly)
        if cand is None:
            continue

        temp_parts.append(cand)
        fit_count += int(item.get("unit_count", 1))
        fit_area += item["area"]

    bbox_area = overall_bbox_area(temp_parts) if temp_parts else 0.0
    return fit_count, fit_area, -bbox_area


def choose_candidate_with_feasibility(
    item,
    placed_parts,
    plate_poly,
    all_parts,
    remaining_noncircle_items,
    remaining_circle_count,
    strategy=None,
):
    strategy = strategy or {}
    ref_bounds = cluster_bounds(placed_parts, plate_poly)
    all_cands = []
    seen = set()

    for mode, angles, rb in candidate_modes_for_item(item, placed_parts, plate_poly, strategy=strategy):
        rb = ref_bounds if mode == "near_vertical" else rb
        cands = enumerate_candidate_placements(
            item=item,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            mode=mode,
            preferred_angles=angles,
            ref_bounds=rb,
            max_candidates=FAST_MAX_CANDIDATES,
        )
        for c in cands:
            key = (round(c["x"], 6), round(c["y"], 6), c["angle"])
            if key not in seen:
                seen.add(key)
                all_cands.append(c)

    all_cands.sort(key=lambda c: c["local_score"])
    all_cands = all_cands[:FAST_MAX_CANDIDATES]

    best = None
    best_score = None

    for cand in all_cands:
        # hard circle feasibility check
        if remaining_circle_count > 0 and strategy.get("enforce_circle_feasibility", True):
            _, possible_circles = place_circles_best_pattern(
                parts=all_parts,
                plate_poly=plate_poly,
                placed_parts=placed_parts + [cand],
                circle_count=remaining_circle_count,
            )
            if possible_circles < remaining_circle_count:
                continue

        future_fit_count, future_fit_area, neg_bbox = simulate_remaining_noncircle_fit(
            placed_parts=placed_parts + [cand],
            remaining_items=remaining_noncircle_items,
            plate_poly=plate_poly,
            max_depth=FAST_LOOKAHEAD_DEPTH,
        )

        score = (
            future_fit_count,
            future_fit_area,
            neg_bbox,
            tuple(-v if isinstance(v, (int, float)) else 0 for v in cand["local_score"]),
        )

        if best is None or score > best_score:
            best = cand
            best_score = score

    return best


# ============================================================
# TRAPEZOID STRIP SOLVER
# ============================================================
def _min_dx_no_overlap(poly_left, poly_right, dy, eps=1e-7, max_iter=50):
    w1, _ = poly_size(poly_left)
    w2, _ = poly_size(poly_right)

    lo = 0.0
    hi = max(w1 + w2, 1.0)

    while poly_left.intersection(translate(poly_right, xoff=hi, yoff=dy)).area > eps:
        hi *= 1.5

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        moved = translate(poly_right, xoff=mid, yoff=dy)
        if poly_left.intersection(moved).area > eps:
            lo = mid
        else:
            hi = mid

    return hi


def _trapezoid_lane_configs(base_poly, plate_min_y, plate_height):
    p0 = rotate_normalize(base_poly, 0)
    p180 = rotate_normalize(base_poly, 180)

    h0 = poly_size(p0)[1]
    h180 = poly_size(p180)[1]

    configs = []

    if h0 <= plate_height + 1e-6:
        configs.append([{"name": "b0", "angle": 0, "poly": p0, "y": plate_min_y}])

    if h180 <= plate_height + 1e-6:
        configs.append([{"name": "b180", "angle": 180, "poly": p180, "y": plate_min_y}])

    if h0 <= plate_height + 1e-6 and h180 <= plate_height + 1e-6:
        configs.append(
            [
                {"name": "b0", "angle": 0, "poly": p0, "y": plate_min_y},
                {"name": "t180", "angle": 180, "poly": p180, "y": plate_min_y + plate_height - h180},
            ]
        )
        configs.append(
            [
                {"name": "b180", "angle": 180, "poly": p180, "y": plate_min_y},
                {"name": "t0", "angle": 0, "poly": p0, "y": plate_min_y + plate_height - h0},
            ]
        )

    return configs


def place_identical_trapezoids_global(items, placed_parts, plate_poly, beam_width=80):
    if not items:
        return placed_parts

    minx, miny, maxx, maxy = plate_poly.bounds
    plate_width = maxx - minx
    plate_height = maxy - miny

    base_poly = items[0]["poly"]
    configs = _trapezoid_lane_configs(base_poly, miny, plate_height)

    best_solution = None
    best_count = -1
    best_used_width = float("inf")

    for cfg in configs:
        m = len(cfg)
        widths = [poly_size(t["poly"])[0] for t in cfg]

        sep = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                dy = cfg[j]["y"] - cfg[i]["y"]
                sep[i][j] = _min_dx_no_overlap(cfg[i]["poly"], cfg[j]["poly"], dy)

        states = [
            {
                "frontier": tuple(0.0 for _ in range(m)),
                "used_width": 0.0,
                "placements": [],
                "counts": tuple(0 for _ in range(m)),
            }
        ]

        max_steps = len(items)
        for _ in range(max_steps):
            new_states = []

            for st in states:
                for k in range(m):
                    x_rel = st["frontier"][k]
                    x_abs = minx + x_rel
                    cand_poly = translate(cfg[k]["poly"], xoff=x_abs, yoff=cfg[k]["y"])

                    if not plate_poly.covers(cand_poly):
                        continue
                    if has_real_overlap(cand_poly, placed_parts):
                        continue

                    used_width = max(st["used_width"], x_rel + widths[k])
                    if used_width > plate_width + 1e-6:
                        continue

                    new_frontier = list(st["frontier"])
                    for j in range(m):
                        new_frontier[j] = max(new_frontier[j], x_rel + sep[k][j])

                    new_counts = list(st["counts"])
                    new_counts[k] += 1

                    new_states.append(
                        {
                            "frontier": tuple(new_frontier),
                            "used_width": used_width,
                            "placements": st["placements"] + [(k, x_rel)],
                            "counts": tuple(new_counts),
                        }
                    )

            if not new_states:
                break

            compact = {}
            for st in new_states:
                key = (tuple(round(v, 3) for v in st["frontier"]), st["counts"])
                score = (
                    st["used_width"],
                    sum(st["frontier"]),
                    max(st["counts"]) - min(st["counts"]) if len(st["counts"]) > 1 else 0,
                )
                old = compact.get(key)
                if old is None or score < old[0]:
                    compact[key] = (score, st)

            states = [v[1] for v in compact.values()]
            states.sort(
                key=lambda st: (
                    st["used_width"],
                    sum(st["frontier"]),
                    max(st["counts"]) - min(st["counts"]) if len(st["counts"]) > 1 else 0,
                )
            )
            states = states[:beam_width]

        if states:
            local_best = max(states, key=lambda st: (len(st["placements"]), -st["used_width"]))
            local_count = len(local_best["placements"])
            local_used_width = local_best["used_width"]

            if local_count > best_count or (local_count == best_count and local_used_width < best_used_width):
                best_count = local_count
                best_used_width = local_used_width
                best_solution = (cfg, local_best)

    if best_solution is None:
        for item in items:
            cand = place_item_bottom_left(item, placed_parts, plate_poly)
            if cand is not None:
                placed_parts.append(cand)
        return placed_parts

    cfg, sol = best_solution

    for item, (k, x_rel) in zip(items, sol["placements"]):
        x_abs = minx + x_rel
        candidate = translate(cfg[k]["poly"], xoff=x_abs, yoff=cfg[k]["y"])
        placed_parts.append(
            {
                "id": item["id"],
                "base_id": item.get("base_id", item["id"]),
                "unit_count": item.get("unit_count", 1),
                "poly": candidate,
                "x": x_abs,
                "y": cfg[k]["y"],
                "angle": cfg[k]["angle"],
                "shape": item["shape"],
            }
        )

    return placed_parts


def place_trapezoids_generic(items, placed_parts, plate_poly, all_parts, remaining_circle_count, strategy=None):
    if not items:
        return placed_parts, []

    remaining = list(items)
    unplaced = []

    # if many identical singles, try strip solver
    if len(remaining) >= 6 and len({p["base_id"] for p in remaining}) == 1 and all(int(p.get("unit_count", 1)) == 1 for p in remaining):
        placed_parts = place_identical_trapezoids_global(remaining, placed_parts, plate_poly)
        return placed_parts, []

    progress = True
    while progress and remaining:
        progress = False
        next_remaining = []

        for idx, item in enumerate(remaining):
            future_noncircles = remaining[:idx] + remaining[idx + 1 :]
            cand = choose_candidate_with_feasibility(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                all_parts=all_parts,
                remaining_noncircle_items=future_noncircles,
                remaining_circle_count=remaining_circle_count,
                strategy=strategy,
            )

            if cand is not None:
                placed_parts.append(cand)
                progress = True
            else:
                next_remaining.append(item)

        remaining = next_remaining

    unplaced.extend(remaining)
    return placed_parts, unplaced


# ============================================================
# RESERVED CIRCLE STRIP
# ============================================================
def candidate_right_strip_widths(parts, plate_poly):
    _, radius, diameter, _ = get_circle_template(parts)
    circle_count = get_circle_count(parts)

    if radius is None or circle_count <= 0:
        return []

    minx, miny, maxx, maxy = plate_poly.bounds
    plate_width = maxx - minx
    plate_height = maxy - miny

    step_y_hex = math.sqrt(3.0) * radius
    max_rows_hex = max(1, int((plate_height - 2.0 * radius) // step_y_hex) + 1)
    max_rows_grid = max(1, int(plate_height // (2.0 * radius)))
    max_rows = max(max_rows_hex, max_rows_grid)

    min_cols = max(1, math.ceil(circle_count / max_rows))

    widths = set()
    for cols in range(min_cols, min(min_cols + 5, circle_count + 1)):
        widths.add(2.0 * radius * cols)
        widths.add(2.0 * radius * cols + radius)

    widths = sorted(w for w in widths if 2.0 * radius <= w <= 0.45 * plate_width)
    valid = []

    for w in widths:
        strip = box(maxx - w, miny, maxx, maxy)
        _, cnt = place_circles_best_pattern(parts, strip, [], circle_count)
        if cnt == circle_count:
            valid.append(w)

    return valid


# ============================================================
# GENERIC NON-CIRCLE SOLVER
# ============================================================
def flatten_groups_for_strategy(groups, strategy=None):
    strategy = strategy or {}
    groups = list(groups)

    group_order = strategy.get("group_order", "default")
    if group_order == "reverse":
        groups = list(reversed(groups))
    elif group_order == "rectangles_first":
        groups.sort(key=lambda g: (g["shape"] != "rectangle", -g["total_square"]))
    elif group_order == "trapezoids_first":
        groups.sort(key=lambda g: (g["shape"] != "trapezoid", -g["total_square"]))
    else:
        groups.sort(key=lambda g: -g["total_square"])

    items = []
    for g in groups:
        group_items = list(g["items"])
        if strategy.get("within_group") == "reverse":
            group_items = list(reversed(group_items))
        elif strategy.get("within_group") == "smallest_first":
            group_items = sorted(group_items, key=lambda p: (p["area"], p["priority"]))
        else:
            group_items = sorted(group_items, key=lambda p: (-p["area"], p["priority"]))
        items.extend(group_items)

    return items


def place_noncircles_generic(parts, noncircle_parts_list, plate_poly, reserved_circle_count=0, strategy=None):
    strategy = strategy or {}
    groups, _ = build_groups_by_total_square(noncircle_parts_list)
    ordered_items = flatten_groups_for_strategy(groups, strategy=strategy)

    remaining = list(ordered_items)
    placed_parts = []
    stalled = []

    progress = True
    while progress and remaining:
        progress = False
        next_remaining = []

        for idx, item in enumerate(remaining):
            if item["shape"] == "trapezoid":
                future_noncircles = remaining[:idx] + remaining[idx + 1 :]
                cand = choose_candidate_with_feasibility(
                    item=item,
                    placed_parts=placed_parts,
                    plate_poly=plate_poly,
                    all_parts=parts,
                    remaining_noncircle_items=future_noncircles,
                    remaining_circle_count=reserved_circle_count,
                    strategy=strategy,
                )
            else:
                future_noncircles = remaining[:idx] + remaining[idx + 1 :]
                cand = choose_candidate_with_feasibility(
                    item=item,
                    placed_parts=placed_parts,
                    plate_poly=plate_poly,
                    all_parts=parts,
                    remaining_noncircle_items=future_noncircles,
                    remaining_circle_count=reserved_circle_count,
                    strategy=strategy,
                )

            if cand is not None:
                placed_parts.append(cand)
                progress = True
            else:
                next_remaining.append(item)

        remaining = next_remaining

    # second trapezoid-only pass for stubborn trapezoids
    if remaining:
        trap_items = [p for p in remaining if p["shape"] == "trapezoid"]
        other_items = [p for p in remaining if p["shape"] != "trapezoid"]

        if trap_items:
            placed_parts, trap_unplaced = place_trapezoids_generic(
                items=trap_items,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                all_parts=parts,
                remaining_circle_count=reserved_circle_count,
                strategy=strategy,
            )
            remaining = other_items + trap_unplaced
        else:
            remaining = other_items

    stalled.extend(remaining)
    return placed_parts, stalled


def place_layout_generic(parts, plate_poly, strategy=None):
    strategy = strategy or {}
    parts_list = build_parts_list(parts)
    circle_count = sum(1 for p in parts_list if p["shape"] == "circle")
    noncircle_parts_list = [p for p in parts_list if p["shape"] != "circle"]

    placed_noncircles, stalled = place_noncircles_generic(
        parts=parts,
        noncircle_parts_list=noncircle_parts_list,
        plate_poly=plate_poly,
        reserved_circle_count=circle_count,
        strategy=strategy,
    )

    placed_final, _ = place_circles_best_pattern(
        parts=parts,
        plate_poly=plate_poly,
        placed_parts=placed_noncircles,
        circle_count=circle_count,
    )
    return placed_final


def place_layout_with_reserved_circle_strip(parts, plate_poly, strip_width, strategy=None):
    minx, miny, maxx, maxy = plate_poly.bounds
    plate_width = maxx - minx

    if strip_width <= 0 or strip_width >= plate_width:
        return []

    left_box = box(minx, miny, maxx - strip_width, maxy)
    right_box = box(maxx - strip_width, miny, maxx, maxy)

    parts_list = build_parts_list(parts)
    noncircle_parts_list = [p for p in parts_list if p["shape"] != "circle"]
    circle_count = sum(1 for p in parts_list if p["shape"] == "circle")

    placed_noncircles, _ = place_noncircles_generic(
        parts=parts,
        noncircle_parts_list=noncircle_parts_list,
        plate_poly=left_box,
        reserved_circle_count=0,  # circles handled in dedicated strip
        strategy=strategy,
    )

    placed_final, _ = place_circles_best_pattern(
        parts=parts,
        plate_poly=right_box,
        placed_parts=placed_noncircles,
        circle_count=circle_count,
    )
    return placed_final


# ============================================================
# DENSE-GROUP STRIP RETRY
# ============================================================
def group_total_area(group_parts):
    return sum(p["poly"].area for p in group_parts)


def group_fill_ratio_bbox(group_parts):
    if not group_parts:
        return 0.0

    u = unary_union([p["poly"] for p in group_parts]).buffer(0)
    if u.is_empty or u.area <= 1e-9:
        return 0.0

    minx, miny, maxx, maxy = u.bounds
    region_area = (maxx - minx) * (maxy - miny)
    if region_area <= 1e-9:
        return 0.0

    return u.area / region_area


def get_largest_dense_group(placed_parts, min_fill=0.95, min_units=4):
    grouped = group_placed_parts_by_base(placed_parts)

    candidates = []
    for base_id, group_parts in grouped.items():
        units = sum(int(p.get("unit_count", 1)) for p in group_parts)
        fill_ratio = group_fill_ratio_bbox(group_parts)
        total_area = group_total_area(group_parts)

        if units >= min_units and fill_ratio >= min_fill:
            candidates.append((base_id, total_area, units, group_parts))

    if not candidates:
        return None, []

    base_id, _, _, group_parts = max(candidates, key=lambda t: (t[1], t[2]))
    return base_id, group_parts


def get_best_side_strip_for_group(plate_poly, fixed_parts):
    if not fixed_parts:
        return None, None

    pminx, pminy, pmaxx, pmaxy = plate_poly.bounds
    u = unary_union([p["poly"] for p in fixed_parts]).buffer(0)
    gminx, gminy, gmaxx, gmaxy = u.bounds

    strips = []

    if gminx > pminx + 1e-6:
        strips.append(("left", box(pminx, pminy, gminx, pmaxy)))
    if gmaxx < pmaxx - 1e-6:
        strips.append(("right", box(gmaxx, pminy, pmaxx, pmaxy)))
    if gminy > pminy + 1e-6:
        strips.append(("bottom", box(pminx, pminy, pmaxx, gminy)))
    if gmaxy < pmaxy - 1e-6:
        strips.append(("top", box(pminx, gmaxy, pmaxx, pmaxy)))

    if not strips:
        return None, None

    name, strip = max(strips, key=lambda t: t[1].area)
    return name, strip


def build_remaining_parts_json(parts, fixed_parts):
    fixed_counts = placed_base_counts(fixed_parts)
    remaining_parts = []

    for part in parts:
        pid = str(part["id"])
        qty = int(part.get("quantity", 1))
        remain = max(0, qty - fixed_counts.get(pid, 0))
        if remain <= 0:
            continue

        new_part = dict(part)
        new_part["quantity"] = remain
        remaining_parts.append(new_part)

    return remaining_parts


def solve_remaining_strip_after_dense_group(parts, plate_poly, first_layout):
    fixed_id, fixed_parts = get_largest_dense_group(first_layout, min_fill=0.95, min_units=4)
    if not fixed_parts:
        return first_layout

    remaining_parts = build_remaining_parts_json(parts, fixed_parts)
    if not remaining_parts:
        return fixed_parts

    _, strip_box = get_best_side_strip_for_group(plate_poly, fixed_parts)
    if strip_box is None:
        return first_layout

    rest_layout = place_layout_generic(remaining_parts, strip_box, strategy={"enforce_circle_feasibility": True})
    merged = list(fixed_parts) + list(rest_layout)

    return merged if layout_rank_key(merged) >= layout_rank_key(first_layout) else first_layout


# ============================================================
# SPECIAL MODE DETECTION
# ============================================================
def classify_layout_mode(parts_list):
    circles = [p for p in parts_list if p["shape"] == "circle"]
    trapezoids = [p for p in parts_list if p["shape"] == "trapezoid"]
    non_circles = [p for p in parts_list if p["shape"] != "circle"]

    trap_base_ids = {p["base_id"] for p in trapezoids}

    if (
        non_circles
        and all(p["shape"] == "trapezoid" for p in non_circles)
        and len(trap_base_ids) == 1
        and len(trapezoids) >= 6
    ):
        return "trapezoid_strip"

    return "generic"


def place_parts_trapezoid_strip_from_parts_list(parts, parts_list, plate_poly):
    trapezoids = [p for p in parts_list if p["shape"] == "trapezoid"]
    circles = [p for p in parts_list if p["shape"] == "circle"]
    others = [p for p in parts_list if p["shape"] not in ("trapezoid", "circle")]

    placed_parts = []
    placed_parts = place_identical_trapezoids_global(
        items=trapezoids,
        placed_parts=placed_parts,
        plate_poly=plate_poly,
        beam_width=80,
    )

    # generic pass for any other shapes
    for item in others:
        cand = quick_place_for_simulation(item, placed_parts, plate_poly)
        if cand is not None:
            placed_parts.append(cand)

    if circles:
        placed_parts, _ = place_circles_best_pattern(
            parts=parts,
            plate_poly=plate_poly,
            placed_parts=placed_parts,
            circle_count=len(circles),
        )

    return placed_parts


# ============================================================
# MASTER SOLVER
# ============================================================
def place_parts_with_existing(parts, plate_poly, strategy=None):
    parts_list = build_parts_list(parts)
    mode = classify_layout_mode(parts_list)

    if mode == "trapezoid_strip":
        return place_parts_trapezoid_strip_from_parts_list(parts, parts_list, plate_poly)

    return place_layout_generic(parts, plate_poly, strategy=strategy)


def nest_parts_with_full_fit(parts, plate_poly, max_rounds=FAST_MAX_ROUNDS):
    expected = expected_part_count(parts)

    best_layout = []
    best_key = (-1, -1.0, float("-inf"))

    def consider(layout, label):
        nonlocal best_layout, best_key
        key = layout_rank_key(layout)
        print(
            f"{label}: placed={placed_part_count(layout)}/{expected}, "
            f"fill_ratio={layout_fill_ratio(layout):.4f}, "
            f"bbox_area={overall_bbox_area(layout):.2f}"
        )
        if key > best_key:
            best_key = key
            best_layout = layout

    strategies = [
        {"name": "default", "enforce_circle_feasibility": True},
        {"name": "vertical_bands", "rect_mode": "vertical_first", "enforce_circle_feasibility": True},
        {"name": "reverse_groups", "group_order": "reverse", "enforce_circle_feasibility": True},
        {"name": "rectangles_first", "group_order": "rectangles_first", "enforce_circle_feasibility": True},
        {"name": "trapezoids_first", "group_order": "trapezoids_first", "enforce_circle_feasibility": True},
        {"name": "smallest_first", "within_group": "smallest_first", "enforce_circle_feasibility": True},
    ]

    # 1) generic direct passes
    for idx, strategy in enumerate(strategies[:max_rounds], start=1):
        print(f"\n=== ROUND {idx} / {max_rounds} ===")
        print(f"Strategy: {strategy['name']}")
        layout = place_parts_with_existing(parts, plate_poly, strategy=strategy)
        consider(layout, f"generic-{strategy['name']}")

        if placed_part_count(layout) == expected:
            return layout

    # 2) dense-group retry
    print("\n=== DENSE GROUP STRIP RETRY ===")
    dense_retry = solve_remaining_strip_after_dense_group(parts, plate_poly, best_layout if best_layout else [])
    consider(dense_retry, "dense-group-strip")
    if placed_part_count(dense_retry) == expected:
        return dense_retry

    # 3) reserved right-strip trials for circles
    circle_count = get_circle_count(parts)
    if circle_count > 0:
        print("\n=== RESERVED RIGHT STRIP TRIALS ===")
        strip_widths = candidate_right_strip_widths(parts, plate_poly)
        for w in strip_widths:
            for strategy in strategies[:4]:
                print(f"Trying right strip width={w:.2f}, strategy={strategy['name']}")
                layout = place_layout_with_reserved_circle_strip(parts, plate_poly, w, strategy=strategy)
                consider(layout, f"right-strip-{int(round(w))}-{strategy['name']}")
                if placed_part_count(layout) == expected:
                    return layout

    best_count = placed_part_count(best_layout)
    raise NestingFailed(
        f"Could not place all parts. Best result was {best_count}/{expected}.",
        best_layout=best_layout,
        best_count=best_count,
        expected_count=expected,
    )


# ============================================================
# PLOT / TABLE
# ============================================================
def plot_arrangement(placed_parts, plate_poly):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect("equal")

    xs, ys = plate_poly.exterior.xy
    ax.plot(xs, ys, "black", linewidth=1.5)

    for p in placed_parts:
        poly = p["poly"]
        polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]

        for g in polys:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, "b")

            c = g.centroid
            ax.text(
                c.x,
                c.y,
                str(p["id"]),
                ha="center",
                va="center",
                fontsize=8,
                color="green",
            )

    ax.set_title("Layout")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def generate_table(placed_parts):
    summary = {}
    for p in placed_parts:
        key = get_part_key(p)
        count = int(p.get("unit_count", 1))
        summary[key] = summary.get(key, 0) + count

    print(f"{'ID':<12} {'Placed Count':<12}")
    for pid, count in summary.items():
        print(f"{pid:<12} {count:<12}")


# ============================================================
# GUI
# ============================================================
class NestingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Nesting - Generic Flow")

        tk.Label(root, text="Layout Width:").grid(row=0, column=0)
        tk.Label(root, text="Layout Height:").grid(row=1, column=0)

        self.width_entry = tk.Entry(root)
        self.height_entry = tk.Entry(root)
        self.width_entry.grid(row=0, column=1)
        self.height_entry.grid(row=1, column=1)

        self.load_button = tk.Button(root, text="Load JSON", command=self.load_json_file)
        self.load_button.grid(row=2, column=0, columnspan=2, pady=5)

        self.run_button = tk.Button(root, text="Run Nesting", command=self.run_nesting)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=5)

        self.parts = None
        self.plate_poly = None

    def load_json_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            self.parts = load_json(file_path)["parts"]
            total_quantity = sum(int(p.get("quantity", 1)) for p in self.parts)
            messagebox.showinfo(
                "Success",
                f"Loaded {len(self.parts)} part objects ({total_quantity} total items)",
            )

    def run_nesting(self):
        if self.parts is None:
            messagebox.showerror("Error", "Please load a JSON file first")
            return

        try:
            w = float(self.width_entry.get())
            h = float(self.height_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Enter valid numeric width and height")
            return

        self.plate_poly = box(0, 0, w, h)

        try:
            placed_parts = nest_parts_with_full_fit(self.parts, self.plate_poly)

            plot_arrangement(placed_parts, self.plate_poly)
            generate_table(placed_parts)

            placed = placed_part_count(placed_parts)
            expected = expected_part_count(self.parts)

            if placed == expected:
                messagebox.showinfo(
                    "Nesting Success",
                    f"All parts were placed successfully: {placed}/{expected}"
                )
            else:
                messagebox.showwarning(
                    "Nesting Incomplete",
                    f"Only {placed}/{expected} parts were placed."
                )

        except NestingFailed as e:
            if e.best_layout:
                plot_arrangement(e.best_layout, self.plate_poly)
                generate_table(e.best_layout)

            messagebox.showwarning(
                "Nesting incomplete",
                f"{e}\n\nShowing best result: {e.best_count}/{e.expected_count} parts placed."
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = NestingApp(root)
    root.mainloop()