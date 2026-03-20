import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Point, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box

class NestingFailed(Exception):
    def __init__(self, message, best_layout=None, best_count=0, expected_count=0):
        super().__init__(message)
        self.best_layout = best_layout or []
        self.best_count = best_count
        self.expected_count = expected_count

FAST_PAIR_GRID_STEPS = 18
FAST_CIRCLE_PHASE_SAMPLES = 40
FAST_BRANCH_CANDIDATES = 10
FAST_LOOKAHEAD_DEPTH = 3
FAST_MAX_ROUNDS = 5
FAST_STRIP_STRATEGIES = 2
MAX_TRAPEZOID_BEAM = 120

def freeze_strategy(strategy):
    if strategy is None:
        return None
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in strategy.items()))

def freeze_bounds(poly):
    return tuple(round(v, 6) for v in poly.bounds)

# -----------------------------
# Load JSON
# -----------------------------
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Geometry conversion
# -----------------------------
def part_to_polygon(part):
    pts = part["contours"][0]["points"]

    # Circle stored as one point with radius
    if len(pts) == 1 and pts[0].get("radius", 0) > 0:
        angle = np.linspace(0, 2 * np.pi, 60)
        r = pts[0]["radius"]
        cx, cy = pts[0]["x"], pts[0]["y"]
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


# -----------------------------
# Shape detection
# -----------------------------
def edge_vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])


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


# -----------------------------
# Trapezoid pairing
# -----------------------------
def combine_trapezoids(poly):
    """
    Create a compact pair from two trapezoids.
    Only used when quantity == 2.
    """
    best_area = float("inf")
    best_union = None

    rotations = [0, 180]

    for angle1 in rotations:
        for angle2 in rotations:
            p1 = rotate(poly, angle1, origin=(0, 0), use_radians=False)
            p2 = rotate(poly, angle2, origin=(0, 0), use_radians=False)

            for dx in np.linspace(-1000, 1000, 70):
                for dy in np.linspace(-1000, 1000, 70):
                    p2_moved = translate(p2, xoff=dx, yoff=dy)

                    if p1.intersection(p2_moved).area > 1e-6:
                        continue

                    combined = p1.union(p2_moved)
                    area = bounding_area(combined)

                    if area < best_area:
                        best_area = area
                        best_union = combined

    return best_union

def get_polygon_vertices(poly):
    """
    Return all exterior vertices from Polygon or MultiPolygon.
    """
    if isinstance(poly, MultiPolygon):
        verts = []
        for g in poly.geoms:
            verts.extend(list(g.exterior.coords)[:-1])
        return verts

    if isinstance(poly, ShapelyPolygon):
        return list(poly.exterior.coords)[:-1]

    return []

def group_nominal_width(group):
    if not group["items"]:
        return 0.0
    return max(poly_size(item["poly"])[0] for item in group["items"])


def split_groups_by_requested_priority(parts_list):
    """
    Priority requested by user:
    1) the ONE rectangle family with the longest original width
    2) trapezoids
    3) remaining rectangles
    4) other polygons
    Circles are returned separately and placed last.
    """
    grouped, circles = build_non_circle_groups(parts_list)

    for g in grouped:
        g["nominal_width"] = group_nominal_width(g)
        g["max_unit_count"] = max(int(it.get("unit_count", 1)) for it in g["items"]) if g["items"] else 1

    rect_groups = [g for g in grouped if g["shape"] == "rectangle"]
    trap_groups = [g for g in grouped if g["shape"] == "trapezoid"]
    other_groups = [g for g in grouped if g["shape"] not in ("rectangle", "trapezoid")]

    first_rect = None
    if rect_groups:
        first_rect = max(
            rect_groups,
            key=lambda g: (g["nominal_width"], g["total_square"], len(g["items"]))
        )

    other_rects = [g for g in rect_groups if first_rect is None or g["base_id"] != first_rect["base_id"]]

    trap_groups.sort(
        key=lambda g: (
            -g["max_unit_count"],   # pairs first
            -g["total_square"],
            g["base_id"],
        )
    )

    other_rects.sort(
        key=lambda g: (
            -g["nominal_width"],
            -g["total_square"],
            g["base_id"],
        )
    )

    other_groups.sort(
        key=lambda g: (
            -g["total_square"],
            g["base_id"],
        )
    )

    return first_rect, trap_groups, other_rects, other_groups, circles


def place_group_in_vertical_bands(group, placed_parts, plate_poly, strategy=None):
    """
    Used only for the FIRST rectangle family (largest width).
    It places that family completely before moving to trapezoids.
    """
    strategy = strategy or {}
    angle_order = strategy.get("anchor_angle_order", [0, 90])

    minx, miny, maxx, maxy = plate_poly.bounds
    x_cursor = minx
    remaining_items = list(group["items"])

    while remaining_items:
        region_width = maxx - x_cursor
        region_height = maxy - miny

        if region_width <= 1e-6 or region_height <= 1e-6:
            break

        variant = best_repeat_variant_for_group(
            remaining_items,
            region_width=region_width,
            region_height=region_height,
            angle_order=angle_order,
        )

        if variant is None:
            break

        if x_cursor + variant["w"] > maxx + 1e-6:
            break

        region_box = box(x_cursor, miny, maxx, maxy)
        anchor_choice = {"group": group, "variant": variant}

        placed_parts, remaining_items, band_box, used_top_y, placed_count = place_anchor_band(
            anchor_choice=anchor_choice,
            remaining_items=remaining_items,
            placed_parts=placed_parts,
            region_box=region_box,
        )

        if placed_count == 0:
            break

        bminx, bminy, bmaxx, bmaxy = band_box.bounds
        if used_top_y < bmaxy - 1e-6 and remaining_items:
            top_box = box(bminx, used_top_y, bmaxx, bmaxy)
            placed_parts, remaining_items = fill_box_greedily(
                remaining_items=remaining_items,
                placed_parts=placed_parts,
                region_box=top_box,
                strategy=strategy,
                max_passes=2,
            )

        x_cursor += variant["w"]

    return placed_parts


def place_group_items_greedily(group, placed_parts, plate_poly, strategy=None):
    """
    Used for:
    - remaining rectangles
    - other polygons
    Order inside the group is still largest area first.
    """
    strategy = strategy or {}
    angle_order = strategy.get("fill_angle_order", [0, 90])

    items = sorted(group["items"], key=lambda p: (-p["area"], p["priority"]))

    for item in items:
        cand = None

        if item["shape"] == "rectangle":
            cand = place_item_top_left(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                preferred_angles=angle_order,
            )
            if cand is None:
                cand = place_item_bottom_left(
                    item=item,
                    placed_parts=placed_parts,
                    plate_poly=plate_poly,
                    preferred_angles=angle_order,
                )

        elif item["shape"] == "trapezoid":
            cand = place_item_bottom_left(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                preferred_angles=angle_order,
            )
            if cand is None:
                cand = place_item_top_left(
                    item=item,
                    placed_parts=placed_parts,
                    plate_poly=plate_poly,
                    preferred_angles=angle_order,
                )

        else:
            cand = place_item_generic(item, placed_parts, plate_poly)

        if cand is not None:
            placed_parts.append(cand)
            print(
                f"Placed part {cand['id']} "
                f"(shape={cand['shape']}) "
                f"at x={cand['x']}, y={cand['y']}, angle={cand['angle']}"
            )
        else:
            print(f"Could not place part {item['id']}, not enough space")

    return placed_parts

def place_parts_requested_priority(parts, parts_list, plate_poly, strategy=None):
    """
    Requested order:
    1) longest-width rectangle family first
    2) trapezoids second
    3) remaining rectangles third
    4) circles last

    Extra cases:
    - CASE 1: if there is only one rectangle family + trapezoids,
      also try trapezoids-first and keep the better result
    - CASE 2 is handled inside place_trapezoids_min_total_space()
    """
    strategy = strategy or {}

    first_rect, trap_groups, other_rects, other_groups, circles = split_groups_by_requested_priority(parts_list)

    def collect_items(groups):
        out = []
        for g in groups:
            out.extend(g["items"])
        return out

    def finish_with_circles(placed_parts):
        if circles:
            print("Priority step 4: circles")
            placed_parts, _ = place_circles_best_pattern(
                parts=parts,
                plate_poly=plate_poly,
                placed_parts=placed_parts,
                circle_count=len(circles),
            )
        return placed_parts

    def solve_rect_first():
        placed_parts = []

        # 1) FIRST: longest-width rectangle family
        if first_rect is not None:
            print(
                f"Priority step 1: rectangle family {first_rect['base_id']} "
                f"(width={first_rect['nominal_width']:.2f})"
            )
            placed_parts = place_group_in_vertical_bands(
                group=first_rect,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                strategy=strategy,
            )

        # 2) SECOND: trapezoids
        trap_items = []
        for g in trap_groups:
            g_items = sorted(
                g["items"],
                key=lambda p: (
                    -int(p.get("unit_count", 1)),  # pairs first
                    -p["area"],
                ),
            )
            trap_items.extend(g_items)

        if trap_items:
            later_items = []
            for g in other_rects + other_groups:
                later_items.extend(g["items"])

            print("Priority step 2: trapezoids")
            placed_parts = place_trapezoids_min_total_space(
                items=trap_items,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                later_items=later_items,
            )

        # 3) THIRD: remaining rectangles
        for g in other_rects:
            print(
                f"Priority step 3: rectangle family {g['base_id']} "
                f"(width={g['nominal_width']:.2f})"
            )
            placed_parts = place_group_items_greedily(
                group=g,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                strategy=strategy,
            )

        # 3b) other polygons
        for g in other_groups:
            print(f"Priority step 3b: other polygon family {g['base_id']}")
            placed_parts = place_group_items_greedily(
                group=g,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                strategy=strategy,
            )

        return finish_with_circles(placed_parts)

    def solve_traps_first():
        placed_parts = []

        trap_items = []
        for g in trap_groups:
            g_items = sorted(
                g["items"],
                key=lambda p: (
                    -int(p.get("unit_count", 1)),  # pairs first
                    -p["area"],
                ),
            )
            trap_items.extend(g_items)

        later_items = []
        if first_rect is not None:
            later_items.extend(first_rect["items"])
        later_items.extend(collect_items(other_rects))
        later_items.extend(collect_items(other_groups))

        # 1 alt) trapezoids first
        if trap_items:
            print("Case 1 alt order: trapezoids first")
            placed_parts = place_trapezoids_min_total_space(
                items=trap_items,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                later_items=later_items,
            )

        # 2 alt) then longest-width rectangle
        if first_rect is not None:
            print(
                f"Case 1 alt order: rectangle family {first_rect['base_id']} "
                f"(width={first_rect['nominal_width']:.2f})"
            )
            placed_parts = place_group_in_vertical_bands(
                group=first_rect,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                strategy=strategy,
            )

        # 3 alt) remaining rectangles
        for g in other_rects:
            placed_parts = place_group_items_greedily(
                group=g,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                strategy=strategy,
            )

        # 3b alt) other polygons
        for g in other_groups:
            placed_parts = place_group_items_greedily(
                group=g,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                strategy=strategy,
            )

        return finish_with_circles(placed_parts)

    rect_first_layout = solve_rect_first()

    # CASE 1:
    # If the instance is basically one rectangle family + trapezoids (+ maybe circles),
    # also try the reverse order and keep the better one.
    only_one_rect_family = first_rect is not None and len(other_rects) == 0
    only_rects_and_traps = len(other_groups) == 0

    if only_one_rect_family and trap_groups and only_rects_and_traps:
        traps_first_layout = solve_traps_first()

        if layout_rank_key(traps_first_layout) > layout_rank_key(rect_first_layout):
            print("Selected alternate order: trapezoids first, then longest rectangle")
            return traps_first_layout

    return rect_first_layout


# -----------------------------
# Build part instances
# -----------------------------
def build_parts_list(parts):
    parts_list = []
    pair_cache = {}
    meta = {}

    # detect pure repeated trapezoid-strip case
    all_non_circles = []
    trap_base_ids = set()

    for part in parts:
        pid = str(part["id"])
        poly = part_to_polygon(part)
        shape_type = detect_shape_type(part, poly)
        meta[pid] = (poly, shape_type)

        if shape_type != "circle":
            all_non_circles.append(shape_type)

        if shape_type == "trapezoid":
            trap_base_ids.add(pid)

    pure_trapezoid_strip_case = (
        len(all_non_circles) > 0
        and all(s == "trapezoid" for s in all_non_circles)
        and len(trap_base_ids) == 1
    )

    for part in parts:
        pid = str(part["id"])
        poly, shape_type = meta[pid]
        qty = int(part.get("quantity", 1))

        # non-trapezoids
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

        # pure trapezoid strip -> ALSO build pairs first
        if pure_trapezoid_strip_case:
            if pid not in pair_cache:
                pair_cache[pid] = combine_trapezoids(poly)

            pair_poly = pair_cache[pid]
            pair_count = qty // 2
            remainder = qty % 2

            if pair_poly is not None:
                for i in range(pair_count):
                    parts_list.append(
                        {
                            "id": f"{pid}_pair_{i}",
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

            continue

        # mixed layout -> make qty//2 pairs + maybe one single
        if pid not in pair_cache:
            pair_cache[pid] = combine_trapezoids(poly)

        pair_poly = pair_cache[pid]
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


# -----------------------------
# Collision helpers
# -----------------------------
def has_real_overlap(candidate, placed_parts, eps=1e-6):
    cminx, cminy, cmaxx, cmaxy = candidate.bounds

    for p in placed_parts:
        pminx, pminy, pmaxx, pmaxy = p["poly"].bounds

        if cmaxx <= pminx + eps or pmaxx <= cminx + eps:
            continue
        if cmaxy <= pminy + eps or pmaxy <= cminy + eps:
            continue

        if candidate.intersection(p["poly"]).area > eps:
            return True
    return False


def valid_candidate(candidate, placed_parts, plate_poly):
    return plate_poly.covers(candidate) and not has_real_overlap(candidate, placed_parts)


# -----------------------------
# Candidate positions
# -----------------------------
def generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh):
    minx, miny, maxx, maxy = plate_poly.bounds

    positions = {
        (minx, miny),
        (minx, maxy - rh),
        (maxx - rw, miny),
        (maxx - rw, maxy - rh),
    }

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

    return sorted({(round(x, 6), round(y, 6)) for x, y in positions})


# -----------------------------
# Placement helpers
# -----------------------------
def rotate_normalize(poly, angle):
    return normalize_poly(rotate(poly, angle, origin="centroid", use_radians=False))


def place_item_bottom_left(item, placed_parts, plate_poly, preferred_angles=None):
    if preferred_angles is None:
        preferred_angles = [0, 90]

    best = None
    best_score = None

    for angle in preferred_angles:
        rotated = get_rotated_variant(item, angle)
        rw, rh = poly_size(rotated)

        for x, y in generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh):
            candidate = translate(rotated, xoff=x, yoff=y)
            if not valid_candidate(candidate, placed_parts, plate_poly):
                continue

            score = (y, x, rw * rh)
            if best is None or score < best_score:
                best = {
                    "id": item["id"],
                    "base_id": item.get("base_id", item["id"]),
                    "unit_count": item.get("unit_count", 1),
                    "poly": candidate,
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                }
                best_score = score

    return best


def place_item_top_left(item, placed_parts, plate_poly, preferred_angles=None):
    if preferred_angles is None:
        preferred_angles = [0, 90]

    _, _, _, plate_top = plate_poly.bounds
    best = None
    best_score = None

    for angle in preferred_angles:
        rotated = get_rotated_variant(item, angle)
        rw, rh = poly_size(rotated)

        for x, y in generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh):
            candidate = translate(rotated, xoff=x, yoff=y)
            if not valid_candidate(candidate, placed_parts, plate_poly):
                continue

            minx, miny, maxx, maxy = candidate.bounds
            score = (plate_top - maxy, minx, rw * rh)
            if best is None or score < best_score:
                best = {
                    "id": item["id"],
                    "poly": candidate,
                    "base_id": item.get("base_id", item["id"]),
                    "unit_count": item.get("unit_count", 1),
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                }
                best_score = score

    return best

def get_part_key(p):
    return p.get("base_id", p.get("id", "unknown"))

def group_placed_parts_by_base(placed_parts):
    groups = {}
    for p in placed_parts:
        key = get_part_key(p)
        groups.setdefault(key, []).append(p)
    return groups

def placed_base_counts(placed_parts):
    counts = {}
    for p in placed_parts:
        key = get_part_key(p)
        counts[key] = counts.get(key, 0) + int(p.get("unit_count", 1))
    return counts

def cluster_bounds(placed_parts, fallback_poly):
    if not placed_parts:
        return fallback_poly.bounds

    minx = min(p["poly"].bounds[0] for p in placed_parts)
    miny = min(p["poly"].bounds[1] for p in placed_parts)
    maxx = max(p["poly"].bounds[2] for p in placed_parts)
    maxy = max(p["poly"].bounds[3] for p in placed_parts)
    return (minx, miny, maxx, maxy)

def place_item_near_reference_vertical(item, placed_parts, plate_poly, ref_bounds):
    ref_right = ref_bounds[2]
    best = None
    best_score = None

    for angle in [90, 270]:
        rotated = get_rotated_variant(item, angle)
        rw, rh = poly_size(rotated)

        for x, y in generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh):
            candidate = translate(rotated, xoff=x, yoff=y)
            if not valid_candidate(candidate, placed_parts, plate_poly):
                continue

            minx, miny, maxx, maxy = candidate.bounds

            penalty_left_of_ref = 0 if minx >= ref_right - 1e-6 else 1
            score = (
                penalty_left_of_ref,
                abs(minx - ref_right),
                miny,
                rw * rh,
            )

            if best is None or score < best_score:
                best = {
                    "id": item["id"],
                    "base_id": item.get("base_id", item["id"]),
                    "unit_count": item.get("unit_count", 1),
                    "poly": candidate,
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                }
                best_score = score

    return best

def place_item_generic(item, placed_parts, plate_poly):
    if item["shape"] == "trapezoid":
        return place_item_bottom_left(item, placed_parts, plate_poly)
    return place_item_top_left(item, placed_parts, plate_poly)


def enumerate_candidate_placements(
    item,
    placed_parts,
    plate_poly,
    mode="bottom_left",
    preferred_angles=None,
    ref_bounds=None,
    max_candidates=24,
):
    """
    Generate valid placements for the current part in original/rotated states.
    mode:
      - "top_left"
      - "near_vertical"
      - "bottom_left"
    """
    if preferred_angles is None:
        preferred_angles = [0, 90]

    _, _, _, plate_top = plate_poly.bounds
    ref_right = ref_bounds[2] if ref_bounds is not None else None

    candidates = []
    seen = set()

    for angle in preferred_angles:
        rotated = get_rotated_variant(item, angle)
        rw, rh = poly_size(rotated)

        for x, y in generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh):
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
                    "poly": candidate_poly,
                    "base_id": item.get("base_id", item["id"]),
                    "unit_count": item.get("unit_count", 1),
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                    "local_score": local_score,
                }
            )

    candidates.sort(key=lambda c: c["local_score"])
    return candidates[:max_candidates]


def quick_place_for_simulation(item, placed_parts, plate_poly):
    """
    Fast greedy placement used only for lookahead simulation.
    """
    candidates = []

    if item["shape"] == "rectangle":
        horiz = orientation_profile(item, "horizontal")
        vert = orientation_profile(item, "vertical")

        candidates.extend(
            enumerate_candidate_placements(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                mode="top_left",
                preferred_angles=[horiz["angle"], 0, 90],
                max_candidates=8,
            )
        )

        if placed_parts:
            ref_bounds = cluster_bounds(placed_parts, plate_poly)
            candidates.extend(
                enumerate_candidate_placements(
                    item=item,
                    placed_parts=placed_parts,
                    plate_poly=plate_poly,
                    mode="near_vertical",
                    preferred_angles=[vert["angle"], 0,90],
                    ref_bounds=ref_bounds,
                    max_candidates=8,
                )
            )

        candidates.extend(
            enumerate_candidate_placements(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                mode="bottom_left",
                preferred_angles=[vert["angle"], horiz["angle"], 0, 90],
                max_candidates=8,
            )
        )

    elif item["shape"] == "trapezoid":
        candidates.extend(
            enumerate_candidate_placements(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                mode="bottom_left",
                preferred_angles=[0, 90],
                max_candidates=10,
            )
        )

    else:
        candidates.extend(
            enumerate_candidate_placements(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                mode="top_left",
                preferred_angles=[0, 90],
                max_candidates=8,
            )
        )
        candidates.extend(
            enumerate_candidate_placements(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                mode="bottom_left",
                preferred_angles=[0, 90],
                max_candidates=8,
            )
        )

    if not candidates:
        return None

    return min(candidates, key=lambda c: c["local_score"])


def simulate_remaining_fit(placed_parts, remaining_items, plate_poly, max_depth=5):
    """
    Simulate whether the next parts can still fit in the extra space.
    Higher score is better.
    """
    temp_parts = list(placed_parts)
    fit_count = 0
    fit_area = 0.0

    # Check biggest remaining parts first
    test_items = sorted(remaining_items, key=lambda p: -p["area"])[:max_depth]

    for item in test_items:
        cand = quick_place_for_simulation(item, temp_parts, plate_poly)
        if cand is None:
            continue

        temp_parts.append(cand)
        fit_count += 1
        fit_area += item["area"]

    if temp_parts:
        union_poly = unary_union([p["poly"] for p in temp_parts])
        minx, miny, maxx, maxy = union_poly.bounds
        bbox_area = (maxx - minx) * (maxy - miny)
    else:
        bbox_area = 0.0

    # More future parts fitting is better.
    # More future area fitting is better.
    # Smaller final bounding envelope is better.
    return (fit_count, fit_area, -bbox_area)


def place_item_with_lookahead(
    item,
    placed_parts,
    plate_poly,
    remaining_items,
    mode="bottom_left",
    preferred_angles=None,
    ref_bounds=None,
    max_candidates=10,
    lookahead_depth=3,
):
    """
    Choose the current placement by checking how well future parts still fit.
    """
    candidates = enumerate_candidate_placements(
        item=item,
        placed_parts=placed_parts,
        plate_poly=plate_poly,
        mode=mode,
        preferred_angles=preferred_angles,
        ref_bounds=ref_bounds,
        max_candidates=max_candidates,
    )

    if not candidates:
        return None

    best = None
    best_score = None

    for cand in candidates:
        score = simulate_remaining_fit(
            placed_parts=placed_parts + [cand],
            remaining_items=remaining_items,
            plate_poly=plate_poly,
            max_depth=lookahead_depth,
        )

        if best is None or score > best_score:
            best = cand
            best_score = score
        elif score == best_score and cand["local_score"] < best["local_score"]:
            best = cand
            best_score = score

    return best


# -----------------------------
# Circle packing helpers
# -----------------------------
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

    samples = FAST_CIRCLE_PHASE_SAMPLES

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

                if placed_centers:
                    score = (placed_now, len(placed_centers))
                else:
                    score = (placed_now, 0)

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

    for p in best_layout[len(placed_parts):]:
        print(f"Placed part {p['id']} (shape=circle) at x={p['x']}, y={p['y']}, angle=0")

    return best_layout, best_count

# -----------------------------
# Trapezoid optimization
# -----------------------------
def trapezoid_cluster_score(current_traps, candidate_poly):
    polys = [p["poly"] for p in current_traps] + [candidate_poly]
    u = unary_union(polys)

    minx, miny, maxx, maxy = u.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    waste = bbox_area - u.area

    return (
        waste,
        bbox_area,
        maxy - miny,
        maxx - minx,
        miny,
        minx,
    )


def generate_trapezoid_candidate_positions(rotated_poly, placed_parts, plate_poly):
    rw, rh = poly_size(rotated_poly)
    positions = set(generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh))

    cand_vertices = get_polygon_vertices(rotated_poly)

    minx, miny, maxx, maxy = plate_poly.bounds
    plate_pts = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]

    # align candidate vertices to plate corners
    for ax, ay in plate_pts:
        for vx, vy in cand_vertices:
            positions.add((round(ax - vx, 6), round(ay - vy, 6)))

    # align candidate vertices to existing part vertices
    for p in placed_parts:
        poly = p["poly"]
        existing_vertices = get_polygon_vertices(poly)

        for ex, ey in existing_vertices:
            for vx, vy in cand_vertices:
                positions.add((round(ex - vx, 6), round(ey - vy, 6)))

    return sorted(positions)


def place_item_min_trapezoid_waste(item, placed_parts, plate_poly, current_traps, preferred_angles=None):
    if preferred_angles is None:
        preferred_angles = [0, 90]

    best = None
    best_score = None

    for angle in preferred_angles:
        rotated = get_rotated_variant(item, angle)

        for x, y in generate_trapezoid_candidate_positions(rotated, placed_parts, plate_poly):
            candidate = translate(rotated, xoff=x, yoff=y)

            if not valid_candidate(candidate, placed_parts, plate_poly):
                continue

            score = trapezoid_cluster_score(current_traps, candidate)

            if best is None or score < best_score:
                best = {
                    "id": item["id"],
                    "base_id": item.get("base_id", item["id"]),
                    "unit_count": item.get("unit_count", 1),
                    "poly": candidate,
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                }
                best_score = score

    return best


def _min_dx_no_overlap(poly_left, poly_right, dy, eps=1e-7, max_iter=60):
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

def get_rotated_variant(item, angle):
    cache = item.setdefault("_rot_cache", {})
    if angle not in cache:
        cache[angle] = rotate_normalize(item["poly"], angle)
    return cache[angle]

def _trapezoid_lane_configs(base_poly, plate_min_y, plate_height):
    p0 = rotate_normalize(base_poly, 0)
    p180 = rotate_normalize(base_poly, 180)

    h0 = poly_size(p0)[1]
    h180 = poly_size(p180)[1]

    configs = []

    if h0 <= plate_height + 1e-6:
        configs.append([
            {"name": "b0", "angle": 0, "poly": p0, "y": plate_min_y},
        ])

    if h180 <= plate_height + 1e-6:
        configs.append([
            {"name": "b180", "angle": 180, "poly": p180, "y": plate_min_y},
        ])

    if h0 <= plate_height + 1e-6 and h180 <= plate_height + 1e-6:
        configs.append([
            {"name": "b0", "angle": 0, "poly": p0, "y": plate_min_y},
            {"name": "t180", "angle": 180, "poly": p180, "y": plate_min_y + plate_height - h180},
        ])
        configs.append([
            {"name": "b180", "angle": 180, "poly": p180, "y": plate_min_y},
            {"name": "t0", "angle": 0, "poly": p0, "y": plate_min_y + plate_height - h0},
        ])
        configs.append([
            {"name": "b0", "angle": 0, "poly": p0, "y": plate_min_y},
            {"name": "t0", "angle": 0, "poly": p0, "y": plate_min_y + plate_height - h0},
        ])
        configs.append([
            {"name": "b180", "angle": 180, "poly": p180, "y": plate_min_y},
            {"name": "t180", "angle": 180, "poly": p180, "y": plate_min_y + plate_height - h180},
        ])

    return configs


def place_identical_trapezoids_global(items, placed_parts, plate_poly, beam_width=MAX_TRAPEZOID_BEAM):
    if not items:
        return list(placed_parts)

    minx, miny, maxx, maxy = plate_poly.bounds
    plate_width = maxx - minx
    plate_height = maxy - miny

    base_poly = items[0]["poly"]

    same_base = len({p.get("base_id", p["id"]) for p in items}) == 1
    pair_mode = (
        same_base
        and len(items) > 2
        and all(int(p.get("unit_count", 1)) == 2 for p in items)
    )

    # --------------------------------------------------
    # HARD RULE:
    # many trapezoid pairs => one compact lane only
    # --------------------------------------------------
    if pair_mode:
        best_layout = None
        best_count = -1
        best_used_width = float("inf")

        for ang in [0, 180]:
            rp = rotate_normalize(base_poly, ang)
            rw, rh = poly_size(rp)

            if rh > plate_height + 1e-6:
                continue

            x_cursor = minx
            layout = list(placed_parts)
            placed_n = 0

            for item in items:
                cand_poly = translate(rp, xoff=x_cursor, yoff=miny)

                if x_cursor + rw > maxx + 1e-6:
                    break
                if not plate_poly.covers(cand_poly):
                    break
                if has_real_overlap(cand_poly, layout):
                    break

                layout.append({
                    "id": item["id"],
                    "base_id": item.get("base_id", item["id"]),
                    "unit_count": item.get("unit_count", 1),
                    "poly": cand_poly,
                    "x": x_cursor,
                    "y": miny,
                    "angle": ang,
                    "shape": item["shape"],
                })
                placed_n += 1
                x_cursor += rw

            used_width = x_cursor - minx

            if (
                placed_n > best_count
                or (placed_n == best_count and used_width < best_used_width)
            ):
                best_count = placed_n
                best_used_width = used_width
                best_layout = layout

        if best_layout is not None:
            return best_layout

    # --------------------------------------------------
    # generic fallback for non-pair or small-count cases
    # --------------------------------------------------
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

        states = [{
            "frontier": tuple(0.0 for _ in range(m)),
            "used_width": 0.0,
            "placements": [],
            "counts": tuple(0 for _ in range(m)),
        }]

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

                    new_states.append({
                        "frontier": tuple(new_frontier),
                        "used_width": used_width,
                        "placements": st["placements"] + [(k, x_rel)],
                        "counts": tuple(new_counts),
                    })

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
            states.sort(key=lambda st: (
                st["used_width"],
                sum(st["frontier"]),
                max(st["counts"]) - min(st["counts"]) if len(st["counts"]) > 1 else 0,
            ))
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
        out = list(placed_parts)
        for item in items:
            cand = place_item_bottom_left(item, out, plate_poly)
            if cand is not None:
                out.append(cand)
        return out

    cfg, sol = best_solution
    out = list(placed_parts)

    for item, (k, x_rel) in zip(items, sol["placements"]):
        x_abs = minx + x_rel
        candidate = translate(cfg[k]["poly"], xoff=x_abs, yoff=cfg[k]["y"])
        out.append({
            "id": item["id"],
            "base_id": item.get("base_id", item["id"]),
            "unit_count": item.get("unit_count", 1),
            "poly": candidate,
            "x": x_abs,
            "y": cfg[k]["y"],
            "angle": cfg[k]["angle"],
            "shape": item["shape"],
        })

    return out


def place_trapezoids_min_total_space(items, placed_parts, plate_poly, later_items=None):
    if later_items is None:
        later_items = []

    if not items:
        return list(placed_parts)

    out = list(placed_parts)
    pair_items = [p for p in items if int(p.get("unit_count", 1)) == 2]
    single_items = [p for p in items if int(p.get("unit_count", 1)) == 1]
    current_traps = [p for p in out if p["shape"] == "trapezoid"]

    # --------------------------------------------------
    # FIRST: many identical trapezoid pairs -> compact lane
    # --------------------------------------------------
    if pair_items:
        same_base = len({p["base_id"] for p in pair_items}) == 1
        if same_base and len(pair_items) > 2:
            out = place_identical_trapezoids_global(
                pair_items,
                out,
                plate_poly,
                beam_width=MAX_TRAPEZOID_BEAM,
            )
            current_traps = [p for p in out if p["shape"] == "trapezoid"]
            pair_items = []

    # leftover pair items
    for i, item in enumerate(pair_items):
        remaining_items = pair_items[i + 1:] + single_items + list(later_items)

        cand = place_item_min_trapezoid_waste(
            item=item,
            placed_parts=out,
            plate_poly=plate_poly,
            current_traps=current_traps,
            preferred_angles=[0, 180, 90, 270],
        )

        if cand is None:
            cand = place_item_with_lookahead(
                item=item,
                placed_parts=out,
                plate_poly=plate_poly,
                remaining_items=remaining_items,
                mode="bottom_left",
                preferred_angles=[0, 180, 90, 270],
            )

        if cand is not None:
            out.append(cand)
            current_traps.append(cand)

    # many identical singles
    if single_items:
        same_base = len({p["base_id"] for p in single_items}) == 1
        if same_base and len(single_items) >= 6:
            return place_identical_trapezoids_global(
                single_items,
                out,
                plate_poly,
                beam_width=MAX_TRAPEZOID_BEAM,
            )

    # leftover single trapezoids
    for i, item in enumerate(single_items):
        remaining_items = single_items[i + 1:] + list(later_items)

        cand = place_item_min_trapezoid_waste(
            item=item,
            placed_parts=out,
            plate_poly=plate_poly,
            current_traps=current_traps,
            preferred_angles=[0, 180, 90, 270],
        )

        if cand is None:
            cand = place_item_with_lookahead(
                item=item,
                placed_parts=out,
                plate_poly=plate_poly,
                remaining_items=remaining_items,
                mode="bottom_left",
                preferred_angles=[0, 180, 90, 270],
            )

        if cand is not None:
            out.append(cand)
            current_traps.append(cand)

    return out


# -----------------------------
# Role inference from geometry
# -----------------------------
def orientation_profile(item, target="horizontal"):
    best = None

    for angle in [0, 90]:
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

def expected_part_count(parts):
    return sum(int(p.get("quantity", 1)) for p in parts)

def placed_part_count(placed_parts):
    # trapezoid_pair should count as 2 if unit_count is set
    return sum(int(p.get("unit_count", 1)) for p in placed_parts)

def reorder_items_for_strategy(items, order_mode):
    items = list(items)

    if order_mode == "reverse":
        return list(reversed(items))

    if order_mode == "largest_first":
        return sorted(items, key=lambda p: (-p["area"], p["priority"]))

    if order_mode == "smallest_first":
        return sorted(items, key=lambda p: (p["area"], p["priority"]))

    return items

# -----------------------------
# Second-Stage Strip Solver
#------------------------------
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

def get_best_side_strip_for_group(plate_poly, fixed_parts):
    """
    Build left/right/top/bottom rectangular strips around the fixed group bbox
    and return the biggest one.
    """
    if not fixed_parts:
        return None, None

    pminx, pminy, pmaxx, pmaxy = plate_poly.bounds
    u = unary_union([p["poly"] for p in fixed_parts]).buffer(0)
    gminx, gminy, gmaxx, gmaxy = u.bounds

    strips = []

    # left strip
    if gminx > pminx + 1e-6:
        strips.append(("left", box(pminx, pminy, gminx, pmaxy)))

    # right strip
    if gmaxx < pmaxx - 1e-6:
        strips.append(("right", box(gmaxx, pminy, pmaxx, pmaxy)))

    # bottom strip
    if gminy > pminy + 1e-6:
        strips.append(("bottom", box(pminx, pminy, pmaxx, gminy)))

    # top strip
    if gmaxy < pmaxy - 1e-6:
        strips.append(("top", box(pminx, gmaxy, pmaxx, pmaxy)))

    if not strips:
        return None, None

    name, strip = max(strips, key=lambda t: t[1].area)
    return name, strip

def solve_remaining_strip_after_dense_group(parts, plate_poly, first_layout):
    """
    Keep the dominant dense group fixed, then solve all remaining parts
    inside the biggest rectangular side strip around that dense group.
    """
    fixed_id, fixed_parts = get_largest_dense_group(
        first_layout,
        min_fill=0.95,
        min_units=4,
    )

    if not fixed_parts:
        return first_layout

    print(f"Keeping dense group fixed: {fixed_id}")

    remaining_parts = build_remaining_parts_json(parts, fixed_parts)
    if not remaining_parts:
        return fixed_parts

    strip_name, strip_box = get_best_side_strip_for_group(plate_poly, fixed_parts)
    if strip_box is None:
        return first_layout

    print(f"Solving remaining parts in {strip_name} strip: bounds={strip_box.bounds}")

    # Solve ONLY in that strip
    rest_layout = place_parts_with_existing(
        remaining_parts,
        strip_box,
    )

    merged = list(fixed_parts) + list(rest_layout)

    # Keep the better result
    if placed_part_count(merged) >= placed_part_count(first_layout):
        return merged

    return first_layout


def get_largest_dense_group(placed_parts, min_fill=0.95, min_units=4):
    """
    Return the largest placed group that already packs densely enough.
    Usually this becomes part 1 in your example.
    """
    grouped = group_placed_parts_by_base(placed_parts)

    candidates = []
    for base_id, group_parts in grouped.items():
        units = sum(int(p.get("unit_count", 1)) for p in group_parts)
        fill_ratio = group_fill_ratio_bbox(group_parts)
        total_area = group_total_area(group_parts)

        print(
            f"group {base_id}: units={units}, "
            f"fill_ratio={fill_ratio:.4f}, total_area={total_area:.3f}"
        )

        if units >= min_units and fill_ratio >= min_fill:
            candidates.append((base_id, total_area, units, group_parts))

    if not candidates:
        return None, []

    # largest dense group first
    base_id, _, _, group_parts = max(candidates, key=lambda t: (t[1], t[2]))
    return base_id, group_parts

def infer_mixed_template_roles(parts_list):
    circles = [p for p in parts_list if p["shape"] == "circle"]
    trapezoids = [p for p in parts_list if p["shape"] == "trapezoid"]

    remaining = [p for p in parts_list if p["shape"] not in ("circle", "trapezoid")]
    rect_like = [p for p in remaining if p["shape"] == "rectangle"]

    shelves = sorted(
        rect_like,
        key=lambda item: orientation_profile(item, "horizontal")["score"],
        reverse=True,
    )

    top_shelf = shelves[0] if len(shelves) >= 1 else None
    second_shelf = shelves[1] if len(shelves) >= 2 else None

    used = set()
    if top_shelf is not None:
        used.add(id(top_shelf))
    if second_shelf is not None:
        used.add(id(second_shelf))

    right_bars = []
    for item in rect_like:
        if id(item) in used:
            continue
        right_bars.append(item)

    right_bars = sorted(
        right_bars,
        key=lambda item: orientation_profile(item, "vertical")["score"],
        reverse=True,
    )

    fillers = [p for p in remaining if p["shape"] != "rectangle"]
    fillers = sorted(fillers, key=lambda p: (p["priority"], -p["area"]))
    trapezoids = sorted(trapezoids, key=lambda p: -p["area"])

    return circles, trapezoids, top_shelf, second_shelf, right_bars, fillers


def _place_mixed_template_attempt(parts, parts_list, plate_poly, variant="A"):
    circles, trapezoids, top_shelf, second_shelf, right_bars, fillers = infer_mixed_template_roles(parts_list)

    placed_parts = []

    def place_shelf(item):
        if item is None:
            return
        prof = orientation_profile(item, "horizontal")
        cand = place_item_top_left(
            item=item,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            preferred_angles=[prof["angle"], 0, 90],
        )
        if cand is not None:
            placed_parts.append(cand)

    def place_right_bars():
        for item in right_bars:
            prof = orientation_profile(item, "vertical")
            ref_bounds = cluster_bounds(placed_parts, plate_poly)

            cand = place_item_near_reference_vertical(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                ref_bounds=ref_bounds,
            )

            if cand is None:
                cand = place_item_bottom_left(
                    item=item,
                    placed_parts=placed_parts,
                    plate_poly=plate_poly,
                    preferred_angles=[prof["angle"], 0, 90],
                )

            if cand is not None:
                placed_parts.append(cand)

    def place_fillers():
        for item in fillers:
            cand = place_item_generic(item, placed_parts, plate_poly)
            if cand is not None:
                placed_parts.append(cand)

    if variant == "A":
        place_shelf(top_shelf)
        place_shelf(second_shelf)

        placed_parts = place_trapezoids_min_total_space(
            items=trapezoids,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
        )

        place_right_bars()
        place_fillers()
    else:
        placed_parts = place_trapezoids_min_total_space(
            items=trapezoids,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
        )

        place_shelf(top_shelf)
        place_shelf(second_shelf)
        place_right_bars()
        place_fillers()

    placed_parts, _ = place_circles_best_pattern(
        parts=parts,
        plate_poly=plate_poly,
        placed_parts=placed_parts,
        circle_count=len(circles),
    )

    return placed_parts


def place_parts_mixed_template_from_parts_list(parts, parts_list, plate_poly):
    layout_a = _place_mixed_template_attempt(parts, parts_list, plate_poly, variant="A")
    layout_b = _place_mixed_template_attempt(parts, parts_list, plate_poly, variant="B")

    return layout_a if layout_rank_key(layout_a) >= layout_rank_key(layout_b) else layout_b

def build_remaining_parts_json(parts, fixed_parts):
    """
    Build a reduced JSON-like part list after subtracting already-fixed parts.
    """
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

def overall_bbox_area(placed_parts):
    if not placed_parts:
        return 0.0

    u = unary_union([p["poly"] for p in placed_parts]).buffer(0)
    if u.is_empty:
        return 0.0

    minx, miny, maxx, maxy = u.bounds
    return (maxx - minx) * (maxy - miny)


def layout_fill_ratio(placed_parts):
    """
    Used area / bounding-box area of the placed layout.
    This is the compactness you are looking at in the red region.
    """
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
    """
    Higher is better.
    1) more placed parts
    2) higher fill ratio
    3) smaller bounding box
    """
    return (
        placed_part_count(placed_parts),
        layout_fill_ratio(placed_parts),
        -overall_bbox_area(placed_parts),
    )

def build_non_circle_groups(parts_list):
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

    grouped = sorted(groups.values(), key=lambda g: -g["total_square"])
    return grouped, circles


def best_repeat_variant_for_group(group_items, region_width, region_height, angle_order=None):
    """
    Choose the orientation that fills available height best and still fits width.
    """
    if not group_items:
        return None

    if angle_order is None:
        angle_order = [0, 90]

    item = group_items[0]
    best = None

    for angle in angle_order:
        rotated = rotate_normalize(item["poly"], angle)
        w, h = poly_size(rotated)

        if w > region_width + 1e-6:
            continue
        if h > region_height + 1e-6:
            continue

        repeat_count = max(1, min(len(group_items), int(region_height // h)))
        used_h = repeat_count * h
        fill_ratio = used_h / region_height

        score = (
            fill_ratio >= 0.95,
            fill_ratio >= 0.90,
            fill_ratio,
            repeat_count,
            -w,
        )

        if best is None or score > best["score"]:
            best = {
                "angle": angle,
                "poly": rotated,
                "w": w,
                "h": h,
                "repeat_count": repeat_count,
                "fill_ratio": fill_ratio,
                "score": score,
            }

    return best

def choose_best_anchor_group(groups, region_width, region_height, strategy=None, top_k=5):
    """
    Height + width strategy:
    - prefer 95% / 90% height fill
    - prefer larger total square
    - when width is narrow, narrow rotated parts naturally win
    """
    strategy = strategy or {}
    angle_order = strategy.get("anchor_angle_order", [0, 90])

    best = None
    for group in groups[:top_k]:
        variant = best_repeat_variant_for_group(
            group["items"],
            region_width=region_width,
            region_height=region_height,
            angle_order=angle_order,
        )
        if variant is None:
            continue

        score = (
            variant["fill_ratio"] >= 0.95,
            variant["fill_ratio"] >= 0.90,
            variant["fill_ratio"],
            group["total_square"],
            variant["repeat_count"],
            -variant["w"],
        )

        cand = {
            "group": group,
            "variant": variant,
            "score": score,
        }

        if best is None or score > best["score"]:
            best = cand

    return best


def remove_first_n_by_base(items, base_id, n):
    out = []
    removed = 0

    for item in items:
        if item["base_id"] == base_id and removed < n:
            removed += 1
            continue
        out.append(item)

    return out


def try_place_item_in_box(item, placed_parts, box_poly, angle_order=None):
    """
    Greedy placement inside a rectangular sub-box.
    """
    if angle_order is None:
        angle_order = [0, 90]

    if item["shape"] == "rectangle":
        cand = place_item_top_left(
            item=item,
            placed_parts=placed_parts,
            plate_poly=box_poly,
            preferred_angles=angle_order,
        )
        if cand is not None:
            return cand

        cand = place_item_bottom_left(
            item=item,
            placed_parts=placed_parts,
            plate_poly=box_poly,
            preferred_angles=angle_order,
        )
        if cand is not None:
            return cand

    if item["shape"] == "trapezoid":
        cand = place_item_bottom_left(
            item=item,
            placed_parts=placed_parts,
            plate_poly=box_poly,
            preferred_angles=angle_order,
        )
        if cand is not None:
            return cand

    cand = place_item_bottom_left(
        item=item,
        placed_parts=placed_parts,
        plate_poly=box_poly,
        preferred_angles=angle_order,
    )
    if cand is not None:
        return cand

    return place_item_top_left(
        item=item,
        placed_parts=placed_parts,
        plate_poly=box_poly,
        preferred_angles=angle_order,
    )

def fill_box_greedily(remaining_items, placed_parts, region_box, strategy=None, max_passes=4):
    """
    Fill leftover box above an anchor band.
    """
    strategy = strategy or {}
    angle_order = strategy.get("fill_angle_order", [0, 90])

    items = sorted(list(remaining_items), key=lambda p: (-p["area"], p["priority"]))

    for _ in range(max_passes):
        progress = False
        new_items = []

        for item in items:
            cand = try_place_item_in_box(
                item=item,
                placed_parts=placed_parts,
                box_poly=region_box,
                angle_order=angle_order,
            )
            if cand is not None:
                placed_parts.append(cand)
                progress = True
                print(
                    f"Placed part {cand['id']} "
                    f"(shape={cand['shape']}) "
                    f"at x={cand['x']}, y={cand['y']}, angle={cand['angle']}"
                )
            else:
                new_items.append(item)

        items = new_items
        if not progress:
            break

    return placed_parts, items

def place_anchor_band(anchor_choice, remaining_items, placed_parts, region_box):
    """
    Place the chosen anchor group bottom-up inside the current region.
    """
    group = anchor_choice["group"]
    variant = anchor_choice["variant"]
    base_id = group["base_id"]

    rminx, rminy, rmaxx, rmaxy = region_box.bounds
    band_width = variant["w"]
    band_box = box(rminx, rminy, rminx + band_width, rmaxy)

    band_items = [p for p in remaining_items if p["base_id"] == base_id]
    repeat_count = min(len(band_items), variant["repeat_count"])

    y_cursor = rminy
    placed_count = 0

    for item in band_items[:repeat_count]:
        poly = translate(variant["poly"], xoff=rminx, yoff=y_cursor)

        if not valid_candidate(poly, placed_parts, band_box):
            continue

        placed_parts.append(
            {
                "id": item["id"],
                "base_id": item.get("base_id", item["id"]),
                "unit_count": item.get("unit_count", 1),
                "poly": poly,
                "x": rminx,
                "y": y_cursor,
                "angle": variant["angle"],
                "shape": item["shape"],
            }
        )

        print(
            f"Placed part {item['id']} "
            f"(shape={item['shape']}) "
            f"at x={rminx}, y={y_cursor}, angle={variant['angle']}"
        )

        y_cursor += variant["h"]
        placed_count += 1

    remaining_items = remove_first_n_by_base(remaining_items, base_id, placed_count)
    return placed_parts, remaining_items, band_box, y_cursor, placed_count

def place_non_circles_height_width_strategy(parts_list, plate_poly, strategy=None):
    """
    Height + width strategy for NON-CIRCLE parts only.
    """
    strategy = strategy or {}

    remaining_non_circles = [p for p in parts_list if p["shape"] != "circle"]
    placed_parts = []

    minx, miny, maxx, maxy = plate_poly.bounds
    x_cursor = minx

    while remaining_non_circles:
        region_width = maxx - x_cursor
        region_height = maxy - miny

        if region_width <= 1e-6 or region_height <= 1e-6:
            break

        current_groups, _ = build_non_circle_groups(remaining_non_circles)
        if not current_groups:
            break

        anchor_choice = choose_best_anchor_group(
            current_groups,
            region_width=region_width,
            region_height=region_height,
            strategy=strategy,
            top_k=strategy.get("anchor_top_k", 5),
        )

        if anchor_choice is None:
            break

        band_width = anchor_choice["variant"]["w"]
        if x_cursor + band_width > maxx + 1e-6:
            break

        region_box = box(x_cursor, miny, maxx, maxy)

        placed_parts, remaining_non_circles, band_box, used_top_y, placed_count = place_anchor_band(
            anchor_choice=anchor_choice,
            remaining_items=remaining_non_circles,
            placed_parts=placed_parts,
            region_box=region_box,
        )

        if placed_count == 0:
            break

        # fill leftover top box above the anchor band
        bminx, bminy, bmaxx, bmaxy = band_box.bounds
        if used_top_y < bmaxy - 1e-6:
            top_box = box(bminx, used_top_y, bmaxx, bmaxy)
            placed_parts, remaining_non_circles = fill_box_greedily(
                remaining_items=remaining_non_circles,
                placed_parts=placed_parts,
                region_box=top_box,
                strategy=strategy,
                max_passes=4,
            )

        x_cursor += band_width

    return placed_parts

def build_non_circle_parts(parts):
    return [dict(p) for p in parts if not is_circle_json_part(p)]

def build_parts_without_circles(parts):
    return [dict(p) for p in parts if not is_circle_json_part(p)]

def count_placed_circles(placed_parts):
    return sum(int(p.get("unit_count", 1)) for p in placed_parts if p["shape"] == "circle")

def get_circle_count(parts):
    return sum(int(p.get("quantity", 1)) for p in parts if is_circle_json_part(p))

def candidate_circle_strip_widths(parts, plate_poly, circle_count, max_cols=5):
    """
    Try a few right-strip widths and keep those that can place ALL circles.
    """
    _, radius, diameter, _ = get_circle_template(parts)
    if radius is None or circle_count <= 0:
        return []

    minx, miny, maxx, maxy = plate_poly.bounds
    plate_width = maxx - minx

    widths = []
    for cols in range(1, max_cols + 1):
        strip_w = cols * diameter
        if strip_w >= plate_width:
            break

        strip_box = box(maxx - strip_w, miny, maxx, maxy)
        _, placed_cnt = place_circles_best_pattern(
            parts=parts,
            plate_poly=strip_box,
            placed_parts=[],
            circle_count=circle_count,
        )

        if placed_cnt == circle_count:
            widths.append(strip_w)

    return widths

def place_parts_with_reserved_circle_strip(parts, plate_poly, strategy=None):
    """
    Reserve a right strip just for circles.
    Then place all non-circles in the left region.
    """
    strategy = strategy or {}

    circle_count = sum(int(p.get("quantity", 1)) for p in parts if is_circle_json_part(p))
    if circle_count <= 0:
        parts_list = build_parts_list(parts)
        return place_non_circles_height_width_strategy(parts_list, plate_poly, strategy=strategy)

    minx, miny, maxx, maxy = plate_poly.bounds
    strip_widths = candidate_circle_strip_widths(parts, plate_poly, circle_count)

    if not strip_widths:
        parts_list = build_parts_list(parts)
        return place_non_circles_height_width_strategy(parts_list, plate_poly, strategy=strategy)

    non_circle_parts = build_parts_without_circles(parts)

    best_layout = []
    best_key = None

    for strip_w in strip_widths:
        left_box = box(minx, miny, maxx - strip_w, maxy)
        right_box = box(maxx - strip_w, miny, maxx, maxy)

        left_parts_list = build_parts_list(non_circle_parts)
        left_layout = place_non_circles_height_width_strategy(
            left_parts_list,
            left_box,
            strategy=strategy,
        )

        full_layout, _ = place_circles_best_pattern(
            parts=parts,
            plate_poly=right_box,
            placed_parts=left_layout,
            circle_count=circle_count,
        )

        circles_placed = count_placed_circles(full_layout)
        key = (
            circles_placed == circle_count,
            placed_part_count(full_layout),
            layout_quality_score(full_layout),
        )

        print(
            f"reserved circle strip width={strip_w:.2f}, "
            f"circles={circles_placed}/{circle_count}, "
            f"placed={placed_part_count(full_layout)}"
        )

        if best_key is None or key > best_key:
            best_key = key
            best_layout = full_layout

    return best_layout

def candidate_right_strip_widths(parts, plate_poly):
    """
    Try a few realistic widths for a dedicated right-side circle strip.
    """
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

    widths = sorted(
        w for w in widths
        if 2.0 * radius <= w <= 0.45 * plate_width
    )

    return widths


def solve_with_right_circle_strip(parts, plate_poly, strip_width, strategy=None):
    """
    Reserve a right strip for circles.
    Place non-circles in the left region, circles only in the strip.
    """
    minx, miny, maxx, maxy = plate_poly.bounds
    plate_width = maxx - minx

    if strip_width <= 0 or strip_width >= plate_width:
        return []

    left_box = box(minx, miny, maxx - strip_width, maxy)
    right_box = box(maxx - strip_width, miny, maxx, maxy)

    non_circle_parts = build_non_circle_parts(parts)

    # Place non-circles only
    left_layout = place_parts_with_existing(non_circle_parts, left_box, strategy=strategy)

    # Place circles only in the right strip
    circle_count = get_circle_count(parts)
    final_layout, _ = place_circles_best_pattern(
        parts=parts,
        plate_poly=right_box,
        placed_parts=left_layout,
        circle_count=circle_count,
    )

    return final_layout

def layout_quality_score(placed_parts):
    if not placed_parts:
        return (0, float("-inf"), float("-inf"))

    units = placed_part_count(placed_parts)

    u = unary_union([p["poly"] for p in placed_parts]).buffer(0)
    minx, miny, maxx, maxy = u.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    waste = bbox_area - u.area

    return (units, -bbox_area, -waste)


def classify_layout_mode(parts_list):
    circles = [p for p in parts_list if p["shape"] == "circle"]
    trapezoids = [p for p in parts_list if p["shape"] == "trapezoid"]
    rectangles = [p for p in parts_list if p["shape"] == "rectangle"]
    non_circles = [p for p in parts_list if p["shape"] != "circle"]

    trap_base_ids = {p["base_id"] for p in trapezoids}

    if (
        non_circles
        and all(p["shape"] == "trapezoid" for p in non_circles)
        and len(trap_base_ids) == 1
        and len(trapezoids) >= 6
    ):
        return "trapezoid_strip"

    if (
        circles
        and len(trapezoids) == 1
        and trapezoids[0].get("unit_count", 1) == 2
        and len(rectangles) >= 2
        and len(non_circles) <= 5
    ):
        return "mixed_template"

    return "generic"


def build_requested_sequence(parts_list):
    rectangle_groups = {}
    trapezoid_pairs = []
    trapezoid_singles = []
    other_polys = []
    circles = []

    for item in parts_list:
        shape = item["shape"]
        if shape == "circle":
            circles.append(item)
        elif shape == "rectangle":
            rectangle_groups.setdefault(item["base_id"], []).append(item)
        elif shape == "trapezoid":
            if int(item.get("unit_count", 1)) == 2:
                trapezoid_pairs.append(item)
            else:
                trapezoid_singles.append(item)
        else:
            other_polys.append(item)

    rect_group_list = []
    for base_id, items in rectangle_groups.items():
        prof = orientation_profile(items[0], "horizontal")
        rect_group_list.append(
            {
                "base_id": base_id,
                "items": list(items),
                "width": prof["w"],
                "height": prof["h"],
                "angle": prof["angle"],
                "total_area": sum(p["area"] for p in items),
            }
        )

    rect_group_list.sort(
        key=lambda g: (-g["width"], -g["total_area"], g["base_id"])
    )
    trapezoid_pairs = sorted(trapezoid_pairs, key=lambda p: (-p["area"], p["base_id"]))
    trapezoid_singles = sorted(trapezoid_singles, key=lambda p: (-p["area"], p["base_id"]))
    other_polys = sorted(other_polys, key=lambda p: (p["priority"], -p["area"], p["base_id"]))

    return rect_group_list, trapezoid_pairs, trapezoid_singles, other_polys, circles


def place_rectangles_longest_width_first(rect_group_list, placed_parts, plate_poly, strategy=None):
    strategy = strategy or {}
    max_candidates = strategy.get("rect_max_candidates", 12)
    lookahead_depth = strategy.get("rect_lookahead_depth", 3)

    if rect_group_list:
        order_msg = ", ".join(
            f"{g['base_id']}({g['width']:.2f})" for g in rect_group_list
        )
        print(f"Rectangle group order by longest horizontal width: {order_msg}")

    for group in rect_group_list:
        preferred_angles = [group["angle"], 0, 90]
        group_items = list(group["items"])

        for idx, item in enumerate(group_items):
            remaining_items = group_items[idx + 1:]

            candidate = place_item_with_lookahead(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                remaining_items=remaining_items,
                mode="top_left",
                preferred_angles=preferred_angles,
                max_candidates=max_candidates,
                lookahead_depth=lookahead_depth,
            )

            if candidate is None:
                candidate = place_item_top_left(
                    item=item,
                    placed_parts=placed_parts,
                    plate_poly=plate_poly,
                    preferred_angles=preferred_angles,
                )

            if candidate is None:
                candidate = place_item_bottom_left(
                    item=item,
                    placed_parts=placed_parts,
                    plate_poly=plate_poly,
                    preferred_angles=preferred_angles,
                )

            if candidate is not None:
                placed_parts.append(candidate)
                print(
                    f"Placed part {candidate['id']} "
                    f"(shape={candidate['shape']}) "
                    f"at x={candidate['x']}, y={candidate['y']}, angle={candidate['angle']}"
                )
            else:
                print(f"Could not place rectangle part {item['id']}, not enough space")

    return placed_parts


def place_other_polygons_after_trapezoids(other_items, placed_parts, plate_poly, strategy=None):
    strategy = strategy or {}

    for idx, item in enumerate(other_items):
        remaining_items = other_items[idx + 1:]
        preferred_angles = strategy.get("other_angles", [0, 90])
        mode = "top_left" if item["shape"] == "rectangle" else "bottom_left"

        candidate = place_item_with_lookahead(
            item=item,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            remaining_items=remaining_items,
            mode=mode,
            preferred_angles=preferred_angles,
            max_candidates=10,
            lookahead_depth=2,
        )

        if candidate is None:
            candidate = place_item_generic(item, placed_parts, plate_poly)

        if candidate is not None:
            placed_parts.append(candidate)
            print(
                f"Placed part {candidate['id']} "
                f"(shape={candidate['shape']}) "
                f"at x={candidate['x']}, y={candidate['y']}, angle={candidate['angle']}"
            )
        else:
            print(f"Could not place part {item['id']}, not enough space")

    return placed_parts


def place_parts_in_requested_sequence(parts, parts_list, plate_poly, strategy=None):
    rect_group_list, trapezoid_pairs, trapezoid_singles, other_polys, circles = build_requested_sequence(parts_list)

    placed_parts = []

    # 1) rectangles: longest horizontal width first (e.g. part1 before part4 before part2)
    placed_parts = place_rectangles_longest_width_first(
        rect_group_list,
        placed_parts,
        plate_poly,
        strategy=strategy,
    )

    # 2) trapezoids: pairs first, then singles
    trapezoid_items = trapezoid_pairs + trapezoid_singles
    if trapezoid_items:
        placed_parts = place_trapezoids_min_total_space(
            items=trapezoid_items,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            later_items=other_polys,
        )

    # 3) other polygons
    placed_parts = place_other_polygons_after_trapezoids(
        other_polys,
        placed_parts,
        plate_poly,
        strategy=strategy,
    )

    # 4) circles last
    if circles:
        placed_parts, _ = place_circles_best_pattern(
            parts=parts,
            plate_poly=plate_poly,
            placed_parts=placed_parts,
            circle_count=len(circles),
        )

    return placed_parts

def place_parts_trapezoid_strip_from_parts_list(parts, parts_list, plate_poly):
    trapezoids = sorted(
        [p for p in parts_list if p["shape"] == "trapezoid"],
        key=lambda p: (-int(p.get("unit_count", 1)), -p["area"])
    )
    circles = [p for p in parts_list if p["shape"] == "circle"]
    others = [p for p in parts_list if p["shape"] not in ("trapezoid", "circle")]
    pair_items = [p for p in trapezoids if int(p.get("unit_count", 1)) == 2]
    single_items = [p for p in trapezoids if int(p.get("unit_count", 1)) == 1]

    placed_parts = []

    # FIRST: many trapezoid pairs -> compact parallelogram strip
    if pair_items:
        placed_parts = place_identical_trapezoids_global(
            items=pair_items,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            beam_width=MAX_TRAPEZOID_BEAM,
        )

    # THEN: leftover single trapezoids
    if single_items:
        placed_parts = place_identical_trapezoids_global(
            items=single_items,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            beam_width=MAX_TRAPEZOID_BEAM,
        )

    # THEN: any non-trapezoid leftovers
    for item in others:
        candidate = place_item_generic(item, placed_parts, plate_poly)
        if candidate:
            placed_parts.append(candidate)

    # LAST: circles
    if circles:
        placed_parts, _ = place_circles_best_pattern(
            parts=parts,
            plate_poly=plate_poly,
            placed_parts=placed_parts,
            circle_count=len(circles),
        )

    return placed_parts

# -----------------------------
# Main nesting
# -----------------------------
def nest_parts_with_full_fit(parts, plate_poly, max_rounds=FAST_MAX_ROUNDS):
    expected = expected_part_count(parts)

    best_layout = []
    best_key = (-1, -1.0, float("-inf"))
    solve_cache = {}

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

    def is_full(layout):
        return placed_part_count(layout) == expected

    print(f"\n=== ROUND 1 / {max_rounds} ===")
    base_layout = place_parts_with_existing(parts, plate_poly, _cache=solve_cache)
    consider(base_layout, "baseline")
    if is_full(base_layout):
        return base_layout

    print("\n=== STRIP SOLVE AFTER DENSE GROUP ===")
    strip_layout = solve_remaining_strip_after_dense_group(parts, plate_poly, base_layout)
    consider(strip_layout, "dense-group-strip")
    if is_full(strip_layout):
        return strip_layout

    print("\n=== RIGHT CIRCLE STRIP COMPACTION ===")
    strip_widths = sorted(set(round(w, 6) for w in candidate_right_strip_widths(parts, plate_poly)))

    compact_strategies = [
        {"name": "default"},
        {
            "name": "reverse_order",
            "side_bar_order": "reverse",
            "trapezoid_order": "reverse",
            "filler_order": "reverse",
        },
        {
            "name": "largest_first_inside_groups",
            "side_bar_order": "largest_first",
            "trapezoid_order": "largest_first",
            "filler_order": "largest_first",
        },
        {
            "name": "rotate_first",
            "top_angles": [0, 180],
            "side_angles": [0, 180],
            "filler_angles": [0, 180],
        },
    ]

    seen_strip_runs = set()
    for w in strip_widths:
        for strategy in compact_strategies:
            run_key = (round(w, 6), freeze_strategy(strategy))
            if run_key in seen_strip_runs:
                continue
            seen_strip_runs.add(run_key)

            print(f"Trying right strip width={w:.2f}, strategy={strategy.get('name', 'custom')}")
            layout = solve_with_right_circle_strip(parts, plate_poly, w, strategy=strategy)
            consider(layout, f"right-strip-{w:.0f}-{strategy.get('name', 'custom')}")
            if is_full(layout):
                return layout

    strategies = [
        {"name": "default"},
        {
            "name": "reverse_order",
            "side_bar_order": "reverse",
            "trapezoid_order": "reverse",
            "filler_order": "reverse",
        },
        {
            "name": "largest_first_inside_groups",
            "side_bar_order": "largest_first",
            "trapezoid_order": "largest_first",
            "filler_order": "largest_first",
        },
        {
            "name": "smallest_first_inside_groups",
            "side_bar_order": "smallest_first",
            "trapezoid_order": "smallest_first",
            "filler_order": "smallest_first",
        },
        {
            "name": "rotate_first",
            "top_angles": [0, 90],
            "side_angles": [0, 90],
            "filler_angles": [0, 90],
        },
    ]

    seen_retry_runs = set()
    unique_strategies = []
    for s in strategies:
        key = freeze_strategy(s)
        if key not in seen_retry_runs:
            seen_retry_runs.add(key)
            unique_strategies.append(s)

    for round_idx, strategy in enumerate(unique_strategies[:max_rounds - 1], start=2):
        print(f"\n=== ROUND {round_idx} / {max_rounds} ===")
        print(f"Strategy: {strategy.get('name', 'custom')}")

        layout = place_parts_with_existing(parts, plate_poly, strategy=strategy, _cache=solve_cache)
        consider(layout, f"retry-{strategy.get('name', 'custom')}")
        if is_full(layout):
            return layout

    best_count = placed_part_count(best_layout)

    if best_count == expected:
        return best_layout

    raise NestingFailed(
        f"Could not place all parts. Best result was {best_count}/{expected}.",
        best_layout=best_layout,
        best_count=best_count,
        expected_count=expected,
    )

def place_parts_with_existing(parts, plate_poly, strategy=None, _cache=None):
    cache_key = None
    if _cache is not None:
        cache_key = (id(parts), freeze_bounds(plate_poly), freeze_strategy(strategy))
        if cache_key in _cache:
            return list(_cache[cache_key])

    parts_list = build_parts_list(parts)
    mode = classify_layout_mode(parts_list)

    print(f"Layout mode: {mode}")

    ordered_layout = place_parts_requested_priority(
        parts=parts,
        parts_list=parts_list,
        plate_poly=plate_poly,
        strategy=strategy,
    )

    if mode == "trapezoid_strip":
        strip_layout = place_parts_trapezoid_strip_from_parts_list(parts, parts_list, plate_poly)
        layout = strip_layout if layout_rank_key(strip_layout) >= layout_rank_key(ordered_layout) else ordered_layout

    elif mode == "mixed_template":
        mixed_layout = place_parts_mixed_template_from_parts_list(parts, parts_list, plate_poly)
        layout = mixed_layout if layout_rank_key(mixed_layout) >= layout_rank_key(ordered_layout) else ordered_layout

    else:
        layout = ordered_layout

    if _cache is not None:
        _cache[cache_key] = list(layout)

    return layout

# -----------------------------
# Plot arrangement
# -----------------------------
def plot_arrangement(placed_parts, plate_poly):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect("equal")

    xs, ys = plate_poly.exterior.xy
    ax.plot(xs, ys, "black")

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

    ax.set_title("Layout Filled According to Geometry-Role Template")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# -----------------------------
# Summary table
# -----------------------------
def generate_table(placed_parts):
    summary = {}
    for p in placed_parts:
        key = get_part_key(p)
        count = int(p.get("unit_count", 1))
        summary[key] = summary.get(key, 0) + count

    print(f"{'ID':<12} {'Placed Count':<12}")
    for pid, count in summary.items():
        print(f"{pid:<12} {count:<12}")


# -----------------------------
# GUI
# -----------------------------
class NestingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Nesting with Full-Fit Validation")

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

            placed = placed_part_count(placed_parts)
            expected = expected_part_count(self.parts)

            if placed == expected:
                plot_arrangement(placed_parts, self.plate_poly)
                generate_table(placed_parts)
                messagebox.showinfo(
                    "Nesting Success",
                    f"All parts were placed successfully: {placed}/{expected}"
                )
                return

            # fallback, just in case
            plot_arrangement(placed_parts, self.plate_poly)
            generate_table(placed_parts)
            messagebox.showwarning(
                "Nesting incomplete",
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
            return
       

if __name__ == "__main__":
    root = tk.Tk()
    app = NestingApp(root)
    root.mainloop()