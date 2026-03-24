import json
import math
import queue
import threading
import traceback
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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

FAST_PAIR_GRID_STEPS = 9#18
FAST_CIRCLE_PHASE_SAMPLES = 20 #40
FAST_BRANCH_CANDIDATES =  5 #10
FAST_LOOKAHEAD_DEPTH = 1#3
FAST_MAX_ROUNDS = 5
FAST_STRIP_STRATEGIES = 2
MAX_TRAPEZOID_BEAM = 60 #120
TRAPEZOID_PAIR_STYLE = "parallelogram_first"

EXACT_GRID_STEPS = (1.0, 0.5, 0.25, 0.2, 0.1)
EXACT_GRID_MAX_BOARD_CELLS = 400
EXACT_GRID_MAX_TOTAL_PIECE_CELLS = 220
EXACT_GRID_MAX_PIECES = 18
EXACT_GRID_TOL = 1e-6

def freeze_strategy(strategy):
    if strategy is None:
        return None
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in strategy.items()))

def freeze_bounds(poly):
    return tuple(round(v, 6) for v in poly.bounds)

ACTIVE_PLACEMENT_OBSERVER = None


def set_placement_observer(observer):
    global ACTIVE_PLACEMENT_OBSERVER
    ACTIVE_PLACEMENT_OBSERVER = observer


def notify_placement_observer(placed_parts, placed_part=None):
    observer = ACTIVE_PLACEMENT_OBSERVER
    if not callable(observer):
        return
    try:
        observer(list(placed_parts), placed_part)
    except Exception as exc:
        print(f"Placement observer failed: {exc}")


def log_placed_part(placed_part):
    print(
        f"Placed part {placed_part['id']} "
        f"(shape={placed_part['shape']}) "
        f"at x={placed_part['x']}, y={placed_part['y']}, angle={placed_part['angle']}"
    )


def commit_placed_part(target_list, placed_part, announce=True):
    target_list.append(placed_part)
    if announce:
        log_placed_part(placed_part)
    notify_placement_observer(target_list, placed_part)
    return placed_part


def notify_layout_extension(previous_layout, current_layout, announce=False):
    if current_layout is None:
        return None
    temp_layout = list(previous_layout)
    for placed_part in current_layout[len(previous_layout):]:
        temp_layout.append(placed_part)
        if announce:
            log_placed_part(placed_part)
        notify_placement_observer(temp_layout, placed_part)
    return current_layout

# -----------------------------
# Load JSON
# -----------------------------
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Geometry conversion
# -----------------------------



def _vertex_xy(vertex):
    return float(vertex.get("x", 0.0)), float(vertex.get("y", 0.0))


def _contour_points(contour):
    points = contour.get("points") or []
    xy = []
    for pt in points:
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            xy.append(_vertex_xy(pt))
    return xy


def _is_circle_contour(contour):
    points = contour.get("points") or []
    return len(points) == 1 and float(points[0].get("radius", 0) or 0) > 0


def contour_to_polygon(contour, samples=72):
    """
    Convert one SIOP_CONTOUR into a polygon.
    Supports standard closed contours and one-point circle contours.
    Arc segments are left for a later step; current step keeps the straight-point chain.
    """
    points = contour.get("points") or []
    if not points:
        return None

    if _is_circle_contour(contour):
        center = points[0]
        r = float(center.get("radius", 0) or 0)
        cx, cy = _vertex_xy(center)
        angle = np.linspace(0, 2 * np.pi, samples, endpoint=False)
        return ShapelyPolygon([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angle])

    xy = _contour_points(contour)
    if len(xy) < 3:
        return None

    ring = list(xy)
    if contour.get("bclose", False) and ring[0] != ring[-1]:
        ring.append(ring[0])

    if len(ring) < 4:
        return None

    poly = ShapelyPolygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return None if poly.is_empty else poly


def contours_to_polygon(contours):
    """
    Convert SIOP_PART.contours or SIOP_PLATE.contours into one polygon.
    First contour is the outer contour; the rest are inner contours / holes.
    """
    if not isinstance(contours, list) or not contours:
        return None

    outer_poly = contour_to_polygon(contours[0])
    if outer_poly is None:
        return None

    holes = []
    for contour in contours[1:]:
        hole_poly = contour_to_polygon(contour)
        if hole_poly is None:
            continue
        holes.append(list(hole_poly.exterior.coords))

    if not holes:
        return outer_poly

    poly = ShapelyPolygon(list(outer_poly.exterior.coords), holes=holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return outer_poly if poly.is_empty else poly


def plate_to_polygon(plate):
    return contours_to_polygon((plate or {}).get("contours") or [])


def build_result_json(placed_parts, plate_id="1", quantity=1):
    """
    Foundation for the documented result types:
    SIOP_PARTINSERT -> SIOP_RESULT_PLATE -> SIOP_RESULTS
    """
    result_parts = []
    for p in placed_parts:
        result_parts.append({
            "id": str(p.get("base_id", p.get("id", ""))),
            "mirrored": False,
            "angle": math.radians(float(p.get("angle", 0.0))),
            "move": {
                "x": float(p.get("x", 0.0)),
                "y": float(p.get("y", 0.0)),
                "radius": 0.0,
                "angle1": 0.0,
                "angle2": 0.0,
            },
            "arc_strikes": [],
        })

    return {
        "plates": [
            {
                "id": str(plate_id),
                "parts": result_parts,
                "quantity": int(quantity),
            }
        ]
    }


def _find_numeric_value_case_insensitive(obj, target_keys):
    """Recursively search for the first numeric value whose key matches target_keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = str(key).strip().lower().replace("_", "")
            if key_norm in target_keys and isinstance(value, (int, float)):
                return float(value)
        for value in obj.values():
            found = _find_numeric_value_case_insensitive(value, target_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_numeric_value_case_insensitive(item, target_keys)
            if found is not None:
                return found
    return None


def extract_plate_size(data):
    """
    Best-effort extraction of plate width/height from the JSON payload.
    Supports common shapes like:
      - {"plate": {"width": 1000, "height": 700}}
      - {"plates": [{"width": 1000, "height": 700}]}
      - {"stock": {"w": 1000, "h": 700}}
      - {"plates": [{"contours": [{"points": [...]}]}]}  # infer from contour bounds
      - nested fields with Chinese aliases.
    """
    width_keys = {
        "width", "platewidth", "sheetwidth", "boardwidth", "stockwidth",
        "w", "xsize", "sizex", "lenx", "platew",
        "板宽", "板材宽", "宽",
    }
    height_keys = {
        "height", "plateheight", "sheetheight", "boardheight", "stockheight",
        "h", "ysize", "sizey", "leny", "plateh",
        "板高", "板材高", "高",
    }

    def _contour_bounds_size(obj):
        if not isinstance(obj, dict):
            return None, None
        contours = obj.get("contours")
        if not isinstance(contours, list) or not contours:
            return None, None

        xs = []
        ys = []
        for contour in contours:
            if not isinstance(contour, dict):
                continue
            points = contour.get("points")
            if not isinstance(points, list):
                continue
            for pt in points:
                if not isinstance(pt, dict):
                    continue
                x = pt.get("x")
                y = pt.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    xs.append(float(x))
                    ys.append(float(y))

        if len(xs) >= 2 and len(ys) >= 2:
            return max(xs) - min(xs), max(ys) - min(ys)
        return None, None

    preferred_containers = []
    if isinstance(data, dict):
        for container_key in ("plate", "plates", "stock", "stocks", "sheet", "sheets", "board", "boards", "material", "materials"):
            if container_key in data:
                preferred_containers.append(data[container_key])

    search_spaces = preferred_containers + [data]

    for space in search_spaces:
        width = _find_numeric_value_case_insensitive(space, width_keys)
        height = _find_numeric_value_case_insensitive(space, height_keys)
        if width is not None and height is not None:
            return float(width), float(height)

        if isinstance(space, list):
            for item in space:
                plate_poly = plate_to_polygon(item) if isinstance(item, dict) else None
                if plate_poly is not None:
                    minx, miny, maxx, maxy = plate_poly.bounds
                    return float(maxx - minx), float(maxy - miny)
                width, height = _contour_bounds_size(item)
                if width is not None and height is not None:
                    return float(width), float(height)
        else:
            plate_poly = plate_to_polygon(space) if isinstance(space, dict) else None
            if plate_poly is not None:
                minx, miny, maxx, maxy = plate_poly.bounds
                return float(maxx - minx), float(maxy - miny)
            width, height = _contour_bounds_size(space)
            if width is not None and height is not None:
                return float(width), float(height)

    return None, None

def part_to_polygon(part):
    contours = (part or {}).get("contours") or []
    poly = contours_to_polygon(contours)
    if poly is None:
        raise ValueError(f"Part {part.get('id', '?')} has no valid contours")
    return poly


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
def _canonical_trapezoid_vertices(poly, tol=1e-4):
    """
    Return the trapezoid in the canonical order [a, b, c, d] where:
    - a->b and d->c are the parallel side walls
    - a and d are the lower endpoints
    - a is the left wall and d is the right wall

    This matches the user's contour example:
      [bottom-left, upper-left, upper-right, bottom-right]

    We try both windings and every cyclic shift because Shapely may change the
    starting vertex and orientation.
    """
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) != 4:
        return None

    orderings = [coords, list(reversed(coords))]

    for ordered in orderings:
        for shift in range(4):
            cand = ordered[shift:] + ordered[:shift]
            a, b, c, d = cand
            if not is_parallel(edge_vector(a, b), edge_vector(d, c), tol):
                continue
            if b[1] + tol < a[1] or c[1] + tol < d[1]:
                continue
            if a[0] > d[0] + tol:
                continue
            return [a, b, c, d]

    return None


def _combine_trapezoids_fallback(poly):
    """
    Preserve the old search-based pairing as a fallback.
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


def _rotate_point(pt, angle_rad):
    x, y = pt
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    return (x * ca - y * sa, x * sa + y * ca)


def _build_sharp_trapezoid_pair(poly):
    """
    Build the paired trapezoid unit requested by the user.

    Let the first trapezoid be [a, b, c, d].
    The second trapezoid is placed as [c1=b, d1, a1, b1=c].

    We therefore compute only:
      d1 = b + (c - d)
      a1 = d1 + (d - a)

    The paired outer shell is the parallelogram:
      [a, d1, a1, d]

    We also keep the two inner trapezoid contours so the plot can show both
    trapezoids sharply inside the parallelogram.
    """
    verts = _canonical_trapezoid_vertices(poly)
    if verts is None:
        return None

    a, b, c, d = verts
    base_angle = math.atan2(d[1] - a[1], d[0] - a[0])

    def to_local(pt):
        return _rotate_point((pt[0] - a[0], pt[1] - a[1]), -base_angle)

    A, B, C, D = [to_local(pt) for pt in (a, b, c, d)]

    d1 = (B[0] + (C[0] - D[0]), B[1] + (C[1] - D[1]))
    a1 = (d1[0] + (D[0] - A[0]), d1[1] + (D[1] - A[1]))

    first_local = ShapelyPolygon([A, B, C, D])
    second_local = ShapelyPolygon([B, d1, a1, C])
    pair_local = ShapelyPolygon([A, d1, a1, D])

    if not pair_local.is_valid or pair_local.area <= poly.area + 1e-6:
        return None
    if not first_local.is_valid or not second_local.is_valid:
        return None

    pminx, pminy, _, _ = pair_local.bounds
    offset_x = -pminx
    offset_y = -pminy

    return {
        "poly": translate(pair_local, xoff=offset_x, yoff=offset_y),
        "display_local_polys": [
            translate(first_local, xoff=offset_x, yoff=offset_y),
            translate(second_local, xoff=offset_x, yoff=offset_y),
        ],
    }


def combine_trapezoids(poly, pair_style=None):
    """
    Create a compact pair from two trapezoids.

    Modes:
    - "parallelogram_first": use the computed parallelogram shell first,
      then fall back to the old search-based pair.
    - "old_only": skip the parallelogram shell and use only the old pair.
    """
    style = pair_style or TRAPEZOID_PAIR_STYLE

    if style != "old_only":
        sharp_pair = _build_sharp_trapezoid_pair(poly)
        if sharp_pair is not None:
            return sharp_pair

    fallback_poly = _combine_trapezoids_fallback(poly)
    if fallback_poly is None:
        return None
    return {"poly": fallback_poly, "display_local_polys": None}

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

    Important behavior:
    once one copy of the current family fails on this plate,
    stop trying the rest of that family on this plate and move on.
    """
    strategy = strategy or {}
    angle_order = strategy.get("fill_angle_order", [0, 90])

    items = sorted(group["items"], key=lambda p: (-p["area"], p["priority"]))
    family_base_id = group.get("base_id") or (items[0].get("base_id") if items else None)

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
            commit_placed_part(placed_parts, cand)
        else:
            log_family_stop_on_plate(family_base_id or item.get("base_id", item["id"]), item["id"])
            break

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

            pair_data = pair_cache[pid]
            pair_poly = pair_data["poly"] if pair_data is not None else None
            pair_display = pair_data.get("display_local_polys") if pair_data is not None else None
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
                            "display_local_polys": pair_display,
                            "source_poly": poly,
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
                            "source_poly": poly,
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
                            "source_poly": poly,
                        }
                    )

            continue

        # mixed layout -> make qty//2 pairs + maybe one single
        if pid not in pair_cache:
            pair_cache[pid] = combine_trapezoids(poly)

        pair_data = pair_cache[pid]
        pair_poly = pair_data["poly"] if pair_data is not None else None
        pair_display = pair_data.get("display_local_polys") if pair_data is not None else None
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
                        "display_local_polys": pair_display,
                        "source_poly": poly,
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


def get_rotated_display_polys(item, angle):
    local_polys = item.get("display_local_polys")
    if not local_polys:
        return None

    cache = item.setdefault("_display_rot_cache", {})
    if angle not in cache:
        base_poly = item["poly"]
        origin = tuple(base_poly.centroid.coords[0])
        rotated_base = rotate(base_poly, angle, origin=origin, use_radians=False)
        minx, miny, _, _ = rotated_base.bounds
        cache[angle] = [
            translate(rotate(g, angle, origin=origin, use_radians=False), xoff=-minx, yoff=-miny)
            for g in local_polys
        ]
    return cache[angle]


def make_placed_part(item, candidate_poly, x, y, angle):
    placed = {
        "id": item["id"],
        "base_id": item.get("base_id", item["id"]),
        "unit_count": item.get("unit_count", 1),
        "poly": candidate_poly,
        "x": x,
        "y": y,
        "angle": angle,
        "shape": item["shape"],
    }

    display_local = get_rotated_display_polys(item, angle)
    if display_local:
        placed["display_polys"] = [translate(g, xoff=x, yoff=y) for g in display_local]

    return placed


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
                best = make_placed_part(item, candidate, x, y, angle)
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
                best = make_placed_part(item, candidate, x, y, angle)
                best_score = score

    return best

def get_part_key(p):
    return p.get("base_id", p.get("id", "unknown"))


def log_family_stop_on_plate(base_id, item_id=None, reason="not enough space"):
    shown = item_id if item_id is not None else base_id
    print(
        f"Could not place part {shown}, {reason}. "
        f"Skipping remaining copies of family {base_id} on this plate and trying other parts."
    )

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
                best = make_placed_part(item, candidate, x, y, angle)
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

            cand = make_placed_part(item, candidate_poly, x, y, angle)
            cand["local_score"] = local_score
            candidates.append(cand)

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

    temp_layout = list(placed_parts)
    for p in best_layout[len(placed_parts):]:
        log_placed_part(p)
        temp_layout.append(p)
        notify_placement_observer(temp_layout, p)

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
                best = make_placed_part(item, candidate, x, y, angle)
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
        def _try_pair_lane(poly_template, use_display=True):
            lane_best_layout = None
            lane_best_count = -1
            lane_best_used_width = float("inf")

            for ang in [0, 180]:
                rp = rotate_normalize(poly_template, ang)
                rw, rh = poly_size(rp)

                if rh > plate_height + 1e-6:
                    continue

                pair_pitch = _min_dx_no_overlap(rp, rp, 0.0)
                if pair_pitch <= 1e-6:
                    pair_pitch = rw

                x_cursor = minx
                right_extent = minx
                layout = list(placed_parts)
                placed_n = 0

                for item in items:
                    cand_poly = translate(rp, xoff=x_cursor, yoff=miny)
                    cminx, _, cmaxx, _ = cand_poly.bounds

                    if cmaxx > maxx + 1e-6:
                        break
                    if cminx < minx - 1e-6:
                        break
                    if not plate_poly.covers(cand_poly):
                        break
                    if has_real_overlap(cand_poly, layout):
                        break

                    item_for_place = dict(item)
                    item_for_place["poly"] = poly_template
                    if not use_display:
                        item_for_place.pop("display_local_polys", None)

                    layout.append(make_placed_part(item_for_place, cand_poly, x_cursor, miny, ang))
                    placed_n += 1
                    right_extent = max(right_extent, cmaxx)
                    x_cursor += pair_pitch

                used_width = right_extent - minx

                if (
                    placed_n > lane_best_count
                    or (placed_n == lane_best_count and used_width < lane_best_used_width)
                ):
                    lane_best_count = placed_n
                    lane_best_used_width = used_width
                    lane_best_layout = layout

            return lane_best_layout, lane_best_count, lane_best_used_width

        best_layout, best_count, best_used_width = _try_pair_lane(base_poly, use_display=True)

        need_old_pair_retry = (
            best_count < len(items)
            and items
            and items[0].get("display_local_polys")
            and items[0].get("source_poly") is not None
        )

        if need_old_pair_retry:
            fallback_poly = _combine_trapezoids_fallback(items[0]["source_poly"])
            if fallback_poly is not None:
                alt_layout, alt_count, alt_used_width = _try_pair_lane(fallback_poly, use_display=False)
                if (
                    alt_layout is not None
                    and (
                        alt_count > best_count
                        or (alt_count == best_count and alt_used_width < best_used_width)
                    )
                ):
                    best_layout = alt_layout
                    best_count = alt_count
                    best_used_width = alt_used_width

        if best_layout is not None:
            notify_layout_extension(list(placed_parts), best_layout)
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
                commit_placed_part(out, cand)
        return out

    cfg, sol = best_solution
    out = list(placed_parts)

    for item, (k, x_rel) in zip(items, sol["placements"]):
        x_abs = minx + x_rel
        candidate = translate(cfg[k]["poly"], xoff=x_abs, yoff=cfg[k]["y"])
        commit_placed_part(out, make_placed_part(item, candidate, x_abs, cfg[k]["y"], cfg[k]["angle"]))

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
            previous_out = list(out)
            out = place_identical_trapezoids_global(
                pair_items,
                out,
                plate_poly,
                beam_width=MAX_TRAPEZOID_BEAM,
            )
            notify_layout_extension(previous_out, out)
            current_traps = [p for p in out if p["shape"] == "trapezoid"]
            pair_items = []

    blocked_pair_base_ids = set()

    # leftover pair items
    for i, item in enumerate(pair_items):
        base_id = item.get("base_id", item["id"])
        if base_id in blocked_pair_base_ids:
            continue

        remaining_items = [p for p in pair_items[i + 1:] if p.get("base_id", p["id"]) not in blocked_pair_base_ids] + single_items + list(later_items)

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
            commit_placed_part(out, cand)
            current_traps.append(cand)
        else:
            blocked_pair_base_ids.add(base_id)
            log_family_stop_on_plate(base_id, item["id"])

    # many identical singles
    if single_items:
        same_base = len({p["base_id"] for p in single_items}) == 1
        if same_base and len(single_items) >= 6:
            previous_out = list(out)
            out = place_identical_trapezoids_global(
                single_items,
                out,
                plate_poly,
                beam_width=MAX_TRAPEZOID_BEAM,
            )
            notify_layout_extension(previous_out, out)
            return out

    blocked_single_base_ids = set()

    # leftover single trapezoids
    for i, item in enumerate(single_items):
        base_id = item.get("base_id", item["id"])
        if base_id in blocked_single_base_ids:
            continue

        remaining_items = [p for p in single_items[i + 1:] if p.get("base_id", p["id"]) not in blocked_single_base_ids] + list(later_items)

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
            commit_placed_part(out, cand)
            current_traps.append(cand)
        else:
            blocked_single_base_ids.add(base_id)
            log_family_stop_on_plate(base_id, item["id"])

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
                commit_placed_part(placed_parts, cand)

    def place_fillers():
        for item in fillers:
            cand = place_item_generic(item, placed_parts, plate_poly)
            if cand is not None:
                commit_placed_part(placed_parts, cand)

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

def _layout_union(placed_parts):
    if not placed_parts:
        return None

    u = unary_union([p["poly"] for p in placed_parts]).buffer(0)
    if u.is_empty:
        return None
    return u


def overall_bbox_area(placed_parts):
    u = _layout_union(placed_parts)
    if u is None:
        return 0.0

    minx, miny, maxx, maxy = u.bounds
    return (maxx - minx) * (maxy - miny)


def overall_bbox_width(placed_parts):
    u = _layout_union(placed_parts)
    if u is None:
        return 0.0

    minx, miny, maxx, maxy = u.bounds
    return maxx - minx


def overall_bbox_height(placed_parts):
    u = _layout_union(placed_parts)
    if u is None:
        return 0.0

    minx, miny, maxx, maxy = u.bounds
    return maxy - miny


def layout_waste_area(placed_parts):
    u = _layout_union(placed_parts)
    if u is None:
        return 0.0

    minx, miny, maxx, maxy = u.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    return max(0.0, bbox_area - u.area)


def layout_fill_ratio(placed_parts):
    """
    Used area / bounding-box area of the placed layout.
    """
    u = _layout_union(placed_parts)
    if u is None:
        return 0.0

    minx, miny, maxx, maxy = u.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    if bbox_area <= 1e-9:
        return 0.0

    return u.area / bbox_area


def layout_rank_key(placed_parts):
    """
    Higher is better.
    Main goal:
    1) place the maximum number of parts
    2) use the minimum space for those placed parts
    3) prefer denser / less wasteful layouts only as tie-breakers
    """
    return (
        placed_part_count(placed_parts),
        -overall_bbox_area(placed_parts),
        -overall_bbox_width(placed_parts),
        -overall_bbox_height(placed_parts),
        layout_fill_ratio(placed_parts),
        -layout_waste_area(placed_parts),
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
                commit_placed_part(placed_parts, cand)
                progress = True
            else:
                new_items.append(item)

        items = new_items
        if not progress:
            break

    return placed_parts, items

def place_anchor_band(anchor_choice, remaining_items, placed_parts, region_box):
    """
    Place the chosen anchor group bottom-up inside the current region.

    Important rule for the current plate:
    once one copy of this base_id stops fitting, stop this family immediately,
    keep the leftover copies for later plates, and move on to the next family.
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
    family_stopped = False

    for item in band_items[:repeat_count]:
        poly = translate(variant["poly"], xoff=rminx, yoff=y_cursor)

        if not valid_candidate(poly, placed_parts, band_box):
            log_family_stop_on_plate(base_id, item["id"])
            family_stopped = True
            break

        commit_placed_part(
            placed_parts,
            make_placed_part(item, poly, rminx, y_cursor, variant["angle"]),
        )

        y_cursor += variant["h"]
        placed_count += 1

    remaining_items = remove_first_n_by_base(remaining_items, base_id, placed_count)
    return placed_parts, remaining_items, band_box, y_cursor, placed_count, family_stopped, base_id

def place_non_circles_height_width_strategy(parts_list, plate_poly, strategy=None):
    """
    Height + width strategy for NON-CIRCLE parts only.
    """
    strategy = strategy or {}

    remaining_non_circles = [p for p in parts_list if p["shape"] != "circle"]
    placed_parts = []
    blocked_base_ids = set()

    minx, miny, maxx, maxy = plate_poly.bounds
    x_cursor = minx

    while remaining_non_circles:
        region_width = maxx - x_cursor
        region_height = maxy - miny

        if region_width <= 1e-6 or region_height <= 1e-6:
            break

        current_groups, _ = build_non_circle_groups(remaining_non_circles)
        current_groups = [g for g in current_groups if g["base_id"] not in blocked_base_ids]
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
            blocked_base_ids.add(anchor_choice["group"]["base_id"])
            continue

        region_box = box(x_cursor, miny, maxx, maxy)

        (
            placed_parts,
            remaining_non_circles,
            band_box,
            used_top_y,
            placed_count,
            family_stopped,
            stopped_base_id,
        ) = place_anchor_band(
            anchor_choice=anchor_choice,
            remaining_items=remaining_non_circles,
            placed_parts=placed_parts,
            region_box=region_box,
        )

        if family_stopped:
            blocked_base_ids.add(stopped_base_id)

        if placed_count == 0:
            blocked_base_ids.add(anchor_choice["group"]["base_id"])
            continue

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
        return (0, float("-inf"), float("-inf"), float("-inf"), float("-inf"))

    return (
        placed_part_count(placed_parts),
        -overall_bbox_area(placed_parts),
        -overall_bbox_width(placed_parts),
        layout_fill_ratio(placed_parts),
        -layout_waste_area(placed_parts),
    )


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
        base_id = group.get("base_id") or (group_items[0].get("base_id") if group_items else None)

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
                commit_placed_part(placed_parts, candidate)
            else:
                log_family_stop_on_plate(base_id or item.get("base_id", item["id"]), item["id"])
                break

    return placed_parts


def place_other_polygons_after_trapezoids(other_items, placed_parts, plate_poly, strategy=None):
    strategy = strategy or {}
    blocked_base_ids = set()

    for idx, item in enumerate(other_items):
        base_id = item.get("base_id", item["id"])
        if base_id in blocked_base_ids:
            continue

        remaining_items = [p for p in other_items[idx + 1:] if p.get("base_id", p["id"]) not in blocked_base_ids]
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
            commit_placed_part(placed_parts, candidate)
        else:
            blocked_base_ids.add(base_id)
            log_family_stop_on_plate(base_id, item["id"])

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
            commit_placed_part(placed_parts, candidate)

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

    global TRAPEZOID_PAIR_STYLE

    overall_best_layout = []
    overall_best_key = (-1, -1.0, float("-inf"))

    def consider_overall(layout, label):
        nonlocal overall_best_layout, overall_best_key
        key = layout_rank_key(layout)
        print(
            f"{label}: placed={placed_part_count(layout)}/{expected}, "
            f"fill_ratio={layout_fill_ratio(layout):.4f}, "
            f"bbox_area={overall_bbox_area(layout):.2f}"
        )
        if key > overall_best_key:
            overall_best_key = key
            overall_best_layout = layout

    def is_full(layout):
        return placed_part_count(layout) == expected

    def run_full_solver(pair_style, label_prefix):
        global TRAPEZOID_PAIR_STYLE
        TRAPEZOID_PAIR_STYLE = pair_style

        best_layout = []
        best_key = (-1, -1.0, float("-inf"))
        solve_cache = {}

        def consider(layout, label):
            nonlocal best_layout, best_key
            key = layout_rank_key(layout)
            if key > best_key:
                best_key = key
                best_layout = layout
            consider_overall(layout, label)

        print(f"\n=== {label_prefix}: ROUND 1 / {max_rounds} ===")
        base_layout = place_parts_with_existing(parts, plate_poly, _cache=solve_cache)
        consider(base_layout, f"{label_prefix}-baseline")
        if is_full(base_layout):
            return base_layout, True

        print(f"\n=== {label_prefix}: STRIP SOLVE AFTER DENSE GROUP ===")
        strip_layout = solve_remaining_strip_after_dense_group(parts, plate_poly, base_layout)
        consider(strip_layout, f"{label_prefix}-dense-group-strip")
        if is_full(strip_layout):
            return strip_layout, True

        print(f"\n=== {label_prefix}: RIGHT CIRCLE STRIP COMPACTION ===")
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

                print(f"Trying {label_prefix} right strip width={w:.2f}, strategy={strategy.get('name', 'custom')}")
                layout = solve_with_right_circle_strip(parts, plate_poly, w, strategy=strategy)
                consider(layout, f"{label_prefix}-right-strip-{w:.0f}-{strategy.get('name', 'custom')}")
                if is_full(layout):
                    return layout, True

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
            print(f"\n=== {label_prefix}: ROUND {round_idx} / {max_rounds} ===")
            print(f"Strategy: {strategy.get('name', 'custom')}")

            layout = place_parts_with_existing(parts, plate_poly, strategy=strategy, _cache=solve_cache)
            consider(layout, f"{label_prefix}-retry-{strategy.get('name', 'custom')}")
            if is_full(layout):
                return layout, True

        return best_layout, False

    layout, full = run_full_solver("parallelogram_first", "PARALLELOGRAM-PAIR")
    if full:
        return layout

    print("\n=== FULL RETRY WITH OLD TRAPEZOID PAIRING ===")
    old_layout, old_full = run_full_solver("old_only", "OLD-PAIR")
    if old_full:
        return old_layout

    best_count = placed_part_count(overall_best_layout)

    if best_count == expected:
        return overall_best_layout

    raise NestingFailed(
        f"Could not place all parts. Best result was {best_count}/{expected}.",
        best_layout=overall_best_layout,
        best_count=best_count,
        expected_count=expected,
    )

def place_parts_with_existing(parts, plate_poly, strategy=None, _cache=None):
    cache_key = None
    if _cache is not None:
        cache_key = (id(parts), freeze_bounds(plate_poly), freeze_strategy(strategy), TRAPEZOID_PAIR_STYLE)
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
# Exact small-instance solver (discrete grid model)
# -----------------------------
@dataclass(frozen=True)
class ExactGridOrientation:
    angle: int
    cells: tuple
    width: int
    height: int
    poly: object


@dataclass(frozen=True)
class ExactGridPieceType:
    name: str
    base_id: str
    shape: str
    count: int
    orientations: tuple


class ExactGridNestingFailed(Exception):
    pass


def _grid_round(value, step):
    return round(value / step) * step


def _is_multiple_of_step(value, step, tol=EXACT_GRID_TOL):
    return abs(value - _grid_round(value, step)) <= tol


def _poly_aligned_to_step(poly, step, tol=EXACT_GRID_TOL):
    if poly is None or poly.is_empty:
        return False
    for x, y in list(poly.exterior.coords):
        if not _is_multiple_of_step(x, step, tol) or not _is_multiple_of_step(y, step, tol):
            return False
    for ring in poly.interiors:
        for x, y in list(ring.coords):
            if not _is_multiple_of_step(x, step, tol) or not _is_multiple_of_step(y, step, tol):
                return False
    return True


def _axis_aligned_rectangle(poly, tol=EXACT_GRID_TOL):
    if poly is None or poly.is_empty:
        return False
    minx, miny, maxx, maxy = poly.bounds
    rect = box(minx, miny, maxx, maxy)
    return rect.symmetric_difference(poly).area <= tol


def _cell_box(ix, iy, step):
    x0 = ix * step
    y0 = iy * step
    return box(x0, y0, x0 + step, y0 + step)


def _polygon_to_exact_cells(poly, step, tol=EXACT_GRID_TOL):
    if poly is None or poly.is_empty:
        return None

    minx, miny, maxx, maxy = poly.bounds
    if not (_is_multiple_of_step(minx, step, tol) and _is_multiple_of_step(miny, step, tol) and
            _is_multiple_of_step(maxx, step, tol) and _is_multiple_of_step(maxy, step, tol)):
        return None

    ix0 = int(round(minx / step))
    iy0 = int(round(miny / step))
    ix1 = int(round(maxx / step))
    iy1 = int(round(maxy / step))

    cells = []
    cell_polys = []
    for ix in range(ix0, ix1):
        for iy in range(iy0, iy1):
            sq = _cell_box(ix, iy, step)
            if poly.covers(sq):
                cells.append((ix - ix0, iy - iy0))
                cell_polys.append(sq)

    if not cells:
        return None

    union = unary_union(cell_polys).buffer(0)
    if union.is_empty:
        return None

    if union.symmetric_difference(translate(poly, xoff=-minx, yoff=-miny)).area > tol:
        return None

    return tuple(sorted(cells))


def _candidate_exact_steps(parts, plate_poly):
    if plate_poly is None or plate_poly.is_empty or not _axis_aligned_rectangle(plate_poly):
        return []

    candidates = []
    for step in EXACT_GRID_STEPS:
        minx, miny, maxx, maxy = plate_poly.bounds
        width = maxx - minx
        height = maxy - miny
        if not (_is_multiple_of_step(minx, step) and _is_multiple_of_step(miny, step) and
                _is_multiple_of_step(width, step) and _is_multiple_of_step(height, step)):
            continue
        ok = True
        for part in parts:
            poly = part_to_polygon(part)
            if not _poly_aligned_to_step(poly, step):
                ok = False
                break
        if ok:
            board_cells = int(round(width / step)) * int(round(height / step))
            if board_cells <= EXACT_GRID_MAX_BOARD_CELLS:
                candidates.append(step)
    return candidates


def _build_exact_piece_types(parts, step):
    piece_types = []
    total_piece_cells = 0
    total_piece_count = 0

    for part in parts:
        base_id = str(part.get('id', '?'))
        qty = int(part.get('quantity', 1) or 1)
        poly = normalize_poly(part_to_polygon(part))
        shape = detect_shape_type(part, poly)

        orientations = []
        seen = set()
        for angle in (0, 90, 180, 270):
            rotated = rotate(poly, angle, origin=tuple(poly.centroid.coords[0]), use_radians=False)
            rotated = normalize_poly(rotated.buffer(0))
            cells = _polygon_to_exact_cells(rotated, step)
            if not cells or cells in seen:
                continue
            seen.add(cells)
            width = max(x for x, _ in cells) + 1
            height = max(y for _, y in cells) + 1
            orientations.append(ExactGridOrientation(angle=angle, cells=cells, width=width, height=height, poly=rotated))

        if not orientations:
            raise ExactGridNestingFailed(f'Part {base_id} is not representable exactly on the discrete grid for step={step}.')

        cell_area = len(orientations[0].cells)
        total_piece_cells += cell_area * qty
        total_piece_count += qty

        piece_types.append(
            ExactGridPieceType(
                name=base_id,
                base_id=base_id,
                shape=shape,
                count=qty,
                orientations=tuple(orientations),
            )
        )

    if total_piece_cells > EXACT_GRID_MAX_TOTAL_PIECE_CELLS:
        raise ExactGridNestingFailed(
            f'Exact discrete search skipped because total piece cells={total_piece_cells} exceeds limit {EXACT_GRID_MAX_TOTAL_PIECE_CELLS}.'
        )

    if total_piece_count > EXACT_GRID_MAX_PIECES:
        raise ExactGridNestingFailed(
            f'Exact discrete search skipped because piece count={total_piece_count} exceeds limit {EXACT_GRID_MAX_PIECES}.'
        )

    return piece_types, total_piece_cells, total_piece_count


def _bit_xy(board_w, x, y):
    return 1 << (y * board_w + x)


def _first_empty_cell(board_w, board_h, occupied):
    total = board_w * board_h
    for idx in range(total):
        if ((occupied >> idx) & 1) == 0:
            return idx % board_w, idx // board_w
    return None


def _remaining_piece_areas(piece_types, counts):
    vals = []
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        area = len(piece_types[i].orientations[0].cells)
        vals.extend([area] * c)
    return vals


def _region_prune(board_w, board_h, occupied, piece_types, counts):
    remaining = _remaining_piece_areas(piece_types, counts)
    if not remaining:
        return False

    min_area = min(remaining)
    possible_sums = {0}
    for a in remaining:
        possible_sums |= {s + a for s in list(possible_sums)}

    visited = set()
    for y in range(board_h):
        for x in range(board_w):
            idx = y * board_w + x
            if ((occupied >> idx) & 1) != 0 or idx in visited:
                continue

            q = deque([(x, y)])
            visited.add(idx)
            region_size = 0

            while q:
                cx, cy = q.popleft()
                region_size += 1
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < board_w and 0 <= ny < board_h:
                        nidx = ny * board_w + nx
                        if ((occupied >> nidx) & 1) == 0 and nidx not in visited:
                            visited.add(nidx)
                            q.append((nx, ny))

            if region_size < min_area:
                return True
            if region_size not in possible_sums:
                return True

    return False


def _anchored_candidates(board_w, board_h, occupied, piece_type, remaining_count, fx, fy):
    if remaining_count <= 0:
        return []

    unique = {}
    for orient in piece_type.orientations:
        for anchor_x, anchor_y in orient.cells:
            ox = fx - anchor_x
            oy = fy - anchor_y
            if ox < 0 or oy < 0:
                continue
            if ox + orient.width > board_w or oy + orient.height > board_h:
                continue

            mask = 0
            valid = True
            for dx, dy in orient.cells:
                px = ox + dx
                py = oy + dy
                bit = _bit_xy(board_w, px, py)
                if occupied & bit:
                    valid = False
                    break
                mask |= bit

            if valid and mask not in unique:
                unique[mask] = (orient, ox, oy)

    return [(mask, orient, ox, oy) for mask, (orient, ox, oy) in unique.items()]


def exact_nest_grid_parts(parts, plate_poly, step):
    piece_types, _, _ = _build_exact_piece_types(parts, step)

    minx, miny, maxx, maxy = plate_poly.bounds
    board_w = int(round((maxx - minx) / step))
    board_h = int(round((maxy - miny) / step))

    total_piece_cells = sum(len(pt.orientations[0].cells) * pt.count for pt in piece_types)
    if total_piece_cells > board_w * board_h:
        raise ExactGridNestingFailed('Total part area exceeds available plate area on the discrete grid.')

    counts0 = tuple(pt.count for pt in piece_types)
    fail_cache = set()
    solution = []

    def search(occupied, counts, placements):
        if sum(counts) == 0:
            solution[:] = placements[:]
            return True

        key = (occupied, counts)
        if key in fail_cache:
            return False

        if _region_prune(board_w, board_h, occupied, piece_types, counts):
            fail_cache.add(key)
            return False

        first_empty = _first_empty_cell(board_w, board_h, occupied)
        if first_empty is None:
            solution[:] = placements[:]
            return True

        fx, fy = first_empty
        options = []
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            cands = _anchored_candidates(board_w, board_h, occupied, piece_types[i], count, fx, fy)
            if cands:
                options.append((len(cands), -len(piece_types[i].orientations[0].cells), i, cands))

        if not options:
            fail_cache.add(key)
            return False

        options.sort()
        _, _, piece_idx, candidates = options[0]
        next_counts = list(counts)
        next_counts[piece_idx] -= 1
        next_counts = tuple(next_counts)

        candidates.sort(key=lambda item: -len(item[1].cells))

        for mask, orient, ox, oy in candidates:
            x = minx + ox * step
            y = miny + oy * step
            placed_poly = translate(orient.poly, xoff=x, yoff=y).buffer(0)
            placements.append({
                'id': piece_types[piece_idx].base_id,
                'base_id': piece_types[piece_idx].base_id,
                'unit_count': 1,
                'poly': placed_poly,
                'x': x,
                'y': y,
                'angle': orient.angle,
                'shape': piece_types[piece_idx].shape,
                'solver': f'exact_grid_{step:g}',
            })
            if search(occupied | mask, next_counts, placements):
                return True
            placements.pop()

        fail_cache.add(key)
        return False

    if not search(0, counts0, []):
        raise ExactGridNestingFailed('No exact discrete arrangement exists for this plate.')

    return list(solution)


def try_exact_small_instance(parts, plate_poly):
    steps = _candidate_exact_steps(parts, plate_poly)
    last_error = None
    for step in steps:
        try:
            return exact_nest_grid_parts(parts, plate_poly, step), step
        except ExactGridNestingFailed as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise ExactGridNestingFailed('Exact discrete search is unavailable for this geometry (needs a rectangular plate and grid-exact parts).')


# -----------------------------
# Plot arrangement / UI helpers
# -----------------------------
def _shared_edge_segments(placed_parts):
    segments = []
    for i, p1 in enumerate(placed_parts):
        b1 = p1["poly"].boundary
        for j in range(i + 1, len(placed_parts)):
            p2 = placed_parts[j]
            inter = b1.intersection(p2["poly"].boundary)
            if inter.is_empty:
                continue
            geoms = getattr(inter, "geoms", [inter])
            for g in geoms:
                if g.geom_type == "LineString" and g.length > 1e-6:
                    segments.append(g)
                elif g.geom_type == "MultiLineString":
                    segments.extend([line for line in g.geoms if line.length > 1e-6])
    return segments


def create_layout_figure(placed_parts, plate_poly, title="排样结果", combine_mode=0, envelope_poly=None, leftover_poly=None):
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=120)
    fig.patch.set_facecolor("#f2f2f2")
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    xs, ys = plate_poly.exterior.xy
    ax.plot(xs, ys, color="#ff4dff", linewidth=1.0)

    if leftover_poly is not None and not getattr(leftover_poly, "is_empty", True):
        geoms = leftover_poly.geoms if isinstance(leftover_poly, MultiPolygon) else [leftover_poly]
        for g in geoms:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color="#18b9ff", linewidth=1.0)

    if envelope_poly is not None and not getattr(envelope_poly, "is_empty", True):
        geoms = envelope_poly.geoms if isinstance(envelope_poly, MultiPolygon) else [envelope_poly]
        for g in geoms:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color="#ff4d4d", linewidth=1.1)

    for p in placed_parts:
        poly = p["poly"]
        polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
        for g in polys:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, facecolor="#0aa51d", edgecolor="white", linewidth=0.55)
            for hole in g.interiors:
                hx, hy = hole.xy
                ax.plot(hx, hy, color="white", linewidth=0.5)
        for g in p.get("display_polys", []) or []:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color="white", linewidth=0.5)
        c = poly.centroid
        ax.plot(c.x, c.y, marker=".", markersize=2.0)

    if int(combine_mode or 0) == 1:
        for seg in _shared_edge_segments(placed_parts):
            xs, ys = seg.xy
            ax.plot(xs, ys, color="yellow", linewidth=0.85)

    minx, miny, maxx, maxy = plate_poly.bounds
    pad_x = max((maxx - minx) * 0.04, 1.0)
    pad_y = max((maxy - miny) * 0.04, 1.0)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, loc="left", pad=8, color="white")
    fig.tight_layout(pad=0.5)
    return fig, ax


def plot_arrangement(placed_parts, plate_poly):
    fig, _ = create_layout_figure(placed_parts, plate_poly)
    plt.show()
    return fig


# -----------------------------
# Summary / stats helpers
# -----------------------------
def summarize_layout(placed_parts):
    summary = {}
    for p in placed_parts:
        key = get_part_key(p)
        count = int(p.get("unit_count", 1))
        summary[key] = summary.get(key, 0) + count
    return summary


def generate_table(placed_parts):
    summary = summarize_layout(placed_parts)
    print(f"{'ID':<12} {'Placed Count':<12}")
    for pid, count in summary.items():
        print(f"{pid:<12} {count:<12}")


def compute_layout_statistics(parts, placed_parts, plate_poly):
    expected = expected_part_count(parts)
    placed = placed_part_count(placed_parts)
    plate_area = plate_poly.area if hasattr(plate_poly, 'area') else 0.0
    used_area = sum(p["poly"].area for p in placed_parts)
    utilization = (used_area / plate_area * 100.0) if plate_area > 1e-9 else 0.0
    nesting_kinds = len({get_part_key(p) for p in placed_parts})
    bbox_area = overall_bbox_area(placed_parts)
    fill_ratio = layout_fill_ratio(placed_parts) * 100.0
    waste_area = layout_waste_area(placed_parts)
    result_plate = {"id": "1", "parts": placed_parts, "quantity": 1}
    env_poly = get_plate_envelope_polygon(result_plate, min_len=100)
    left_poly = get_plate_left_polygon(result_plate, plate_poly, min_len=100)
    return {
        "placed": placed,
        "expected": expected,
        "utilization": utilization,
        "part_kinds": nesting_kinds,
        "layout_count": 1,
        "bbox_area": bbox_area,
        "fill_ratio": fill_ratio,
        "waste_area": waste_area,
        "envelope_area": 0.0 if env_poly is None else float(env_poly.area),
        "leftover_area": 0.0 if left_poly is None else float(left_poly.area),
    }



def _filter_short_edges(coords, min_len):
    if not coords:
        return coords
    cleaned = [coords[0]]
    for pt in coords[1:]:
        prev = cleaned[-1]
        if math.hypot(pt[0] - prev[0], pt[1] - prev[1]) >= min_len:
            cleaned.append(pt)
    if len(cleaned) >= 3 and math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) < min_len:
        cleaned[-1] = cleaned[0]
    return cleaned


def polygon_to_siop_contour(poly, min_len=100):
    if poly is None or poly.is_empty:
        return {"points": [], "bclose": True}
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    coords = list(poly.exterior.coords)
    coords = _filter_short_edges(coords, float(min_len))
    points = [{"x": float(x), "y": float(y), "radius": 0.0, "angle1": 0.0, "angle2": 0.0} for x, y in coords[:-1]]
    return {"points": points, "bclose": True}


def get_plate_envelope_polygon(result_plate, min_len=100):
    polys = []
    for item in result_plate.get("parts", []):
        poly = item.get("poly")
        if poly is not None and not poly.is_empty:
            polys.append(poly)
    if not polys:
        return None
    env = unary_union(polys).convex_hull
    if not env.is_valid:
        env = env.buffer(0)
    return env


def get_plate_left_polygon(result_plate, plate_poly, min_len=100):
    env = get_plate_envelope_polygon(result_plate, min_len=min_len)
    if env is None:
        return plate_poly
    left = plate_poly.difference(env)
    if hasattr(left, "is_empty") and left.is_empty:
        return None
    if not left.is_valid:
        left = left.buffer(0)
    return left


def FOP_GETPLATE_ENVELOP(result_plate, minLen=100):
    env = get_plate_envelope_polygon(result_plate, min_len=minLen)
    return polygon_to_siop_contour(env, min_len=minLen)


def FOP_GETPLATE_LEFT(result_plate, plate_poly, minLen=100):
    left = get_plate_left_polygon(result_plate, plate_poly, min_len=minLen)
    return polygon_to_siop_contour(left, min_len=minLen)


def parse_autonest_input(data):
    if isinstance(data, str):
        data = json.loads(data)
    params = data if isinstance(data, dict) else {}
    return {
        "LeftDist": float(params.get("LeftDist", 0.0) or 0.0),
        "RightDist": float(params.get("RightDist", 0.0) or 0.0),
        "TopDist": float(params.get("TopDist", 0.0) or 0.0),
        "BottomDist": float(params.get("BottomDist", 0.0) or 0.0),
        "PartGap": float(params.get("PartGap", 0.0) or 0.0),
        "Compensation": float(params.get("Compensation", 0.0) or 0.0),
        "CutNumbers": int(params.get("CutNumbers", 1) or 1),
        "Nozzels": int(params.get("Nozzels", 1) or 1),
        "CarryArc": bool(params.get("CarryArc", False)),
        "CombineMode": int(params.get("CombineMode", 0) or 0),
    }


def apply_plate_margins(plate_poly, params):
    if plate_poly is None:
        return None
    left = float(params.get("LeftDist", 0.0) or 0.0)
    right = float(params.get("RightDist", 0.0) or 0.0)
    top = float(params.get("TopDist", 0.0) or 0.0)
    bottom = float(params.get("BottomDist", 0.0) or 0.0)
    minx, miny, maxx, maxy = plate_poly.bounds
    inner = box(minx + left, miny + bottom, maxx - right, maxy - top)
    clipped = plate_poly.intersection(inner)
    if clipped.is_empty:
        return inner
    return clipped


def scale_part_for_compensation(part, compensation):
    if abs(compensation) < 1e-9:
        return part
    scaled = dict(part)
    new_contours = []
    for contour in (part.get("contours") or []):
        new_contour = dict(contour)
        pts = []
        for pt in contour.get("points") or []:
            new_pt = dict(pt)
            if "x" in new_pt:
                new_pt["x"] = float(new_pt["x"]) * (1.0 + compensation)
            if "y" in new_pt:
                new_pt["y"] = float(new_pt["y"]) * (1.0 + compensation)
            if "radius" in new_pt and new_pt["radius"]:
                new_pt["radius"] = float(new_pt["radius"]) * (1.0 + compensation)
            pts.append(new_pt)
        new_contour["points"] = pts
        new_contours.append(new_contour)
    scaled["contours"] = new_contours
    return scaled


def apply_nesting_options(parts, plate_poly, params):
    part_gap = float(params.get("PartGap", 0.0) or 0.0)
    compensation = float(params.get("Compensation", 0.0) or 0.0)
    adjusted_parts = [scale_part_for_compensation(p, compensation) for p in parts]
    if part_gap > 1e-9:
        adjusted_parts = [{**p, "_extra_gap": part_gap} for p in adjusted_parts]
    adjusted_plate = apply_plate_margins(plate_poly, params)
    return adjusted_parts, adjusted_plate

def format_plan_name(candidate, index):
    stats = candidate["stats"]
    return (
        f"方案{index + 1} | {candidate['name']} | "
        f"{stats['placed']}/{stats['expected']} | 利用率 {stats['utilization']:.2f}%"
    )


def generate_candidate_layouts(parts, plate_poly, live_callback=None):
    solve_cache = {}
    candidates = []
    seen = set()

    def add_candidate(name, builder):
        payload = {"name": name, "parts": parts, "plate_poly": plate_poly}
        previous_observer = ACTIVE_PLACEMENT_OBSERVER
        if live_callback is not None:
            live_callback("start", dict(payload))
            set_placement_observer(lambda placed_parts, placed_part=None: live_callback(
                "placement",
                dict(payload, layout=list(placed_parts), placed_part=placed_part),
            ))
        layout = []
        try:
            layout = builder()
        except NestingFailed as exc:
            layout = exc.best_layout or []
        except Exception as exc:
            print(f"Candidate '{name}' failed: {exc}")
            layout = []
        finally:
            set_placement_observer(previous_observer)
            if live_callback is not None:
                live_callback("finish", dict(payload, layout=list(layout)))

        if not layout:
            return

        fingerprint = tuple(sorted(
            (p["id"], round(p["x"], 4), round(p["y"], 4), int(round(p.get("angle", 0))), int(p.get("unit_count", 1)))
            for p in layout
        ))
        if fingerprint in seen:
            return
        seen.add(fingerprint)

        stats = compute_layout_statistics(parts, layout, plate_poly)
        candidates.append({
            "name": name,
            "layout": layout,
            "stats": stats,
            "summary": summarize_layout(layout),
        })

    strategies = [
        ("精确离散求解(小规模)", lambda: try_exact_small_instance(parts, plate_poly)[0]),
        ("默认优先级", lambda: place_parts_with_existing(parts, plate_poly, _cache=solve_cache)),
        ("反向顺序", lambda: place_parts_with_existing(parts, plate_poly, strategy={
            "name": "reverse_order",
            "side_bar_order": "reverse",
            "trapezoid_order": "reverse",
            "filler_order": "reverse",
        }, _cache=solve_cache)),
        ("组内大件优先", lambda: place_parts_with_existing(parts, plate_poly, strategy={
            "name": "largest_first_inside_groups",
            "side_bar_order": "largest_first",
            "trapezoid_order": "largest_first",
            "filler_order": "largest_first",
        }, _cache=solve_cache)),
        ("旋转优先", lambda: place_parts_with_existing(parts, plate_poly, strategy={
            "name": "rotate_first",
            "anchor_angle_order": [90, 0],
            "fill_angle_order": [90, 0],
            "top_angles": [90, 0],
            "side_angles": [90, 0],
            "filler_angles": [90, 0],
        }, _cache=solve_cache)),
        ("完整求解", lambda: nest_parts_with_full_fit(parts, plate_poly)),
    ]

    for name, builder in strategies:
        add_candidate(name, builder)

    candidates.sort(key=lambda c: (
        c["stats"]["placed"],
        c["stats"]["utilization"],
        c["stats"]["fill_ratio"],
        -c["stats"]["waste_area"],
    ), reverse=True)

    for idx, cand in enumerate(candidates):
        cand["display_name"] = format_plan_name(cand, idx)

    return candidates


def get_strategy_builders(parts, plate_poly, solve_cache=None):
    solve_cache = solve_cache or {}
    return [
        ("默认优先级", lambda: place_parts_with_existing(parts, plate_poly, _cache=solve_cache)),
        ("组内大件优先", lambda: place_parts_with_existing(parts, plate_poly, strategy={
            "name": "largest_first_inside_groups",
            "side_bar_order": "largest_first",
            "trapezoid_order": "largest_first",
            "filler_order": "largest_first",
        }, _cache=solve_cache)),
        ("旋转优先", lambda: place_parts_with_existing(parts, plate_poly, strategy={
            "name": "rotate_first",
            "anchor_angle_order": [90, 0],
            "fill_angle_order": [90, 0],
            "top_angles": [90, 0],
            "side_angles": [90, 0],
            "filler_angles": [90, 0],
        }, _cache=solve_cache)),
    # return [
    #     ("精确离散求解(小规模)", lambda: try_exact_small_instance(parts, plate_poly)[0]),
    #     ("默认优先级", lambda: place_parts_with_existing(parts, plate_poly, _cache=solve_cache)),
    #     ("反向顺序", lambda: place_parts_with_existing(parts, plate_poly, strategy={
    #         "name": "reverse_order",
    #         "side_bar_order": "reverse",
    #         "trapezoid_order": "reverse",
    #         "filler_order": "reverse",
    #     }, _cache=solve_cache)),
    #     ("组内大件优先", lambda: place_parts_with_existing(parts, plate_poly, strategy={
    #         "name": "largest_first_inside_groups",
    #         "side_bar_order": "largest_first",
    #         "trapezoid_order": "largest_first",
    #         "filler_order": "largest_first",
    #     }, _cache=solve_cache)),
    #     ("旋转优先", lambda: place_parts_with_existing(parts, plate_poly, strategy={
    #         "name": "rotate_first",
    #         "anchor_angle_order": [90, 0],
    #         "fill_angle_order": [90, 0],
    #         "top_angles": [90, 0],
    #         "side_angles": [90, 0],
    #         "filler_angles": [90, 0],
    #     }, _cache=solve_cache)),
    #     ("完整求解", lambda: nest_parts_with_full_fit(parts, plate_poly)),
    ]


def layout_fingerprint(layout):
    return tuple(sorted(
        (
            str(p.get("base_id", p.get("id", ""))),
            round(float(p.get("x", 0.0)), 4),
            round(float(p.get("y", 0.0)), 4),
            int(round(float(p.get("angle", 0.0)))),
            int(p.get("unit_count", 1)),
        )
        for p in (layout or [])
    ))


def expand_plate_slots(plates_json, fallback_plate=None):
    slots = []
    if plates_json:
        counter = 1
        for plate in plates_json:
            qty = int((plate or {}).get("quantity", 1) or 1)
            qty = max(1, qty)
            for local_index in range(qty):
                slot = dict(plate or {})
                slot["_slot_no"] = counter
                slot["_slot_local_index"] = local_index + 1
                slots.append(slot)
                counter += 1
    elif fallback_plate is not None:
        slot = dict(fallback_plate)
        slot.setdefault("id", "1")
        slot["_slot_no"] = 1
        slot["_slot_local_index"] = 1
        slots.append(slot)
    return slots


def build_plan_entries_from_plate_results(plate_results):
    grouped = {}
    for plate_result in plate_results:
        key = (plate_result.get("plate_size_text", "-"), layout_fingerprint(plate_result.get("layout", [])))
        grouped.setdefault(key, []).append(plate_result)

    plan_entries = []
    for rows in grouped.values():
        rep = rows[0]
        rep_stats = rep["stats"]
        plan_entries.append({
            "utilization": float(rep_stats.get("utilization", 0.0)),
            "placed": int(rep_stats.get("placed", 0)),
            "count": len(rows),
            "size": rep.get("plate_size_text", "-"),
            "layout": rep.get("layout", []),
            "plate_poly": rep.get("plate_poly"),
            "plate_id": str(rep.get("plate_id", "1")),
            "representative_rows": rows,
        })

    plan_entries.sort(key=lambda e: (e["count"], e["utilization"], e["placed"]), reverse=True)
    for idx, entry in enumerate(plan_entries, start=1):
        entry["name"] = f"排版{idx}"
    return plan_entries


def compute_candidate_statistics(parts, plate_results, total_available_plates):
    expected = expected_part_count(parts)
    placed = sum(int(r["stats"].get("placed", 0)) for r in plate_results)
    used_area = sum(sum(p["poly"].area for p in r.get("layout", [])) for r in plate_results)
    total_plate_area = sum((r.get("plate_poly").area if r.get("plate_poly") is not None else 0.0) for r in plate_results)
    utilization = (used_area / total_plate_area * 100.0) if total_plate_area > 1e-9 else 0.0
    used_plate_count = len(plate_results)
    progress_pct = (used_plate_count / total_available_plates * 100.0) if total_available_plates > 0 else 0.0
    plan_entries = build_plan_entries_from_plate_results(plate_results)
    return {
        "placed": placed,
        "expected": expected,
        "utilization": utilization,
        "part_kinds": len(plan_entries),
        "layout_count": used_plate_count,
        "fill_ratio": progress_pct,
        "used_plate_count": used_plate_count,
        "total_plate_count": int(total_available_plates),
        "envelope_area": sum(float(r["stats"].get("envelope_area", 0.0)) for r in plate_results),
        "leftover_area": sum(float(r["stats"].get("leftover_area", 0.0)) for r in plate_results),
    }


def _run_strategy_builder(builder, strategy_name, adjusted_parts, adjusted_plate, live_callback=None):
    payload = {"name": strategy_name, "parts": adjusted_parts, "plate_poly": adjusted_plate}
    previous_observer = ACTIVE_PLACEMENT_OBSERVER
    if live_callback is not None:
        live_callback("start", dict(payload))
        set_placement_observer(lambda placed_parts, placed_part=None: live_callback(
            "placement",
            dict(payload, layout=list(placed_parts), placed_part=placed_part),
        ))
    layout = []
    try:
        layout = builder()
    except NestingFailed as exc:
        layout = exc.best_layout or []
    except Exception as exc:
        print(f"Overall candidate '{strategy_name}' failed: {exc}")
        layout = []
    finally:
        set_placement_observer(previous_observer)
        if live_callback is not None:
            live_callback("finish", dict(payload, layout=list(layout)))
    return layout


def generate_overall_candidates(parts, plate_slots, params, fallback_width=None, fallback_height=None, live_callback_factory=None):
    candidates = []
    seen = set()
    total_available_plates = len(plate_slots)
    strategy_names = [name for name, _ in get_strategy_builders(parts, box(0, 0, 1, 1), solve_cache={})]

    for strategy_name in strategy_names:
        remaining_parts = [dict(p) for p in parts]
        plate_results = []
        solve_cache = {}

        for slot_index, slot in enumerate(plate_slots, start=1):
            base_plate_poly = plate_to_polygon(slot) if slot else None
            if base_plate_poly is None and fallback_width is not None and fallback_height is not None:
                base_plate_poly = box(0, 0, float(fallback_width), float(fallback_height))
            if base_plate_poly is None:
                continue

            adjusted_parts, adjusted_plate = apply_nesting_options(remaining_parts, base_plate_poly, params)
            if adjusted_plate is None or adjusted_plate.is_empty:
                continue

            builders = dict(get_strategy_builders(adjusted_parts, adjusted_plate, solve_cache=solve_cache))
            builder = builders.get(strategy_name)
            if builder is None:
                continue

            live_callback = None
            if callable(live_callback_factory):
                try:
                    plate_label = f"板材{slot.get('id', slot.get('_slot_no', slot_index))}"
                except Exception:
                    plate_label = f"板材{slot_index}"
                base_live_callback = live_callback_factory(plate_label)

                def live_callback(event, payload, _base_live_callback=base_live_callback, _plate_label=plate_label,
                                  _strategy_name=strategy_name, _parts=parts, _plate_results=plate_results,
                                  _slot=slot, _slot_index=slot_index, _adjusted_parts=adjusted_parts,
                                  _adjusted_plate=adjusted_plate):
                    live_layout = list(payload.get("layout") or [])
                    combined_plate_results = list(_plate_results)
                    if event in ("placement", "finish") and live_layout:
                        plate_stats = compute_layout_statistics(_adjusted_parts, live_layout, _adjusted_plate)
                        minx, miny, maxx, maxy = _adjusted_plate.bounds
                        combined_plate_results.append({
                            "plate_id": str(_slot.get("id", _slot.get("_slot_no", _slot_index))),
                            "slot_no": int(_slot.get("_slot_no", _slot_index)),
                            "plate_poly": _adjusted_plate,
                            "layout": live_layout,
                            "stats": plate_stats,
                            "plate_size_text": f"{maxx - minx:.2f} * {maxy - miny:.2f}",
                        })
                    plan_entries = build_plan_entries_from_plate_results(combined_plate_results)
                    stats = compute_candidate_statistics(_parts, combined_plate_results, total_available_plates) if combined_plate_results else {
                        "placed": 0,
                        "expected": expected_part_count(_parts),
                        "utilization": 0.0,
                        "part_kinds": 0,
                        "layout_count": 0,
                        "fill_ratio": 0.0,
                    }
                    enriched = dict(payload)
                    enriched["name"] = _strategy_name
                    enriched["display_name"] = _strategy_name
                    enriched["status_label"] = f"{_plate_label}-{_strategy_name}"
                    enriched["plan_entries"] = plan_entries
                    enriched["stats"] = stats
                    _base_live_callback(event, enriched)

            layout = _run_strategy_builder(builder, strategy_name, adjusted_parts, adjusted_plate, live_callback=live_callback)
            if not layout or placed_part_count(layout) <= 0:
                continue

            plate_stats = compute_layout_statistics(adjusted_parts, layout, adjusted_plate)
            minx, miny, maxx, maxy = adjusted_plate.bounds
            plate_results.append({
                "plate_id": str(slot.get("id", slot.get("_slot_no", slot_index))),
                "slot_no": int(slot.get("_slot_no", slot_index)),
                "plate_poly": adjusted_plate,
                "layout": layout,
                "stats": plate_stats,
                "plate_size_text": f"{maxx - minx:.2f} * {maxy - miny:.2f}",
            })

            remaining_parts = build_remaining_parts_json(remaining_parts, layout)
            if not remaining_parts:
                break

        if not plate_results:
            continue

        overall_fp = tuple((r["plate_size_text"], layout_fingerprint(r.get("layout", []))) for r in plate_results)
        if overall_fp in seen:
            continue
        seen.add(overall_fp)

        candidates.append({
            "name": strategy_name,
            "plate_results": plate_results,
            "plan_entries": build_plan_entries_from_plate_results(plate_results),
            "stats": compute_candidate_statistics(parts, plate_results, total_available_plates),
        })

    candidates.sort(key=lambda c: (
        c["stats"]["placed"],
        c["stats"]["utilization"],
        c["stats"]["fill_ratio"],
        -c["stats"]["part_kinds"],
    ), reverse=True)
    return candidates


def build_results_json_from_candidate(candidate):
    if not candidate:
        return {"plates": []}
    result_plates = []
    for entry in candidate.get("plan_entries", []):
        result_plates.append({
            "id": str(entry.get("plate_id", "1")),
            "parts": [
                {
                    "id": str(p.get("base_id", p.get("id", ""))),
                    "mirrored": False,
                    "angle": math.radians(float(p.get("angle", 0.0))),
                    "move": {
                        "x": float(p.get("x", 0.0)),
                        "y": float(p.get("y", 0.0)),
                        "radius": 0.0,
                        "angle1": 0.0,
                        "angle2": 0.0,
                    },
                    "arc_strikes": [],
                }
                for p in entry.get("layout", [])
            ],
            "quantity": int(entry.get("count", 1)),
        })
    return {"plates": result_plates}


# -----------------------------
# GUI
# -----------------------------
class NestingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nest AutoNesting UI")
        self.root.geometry("1500x920")
        self.root.configure(bg="#efefef")

        self.parts = None
        self.plate_poly = None
        self.plates_json = []
        self.plate_json = None
        self.plate_width = None
        self.plate_height = None
        self.candidates = []
        self.selected_candidate = None
        self.current_candidate_index = None
        self.current_plan_entry = None
        self.current_plan_entries = []
        self.total_available_plates = 0
        self.confirmed_result_json = None
        self.autonest_params = parse_autonest_input({})
        self.current_figure = None
        self.current_canvas = None
        self.current_toolbar = None
        self._worker_thread = None
        self._worker_queue = queue.Queue()
        self._current_job_id = 0
        self._is_running = False

        self._configure_style()
        self._build_ui()

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background="#efefef")
        style.configure("Card.TFrame", background="#efefef")
        style.configure("Summary.TLabel", background="#efefef", font=("Microsoft YaHei", 11))
        style.configure("Small.TLabel", background="#efefef", font=("Microsoft YaHei", 10))
        style.configure("Blue.TLabel", background="#efefef", foreground="#2f74c8", font=("Microsoft YaHei", 10))
        style.configure("Section.TLabelframe", background="#efefef")
        style.configure("Section.TLabelframe.Label", background="#efefef", font=("Microsoft YaHei", 10))
        style.configure("Green.Horizontal.TProgressbar", troughcolor="#f5f5f5", background="#18bf2f", lightcolor="#49d95c", darkcolor="#18bf2f", bordercolor="#bfbfbf")
        style.configure("Treeview", rowheight=28, font=("Microsoft YaHei", 10))
        style.configure("Treeview.Heading", font=("Microsoft YaHei", 10))

    def _build_ui(self):
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)

        top = ttk.Frame(self.root, padding=(14, 10, 14, 8), style="App.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        ctrl = ttk.Frame(top, style="App.TFrame")
        ctrl.grid(row=0, column=0, sticky="w", pady=(0, 8))
        for col in range(3):
            ctrl.columnconfigure(col, weight=0)

        self.plate_size_var = tk.StringVar(value="Not loaded")
        self.left_dist_var = tk.DoubleVar(value=0.0)
        self.right_dist_var = tk.DoubleVar(value=0.0)
        self.top_dist_var = tk.DoubleVar(value=0.0)
        self.bottom_dist_var = tk.DoubleVar(value=0.0)
        self.part_gap_var = tk.DoubleVar(value=0.0)
        self.compensation_var = tk.DoubleVar(value=0.0)
        self.cut_numbers_var = tk.IntVar(value=1)
        self.nozzels_var = tk.IntVar(value=1)
        self.carry_arc_var = tk.BooleanVar(value=False)
        self.combine_mode_var = tk.IntVar(value=0)

        self.load_button = ttk.Button(ctrl, text="Load JSON", command=self.load_json_file)
        self.load_button.grid(row=0, column=0, padx=(0, 8))
        self.run_button = ttk.Button(ctrl, text="Run Nesting", command=self.run_nesting)
        self.run_button.grid(row=0, column=1, padx=8)
        self.confirm_button = ttk.Button(ctrl, text="Confirm Plan", command=self.confirm_plan)
        self.confirm_button.grid(row=0, column=2, padx=(8, 0))

        self.header_var = tk.StringVar(value="第0次优化，已排入0个，平均利用率0.00%")
        ttk.Label(top, textvariable=self.header_var, style="Summary.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 6))

        progress_wrap = tk.Frame(top, bg="#f7f7f7", highlightbackground="#bdbdbd", highlightthickness=1)
        progress_wrap.grid(row=2, column=0, sticky="ew")
        progress_wrap.columnconfigure(0, weight=1)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(progress_wrap, orient="horizontal", mode="determinate", variable=self.progress_var, maximum=100, style="Green.Horizontal.TProgressbar")
        self.progress.grid(row=0, column=0, sticky="ew", padx=2, pady=2, ipady=8)
        self.progress_label = tk.Label(progress_wrap, text="0%", bg="#f7f7f7", fg="black", font=("Microsoft YaHei", 11))
        self.progress_label.place(relx=0.5, rely=0.5, anchor="center")

        summary_line = ttk.Frame(top, style="App.TFrame")
        summary_line.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.result_title_var = tk.StringVar(value="◎ 排样结果1：")
        self.summary_util_var = tk.StringVar(value="0.00%")
        self.summary_parts_var = tk.StringVar(value="0/0")
        self.summary_kinds_var = tk.StringVar(value="0")
        self.summary_layout_count_var = tk.StringVar(value="0")
        ttk.Label(summary_line, textvariable=self.result_title_var, style="Small.TLabel").pack(side=tk.LEFT)
        ttk.Label(summary_line, text="总利用率", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(summary_line, textvariable=self.summary_util_var, style="Blue.TLabel").pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(summary_line, text="零件总数", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(summary_line, textvariable=self.summary_parts_var, style="Blue.TLabel").pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(summary_line, text="排版种类", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(summary_line, textvariable=self.summary_kinds_var, style="Blue.TLabel").pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(summary_line, text="排版总数", style="Small.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(summary_line, textvariable=self.summary_layout_count_var, style="Blue.TLabel").pack(side=tk.LEFT)

        main = ttk.Frame(self.root, padding=(12, 6, 12, 8), style="App.TFrame")
        main.grid(row=1, column=0, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)

        preview_panel = tk.Frame(main, bg="white", highlightbackground="#c6c6c6", highlightthickness=1)
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        preview_panel.rowconfigure(0, weight=1)
        preview_panel.columnconfigure(0, weight=1)
        self.canvas_host = ttk.Frame(preview_panel)
        self.canvas_host.grid(row=0, column=0, sticky="nsew")
        self.canvas_host.rowconfigure(0, weight=1)
        self.canvas_host.columnconfigure(0, weight=1)
        self.toolbar_frame = ttk.Frame(preview_panel)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")

        plan_panel = tk.Frame(main, bg="white", highlightbackground="#c6c6c6", highlightthickness=1)
        plan_panel.grid(row=0, column=1, sticky="nsew")
        plan_panel.rowconfigure(0, weight=1)
        plan_panel.columnconfigure(0, weight=1)
        self.plan_tree = ttk.Treeview(
            plan_panel,
            columns=("name", "util", "parts", "count", "size"),
            show="headings",
            height=8,
        )
        headings = [("name", "排样名称", 90), ("util", "利用率(%)", 90), ("parts", "零件数", 70), ("count", "排版数", 70), ("size", "板材尺寸", 165)]
        for key, title, width in headings:
            self.plan_tree.heading(key, text=title)
            self.plan_tree.column(key, width=width, anchor="center")
        self.plan_tree.column("name", anchor="w")
        self.plan_tree.grid(row=0, column=0, sticky="nsew")
        plan_scroll = ttk.Scrollbar(plan_panel, orient="vertical", command=self.plan_tree.yview)
        plan_scroll.grid(row=0, column=1, sticky="ns")
        self.plan_tree.configure(yscrollcommand=plan_scroll.set)
        self.plan_tree.bind("<<TreeviewSelect>>", self.on_plan_select)


        self.detail_tree = None

        bottom = ttk.Frame(self.root, padding=(12, 0, 12, 12), style="App.TFrame")
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, text="排样结果", style="Summary.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.result_tree = ttk.Treeview(
            bottom,
            columns=("result", "util", "parts", "kinds", "count"),
            show="headings",
            height=5,
        )
        bottom_headings = [("result", "排样结果", 180), ("util", "总利用率(%)", 180), ("parts", "零件总数", 180), ("kinds", "排版种类", 180), ("count", "排版总数", 180)]
        for key, title, width in bottom_headings:
            self.result_tree.heading(key, text=title)
            self.result_tree.column(key, width=width, anchor="center")
        self.result_tree.column("result", anchor="w")
        self.result_tree.grid(row=1, column=0, sticky="ew")
        self.result_tree.bind("<<TreeviewSelect>>", self.on_result_select)

        action_row = ttk.Frame(bottom, style="App.TFrame")
        action_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.auto_select_button = ttk.Button(action_row, text="返回自动选择", command=self.select_best_candidate)
        self.auto_select_button.pack(side=tk.LEFT)
        self.bottom_confirm_button = ttk.Button(action_row, text="确认", command=self.confirm_plan)
        self.bottom_confirm_button.pack(side=tk.RIGHT)
        self.cancel_button = ttk.Button(action_row, text="取消", command=self.root.quit)
        self.cancel_button.pack(side=tk.RIGHT, padx=(0, 8))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var, style="Small.TLabel").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self._pump_worker_queue()


    def _collect_params(self):
        self.autonest_params = {
            "LeftDist": float(self.left_dist_var.get() or 0.0),
            "RightDist": float(self.right_dist_var.get() or 0.0),
            "TopDist": float(self.top_dist_var.get() or 0.0),
            "BottomDist": float(self.bottom_dist_var.get() or 0.0),
            "PartGap": float(self.part_gap_var.get() or 0.0),
            "Compensation": float(self.compensation_var.get() or 0.0),
            "CutNumbers": int(self.cut_numbers_var.get() or 1),
            "Nozzels": int(self.nozzels_var.get() or 1),
            "CarryArc": bool(self.carry_arc_var.get()),
            "CombineMode": int(self.combine_mode_var.get() or 0),
        }
        return self.autonest_params

    def select_best_candidate(self):
        if not self.candidates or not self.result_tree.get_children():
            return
        first_id = self.result_tree.get_children()[0]
        self.result_tree.selection_set(first_id)
        self.result_tree.focus(first_id)
        self.show_candidate(0)

    def on_result_select(self, _event=None):
        selection = self.result_tree.selection()
        if not selection:
            return
        iid = selection[0]
        if iid == "live_result":
            return
        try:
            index = self.result_tree.index(iid)
        except Exception:
            return
        self.show_candidate(index)

    def _set_plan_entries(self, plan_entries, render_first=False):
        for row in self.plan_tree.get_children():
            self.plan_tree.delete(row)
        self.current_plan_entries = list(plan_entries or [])
        for index, entry in enumerate(self.current_plan_entries, start=1):
            iid = f"plan_{index}"
            self.plan_tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(
                    entry.get("name", f"排版{index}"),
                    f"{float(entry.get('utilization', 0.0)):.2f}%",
                    str(int(entry.get("placed", 0))),
                    str(int(entry.get("count", 1))),
                    entry.get("size", "-"),
                ),
            )
        if render_first and self.current_plan_entries:
            try:
                first_id = self.plan_tree.get_children()[0]
                self.plan_tree.selection_set(first_id)
                self.plan_tree.focus(first_id)
            except Exception:
                pass
            self._show_plan_entry(0)

    def _load_candidate_plan_entries(self, candidate):
        self._set_plan_entries(candidate.get("plan_entries", []), render_first=False)

    def _populate_detail_tree(self, candidate):
        if self.detail_tree is None:
            return
        for row in self.detail_tree.get_children():
            self.detail_tree.delete(row)
        for idx, part in enumerate(candidate["layout"], start=1):
            self.detail_tree.insert(
                "",
                tk.END,
                iid=f"detail_{idx}",
                values=(
                    str(part.get("base_id", part.get("id", ""))),
                    str(int(part.get("unit_count", 1))),
                    f"{float(part.get('angle', 0.0)):.1f}",
                    f"{float(part.get('x', 0.0)):.2f}",
                    f"{float(part.get('y', 0.0)):.2f}",
                ),
            )

    def _plate_size_text(self, plate_poly=None):
        plate_poly = self.plate_poly if plate_poly is None else plate_poly
        if plate_poly is None:
            return "-"
        minx, miny, maxx, maxy = plate_poly.bounds
        return f"{maxx - minx:.2f} * {maxy - miny:.2f}"

    def _compute_live_stats(self, payload):
        stats = dict(payload.get("stats") or {})
        stats.setdefault("placed", 0)
        stats.setdefault("expected", expected_part_count(self.parts or []))
        stats.setdefault("utilization", 0.0)
        stats.setdefault("part_kinds", len(payload.get("plan_entries") or []))
        stats.setdefault("layout_count", sum(int(entry.get("count", 1)) for entry in (payload.get("plan_entries") or [])))
        stats.setdefault("fill_ratio", 0.0)
        return stats

    def _ensure_live_result_row(self, label):
        values = (label, f"{self._bar_text(0.0)}  0.00%", "0/0", "0", "0")
        if "live_result" in self.result_tree.get_children():
            self.result_tree.item("live_result", values=values)
        else:
            self.result_tree.insert("", 0, iid="live_result", values=values)

    def _update_live_result_row(self, label, payload):
        stats = self._compute_live_stats(payload)
        util_bar = self._bar_text(stats["utilization"])
        self._ensure_live_result_row(label)
        self.result_tree.item(
            "live_result",
            values=(
                label,
                f"{util_bar}  {stats['utilization']:.2f}%",
                f"{stats['placed']}/{stats['expected']}",
                str(stats["part_kinds"]),
                str(stats["layout_count"]),
            ),
        )
        # Do not update the preview/detail region in real time.
        # Keep those regions stable until a final candidate is ready.
        self.header_var.set(f"{label}，已排入{stats['placed']}个，平均利用率{stats['utilization']:.2f}%")
        self.progress_var.set(stats["fill_ratio"])
        self.progress_label.configure(text=f"{stats['fill_ratio']:.0f}%")
        self.result_title_var.set(f"◎ {label}：")
        self.summary_util_var.set(f"{stats['utilization']:.2f}%")
        self.summary_parts_var.set(f"{stats['placed']}/{stats['expected']}")
        self.summary_kinds_var.set(str(stats["part_kinds"]))
        self.summary_layout_count_var.set(str(stats["layout_count"]))
        self.status_var.set(f"Running {payload.get('status_label', label)} ...")
        self.root.update_idletasks()

    def _make_live_callback(self, plate_label, job_id=None):
        active_job_id = self._current_job_id if job_id is None else job_id

        def callback(event, payload):
            self._queue_ui_message(active_job_id, "live_event", plate_label, event, dict(payload))

        return callback

    def _set_running_state(self, is_running):
        self._is_running = bool(is_running)
        state = "disabled" if self._is_running else "normal"
        for button in (
            getattr(self, "load_button", None),
            getattr(self, "run_button", None),
            getattr(self, "confirm_button", None),
            getattr(self, "auto_select_button", None),
            getattr(self, "bottom_confirm_button", None),
        ):
            if button is None:
                continue
            try:
                button.configure(state=state)
            except Exception:
                pass

    def _queue_ui_message(self, job_id, kind, *payload):
        try:
            self._worker_queue.put((job_id, kind, payload))
        except Exception:
            pass

    def _pump_worker_queue(self):
        try:
            while True:
                job_id, kind, payload = self._worker_queue.get_nowait()
                if job_id != self._current_job_id:
                    continue
                if kind == "live_event":
                    plate_label, event, data = payload
                    self._handle_live_event(plate_label, event, data)
                elif kind == "done":
                    candidates, multi_plate_mode = payload
                    self._apply_nesting_results(candidates, multi_plate_mode)
                elif kind == "error":
                    message_text, detail_text = payload
                    self._set_running_state(False)
                    self.status_var.set("Nesting failed")
                    messagebox.showerror("Error", message_text)
                    if detail_text:
                        print(detail_text)
        except queue.Empty:
            pass
        finally:
            try:
                self.root.after(40, self._pump_worker_queue)
            except tk.TclError:
                pass

    def _handle_live_event(self, plate_label, event, payload):
        label = payload.get("display_name") or payload.get("name") or "实时结果"
        if event == "start":
            self._ensure_live_result_row(label)
            self.status_var.set(f"Running {payload.get('status_label', label)} ...")
            # Do not populate the preview/detail region during live progress.
            self.root.update_idletasks()
        elif event in ("placement", "finish"):
            self._update_live_result_row(label, payload)

    def _apply_nesting_results(self, candidates, multi_plate_mode):
        self._set_running_state(False)
        self.candidates = list(candidates or [])
        self.selected_candidate = None
        self.current_candidate_index = None
        self.current_plan_entry = None
        self.current_plan_entries = []

        for row in self.plan_tree.get_children():
            self.plan_tree.delete(row)
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)

        for index, candidate in enumerate(self.candidates, start=1):
            stats = candidate["stats"]
            util_bar = self._bar_text(stats["utilization"])
            self.result_tree.insert(
                "",
                tk.END,
                iid=f"result_{index}",
                values=(
                    f"排样结果{index}",
                    f"{util_bar}  {stats['utilization']:.2f}%",
                    f"{stats['placed']}/{stats['expected']}",
                    str(stats["part_kinds"]),
                    str(stats["layout_count"]),
                ),
            )

        if not self.candidates:
            messagebox.showwarning("No Result", "No valid nesting result was generated.")
            self.status_var.set("No result")
            return

        first_id = self.result_tree.get_children()[0]
        self.result_tree.selection_set(first_id)
        self.result_tree.focus(first_id)
        self.show_candidate(0)
        self.status_var.set(f"Generated {len(self.candidates)} sorted candidate plan(s)")

    def _worker_run_nesting(self, job_id, params, plate_entries, multi_plate_mode):
        try:
            fallback_plate = self.plate_json or {"id": "1"}
            plate_slots = expand_plate_slots(self.plates_json, fallback_plate=fallback_plate)
            if not plate_slots:
                plate_slots = expand_plate_slots([], fallback_plate={"id": "1"})
            self.total_available_plates = len(plate_slots)

            candidates = generate_overall_candidates(
                self.parts,
                plate_slots,
                params,
                fallback_width=self.plate_width,
                fallback_height=self.plate_height,
                live_callback_factory=lambda plate_label: self._make_live_callback(plate_label, job_id),
            )

            self._queue_ui_message(job_id, "done", candidates, len(plate_slots) > 1)
        except Exception as exc:
            self._queue_ui_message(job_id, "error", f"Nesting failed: {exc}", traceback.format_exc())

    def load_json_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return

        try:
            data = load_json(file_path)
            self.parts = data["parts"]
            plates = data.get("plates") or []
            self.plates_json = list(plates)
            self.plate_json = plates[0] if plates else None

            plate_w, plate_h = extract_plate_size(data)
            self.plate_width = plate_w
            self.plate_height = plate_h
            if plate_w is not None and plate_h is not None:
                self.plate_size_var.set(f"{plate_w:g} x {plate_h:g}")
            else:
                self.plate_size_var.set("Not found in JSON")

            total_quantity = sum(int(p.get("quantity", 1)) for p in self.parts)
            size_text = f" | plate {plate_w:g} x {plate_h:g}" if plate_w is not None and plate_h is not None else " | plate size not found in JSON"
            plate_count = len(self.plates_json) if self.plates_json else (1 if self.plate_json else 0)
            self.status_var.set(f"Loaded {len(self.parts)} parts / {total_quantity} items / {plate_count} plate(s){size_text}")
            messagebox.showinfo(
                "Success",
                f"Loaded {len(self.parts)} part objects ({total_quantity} total items)"
                + (f"\nDetected plate size: {plate_w:g} x {plate_h:g}" if plate_w is not None and plate_h is not None else "\nPlate size was not found in JSON."),
            )
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load JSON: {exc}")

    def run_nesting(self):
        if self.parts is None:
            messagebox.showerror("Error", "Please load a JSON file first")
            return

        if self.plate_width is None and not self.plates_json:
            messagebox.showerror("Error", "Plate width and height must come from the JSON file.")
            return

        if self._is_running:
            self.status_var.set("Nesting is already running...")
            return

        params = self._collect_params()
        plate_entries = self.plates_json or ([self.plate_json] if self.plate_json else [])
        if not plate_entries:
            plate_entries = [{"id": "1"}]

        self.candidates = []
        self.selected_candidate = None
        self.current_candidate_index = None
        self.current_plan_entry = None
        self.current_plan_entries = []
        multi_plate_mode = len(plate_entries) > 1
        self._current_job_id += 1
        job_id = self._current_job_id

        for row in self.plan_tree.get_children():
            self.plan_tree.delete(row)
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)

        self._set_running_state(True)
        self.status_var.set("Calculating candidate nesting plans...")
        self.header_var.set("第0次优化，已排入0个，平均利用率0.00%")
        self.progress_var.set(0.0)
        self.progress_label.configure(text="0%")
        self.result_title_var.set("◎ 排样结果1：")
        self.summary_util_var.set("0.00%")
        self.summary_parts_var.set("0/0")
        self.summary_kinds_var.set("0")
        self.summary_layout_count_var.set("0")
        self.root.update_idletasks()

        self._worker_thread = threading.Thread(
            target=self._worker_run_nesting,
            args=(job_id, params, plate_entries, multi_plate_mode),
            daemon=True,
        )
        self._worker_thread.start()

    def _bar_text(self, value):
        filled = max(0, min(12, int(round(value / 100.0 * 12))))
        return "█" * filled + "░" * (12 - filled)

    def on_plan_select(self, _event=None):
        selection = self.plan_tree.selection()
        if not selection:
            return
        iid = selection[0]
        try:
            index = self.plan_tree.index(iid)
        except Exception:
            return
        self._show_plan_entry(index)

    def show_candidate(self, index):
        if index < 0 or index >= len(self.candidates):
            return

        candidate = self.candidates[index]
        self.selected_candidate = candidate
        self.current_candidate_index = index
        self._load_candidate_plan_entries(candidate)
        self._update_header(candidate, index)
        self._highlight_result_row(index)

        if self.current_plan_entries:
            first_id = self.plan_tree.get_children()[0]
            self.plan_tree.selection_set(first_id)
            self.plan_tree.focus(first_id)
            self._show_plan_entry(0)
        else:
            self.current_plan_entry = None

    def _show_plan_entry(self, index):
        if index < 0 or index >= len(self.current_plan_entries):
            return
        entry = self.current_plan_entries[index]
        self.current_plan_entry = entry
        self._render_candidate(entry)
        self._populate_detail_tree(entry)

    def _render_candidate(self, plan_entry):
        if self.current_canvas is not None:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
        if self.current_toolbar is not None:
            self.current_toolbar.destroy()
            self.current_toolbar = None
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None

        render_plate_poly = plan_entry.get("plate_poly")
        if render_plate_poly is not None:
            self.plate_poly = render_plate_poly
        result_plate = {"id": str(plan_entry.get("plate_id", "1")), "parts": plan_entry.get("layout", []), "quantity": int(plan_entry.get("count", 1))}
        env_poly = get_plate_envelope_polygon(result_plate, min_len=100)
        left_poly = get_plate_left_polygon(result_plate, render_plate_poly, min_len=100)
        fig, _ = create_layout_figure(
            plan_entry.get("layout", []),
            render_plate_poly,
            title="",
            combine_mode=self.autonest_params.get("CombineMode", 0),
            envelope_poly=env_poly,
            leftover_poly=left_poly,
        )
        self.current_figure = fig
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.canvas_host)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.current_toolbar = NavigationToolbar2Tk(self.current_canvas, self.toolbar_frame, pack_toolbar=False)
        self.current_toolbar.update()
        self.current_toolbar.pack(side=tk.LEFT)

    def _update_header(self, candidate, index):
        stats = candidate["stats"]
        self.header_var.set(f"第{index + 1}次优化，已排入{stats['placed']}个，平均利用率{stats['utilization']:.2f}%")
        self.progress_var.set(stats["fill_ratio"])
        self.progress_label.configure(text=f"{stats['fill_ratio']:.0f}%")
        self.result_title_var.set(f"◎ 排样结果{index + 1}：")
        self.summary_util_var.set(f"{stats['utilization']:.2f}%")
        self.summary_parts_var.set(f"{stats['placed']}/{stats['expected']}")
        self.summary_kinds_var.set(str(stats["part_kinds"]))
        self.summary_layout_count_var.set(str(stats["layout_count"]))

    def _highlight_result_row(self, index):
        for item in self.result_tree.get_children():
            self.result_tree.item(item, tags=())
        selected = f"result_{index + 1}"
        if selected in self.result_tree.get_children():
            self.result_tree.tag_configure("selected", background="#9ebfe6")
            self.result_tree.item(selected, tags=("selected",))

    def confirm_plan(self):
        if not self.selected_candidate:
            messagebox.showwarning("No Selection", "Please run nesting and select a plan first.")
            return

        stats = self.selected_candidate["stats"]
        self.confirmed_result_json = build_results_json_from_candidate(self.selected_candidate)
        messagebox.showinfo(
            "方案已确认",
            (
                "已选择当前套料方案。\n\n"
                f"方案: 排样结果{(self.current_candidate_index or 0) + 1}\n"
                f"排样结果: {stats['placed']}/{stats['expected']}\n"
                f"利用率: {stats['utilization']:.2f}%\n"
                f"排版种类: {stats['part_kinds']}\n"
                f"排版总数: {stats['layout_count']}\n"
                f"用板进度: {stats['used_plate_count']}/{stats['total_plate_count']}"
            )
        )
        self.status_var.set(f"Confirmed plan: 排样结果{(self.current_candidate_index or 0) + 1}")


def FOP_AUTONEST_PARAM(input_json, parent=None):
    params = parse_autonest_input(input_json)
    root = tk.Tk() if parent in (None, 0) else tk.Toplevel(parent)
    app = NestingApp(root)
    app.left_dist_var.set(params["LeftDist"])
    app.right_dist_var.set(params["RightDist"])
    app.top_dist_var.set(params["TopDist"])
    app.bottom_dist_var.set(params["BottomDist"])
    app.part_gap_var.set(params["PartGap"])
    app.compensation_var.set(params["Compensation"])
    app.cut_numbers_var.set(params["CutNumbers"])
    app.nozzels_var.set(params["Nozzels"])
    app.carry_arc_var.set(params["CarryArc"])
    app.combine_mode_var.set(params["CombineMode"])
    root.mainloop()
    return json.dumps(app.confirmed_result_json or {"plates": []}, ensure_ascii=False)


if __name__ == "__main__":
    root = tk.Tk()
    app = NestingApp(root)
    root.mainloop()
