import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box
from shapely.ops import unary_union


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


def rotate_normalize(poly, angle):
    return normalize_poly(rotate(poly, angle, origin="centroid", use_radians=False))


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
# Bar-like role helpers
# -----------------------------
def minimum_rotated_rect_dims(poly):
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:-1]

    lengths = []
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        lengths.append(math.hypot(x2 - x1, y2 - y1))

    long_side = max(lengths)
    short_side = min(lengths)
    return long_side, short_side, mrr.area


def is_bar_like(poly, min_aspect=3.0, min_fill=0.70):
    long_side, short_side, mrr_area = minimum_rotated_rect_dims(poly)

    if short_side < 1e-9 or mrr_area < 1e-9:
        return False

    aspect = long_side / short_side
    fill_ratio = poly.area / mrr_area

    return aspect >= min_aspect and fill_ratio >= min_fill


def orientation_profile(item, target="horizontal"):
    best = None

    for angle in [0, 90, 180, 270]:
        rotated = rotate_normalize(item["poly"], angle)
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


def split_bar_roles(items):
    top_shelf = None
    side_bars = []
    fillers = list(items)

    bar_like = [p for p in items if is_bar_like(p["poly"])]

    if bar_like:
        top_shelf = max(
            bar_like,
            key=lambda item: orientation_profile(item, "horizontal")["score"],
        )
        if top_shelf in fillers:
            fillers.remove(top_shelf)
        bar_like.remove(top_shelf)

    side_bars = sorted(
        bar_like,
        key=lambda item: orientation_profile(item, "vertical")["score"],
        reverse=True,
    )

    for item in side_bars:
        if item in fillers:
            fillers.remove(item)

    return top_shelf, side_bars, fillers


# -----------------------------
# Trapezoid pairing
# -----------------------------
def combine_trapezoids(poly):
    """
    Create a compact pair from two trapezoids.
    This is still the slowest part of the script.
    """
    best_area = float("inf")
    best_union = None

    rotations = [0, 90, 180, 270]

    for angle1 in rotations:
        for angle2 in rotations:
            p1 = rotate(poly, angle1, origin=(0, 0), use_radians=False)
            p2 = rotate(poly, angle2, origin=(0, 0), use_radians=False)

            for dx in np.linspace(-1000, 1000, 60):
                for dy in np.linspace(-1000, 1000, 60):
                    p2_moved = translate(p2, xoff=dx, yoff=dy)

                    if p1.intersection(p2_moved).area > 1e-6:
                        continue

                    combined = p1.union(p2_moved)
                    area = bounding_area(combined)

                    if area < best_area:
                        best_area = area
                        best_union = combined

    return best_union


# -----------------------------
# Build part instances
# -----------------------------
def build_parts_list(parts):
    parts_list = []

    for part in parts:
        poly = part_to_polygon(part)
        qty = int(part.get("quantity", 1))
        pid = str(part["id"])
        shape_type = detect_shape_type(part, poly)

        if shape_type == "trapezoid" and qty >= 2:
            pair_poly = combine_trapezoids(poly)

            if pair_poly is not None:
                pair_count = qty // 2
                remainder = qty % 2

                for _ in range(pair_count):
                    parts_list.append(
                        {
                            "id": pid + "_pair",
                            "base_id": pid,
                            "poly": pair_poly,
                            "shape": "trapezoid",
                            "priority": shape_priority("trapezoid"),
                            "area": bounding_area(pair_poly),
                        }
                    )

                if remainder == 1:
                    parts_list.append(
                        {
                            "id": pid,
                            "base_id": pid,
                            "poly": poly,
                            "shape": shape_type,
                            "priority": shape_priority(shape_type),
                            "area": bounding_area(poly),
                        }
                    )
            else:
                for _ in range(qty):
                    parts_list.append(
                        {
                            "id": pid,
                            "base_id": pid,
                            "poly": poly,
                            "shape": shape_type,
                            "priority": shape_priority(shape_type),
                            "area": bounding_area(poly),
                        }
                    )
        else:
            for _ in range(qty):
                parts_list.append(
                    {
                        "id": pid,
                        "base_id": pid,
                        "poly": poly,
                        "shape": shape_type,
                        "priority": shape_priority(shape_type),
                        "area": bounding_area(poly),
                    }
                )

    parts_list.sort(key=lambda p: (p["priority"], -p["area"]))
    return parts_list


def split_parts_by_shape(parts_list):
    rectangles = [p for p in parts_list if p["shape"] == "rectangle"]
    trapezoids = [p for p in parts_list if p["shape"] == "trapezoid"]
    pentagons = [p for p in parts_list if p["shape"] == "pentagon"]
    hexagons = [p for p in parts_list if p["shape"] == "hexagon"]
    circles = [p for p in parts_list if p["shape"] == "circle"]

    others = [
        p
        for p in parts_list
        if p["shape"] not in {"rectangle", "trapezoid", "pentagon", "hexagon", "circle"}
    ]

    return rectangles, trapezoids, pentagons, hexagons, others, circles


# -----------------------------
# Debug helpers
# -----------------------------
def init_place_debug(item, mode):
    return {
        "id": item["id"],
        "shape": item["shape"],
        "mode": mode,
        "rotations": [],
        "too_large": [],
        "tested_positions": 0,
        "outside_plate": 0,
        "overlap": 0,
        "valid_positions": 0,
        "candidate_positions": 0,
    }


def format_place_debug(debug):
    parts = [
        f"Could not place part {debug['id']}",
        f"shape={debug['shape']}",
        f"mode={debug['mode']}",
    ]

    if debug["too_large"] and debug["tested_positions"] == 0:
        dims = ", ".join(
            [f"angle={a} size=({w:.2f},{h:.2f})" for a, w, h in debug["too_large"]]
        )
        parts.append(f"reason=too large for plate [{dims}]")
        return " | ".join(parts)

    if debug["candidate_positions"] == 0:
        parts.append("reason=no candidate positions generated")
        return " | ".join(parts)

    if debug["valid_positions"] == 0:
        parts.append(
            "reason=no valid position "
            f"(tested={debug['tested_positions']}, "
            f"outside={debug['outside_plate']}, "
            f"overlap={debug['overlap']})"
        )
        return " | ".join(parts)

    parts.append(
        f"reason=no selected candidate "
        f"(tested={debug['tested_positions']}, valid={debug['valid_positions']})"
    )
    return " | ".join(parts)


# -----------------------------
# Collision / candidate helpers
# -----------------------------
def has_real_overlap(candidate, placed_parts, eps=1e-6):
    for p in placed_parts:
        if candidate.intersection(p["poly"]).area > eps:
            return True
    return False


def valid_candidate(candidate, placed_parts, plate_poly):
    return plate_poly.covers(candidate) and not has_real_overlap(candidate, placed_parts)


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
def cluster_bounds(placed_parts, fallback_poly):
    if not placed_parts:
        return fallback_poly.bounds

    minx = min(p["poly"].bounds[0] for p in placed_parts)
    miny = min(p["poly"].bounds[1] for p in placed_parts)
    maxx = max(p["poly"].bounds[2] for p in placed_parts)
    maxy = max(p["poly"].bounds[3] for p in placed_parts)
    return (minx, miny, maxx, maxy)


def place_item_bottom_left(item, placed_parts, plate_poly, preferred_angles=None):
    if preferred_angles is None:
        preferred_angles = [0, 90, 180, 270]

    best = None
    best_score = None
    debug = init_place_debug(item, "bottom-left")

    plate_minx, plate_miny, plate_maxx, plate_maxy = plate_poly.bounds
    plate_w = plate_maxx - plate_minx
    plate_h = plate_maxy - plate_miny

    for angle in preferred_angles:
        debug["rotations"].append(angle)

        rotated = rotate_normalize(item["poly"], angle)
        rw, rh = poly_size(rotated)

        if rw > plate_w or rh > plate_h:
            debug["too_large"].append((angle, rw, rh))
            continue

        positions = generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh)
        debug["candidate_positions"] += len(positions)

        for x, y in positions:
            candidate = translate(rotated, xoff=x, yoff=y)
            debug["tested_positions"] += 1

            if not plate_poly.covers(candidate):
                debug["outside_plate"] += 1
                continue

            if has_real_overlap(candidate, placed_parts):
                debug["overlap"] += 1
                continue

            debug["valid_positions"] += 1
            score = (y, x, rw * rh)

            if best is None or score < best_score:
                best = {
                    "id": item["id"],
                    "poly": candidate,
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                }
                best_score = score

    return best, debug


def place_item_top_left(item, placed_parts, plate_poly, preferred_angles=None):
    if preferred_angles is None:
        preferred_angles = [0, 90, 180, 270]

    _, _, _, plate_top = plate_poly.bounds
    best = None
    best_score = None
    debug = init_place_debug(item, "top-left")

    plate_minx, plate_miny, plate_maxx, plate_maxy = plate_poly.bounds
    plate_w = plate_maxx - plate_minx
    plate_h = plate_maxy - plate_miny

    for angle in preferred_angles:
        debug["rotations"].append(angle)

        rotated = rotate_normalize(item["poly"], angle)
        rw, rh = poly_size(rotated)

        if rw > plate_w or rh > plate_h:
            debug["too_large"].append((angle, rw, rh))
            continue

        positions = generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh)
        debug["candidate_positions"] += len(positions)

        for x, y in positions:
            candidate = translate(rotated, xoff=x, yoff=y)
            debug["tested_positions"] += 1

            if not plate_poly.covers(candidate):
                debug["outside_plate"] += 1
                continue

            if has_real_overlap(candidate, placed_parts):
                debug["overlap"] += 1
                continue

            debug["valid_positions"] += 1
            minx, miny, maxx, maxy = candidate.bounds
            score = (plate_top - maxy, minx, rw * rh)

            if best is None or score < best_score:
                best = {
                    "id": item["id"],
                    "poly": candidate,
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                }
                best_score = score

    return best, debug


def place_item_near_reference_vertical(item, placed_parts, plate_poly, ref_bounds):
    ref_right = ref_bounds[2]
    best = None
    best_score = None
    debug = init_place_debug(item, "right-lane-vertical")

    plate_minx, plate_miny, plate_maxx, plate_maxy = plate_poly.bounds
    plate_w = plate_maxx - plate_minx
    plate_h = plate_maxy - plate_miny

    for angle in [90, 270]:
        debug["rotations"].append(angle)

        rotated = rotate_normalize(item["poly"], angle)
        rw, rh = poly_size(rotated)

        if rw > plate_w or rh > plate_h:
            debug["too_large"].append((angle, rw, rh))
            continue

        positions = generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh)
        debug["candidate_positions"] += len(positions)

        for x, y in positions:
            candidate = translate(rotated, xoff=x, yoff=y)
            debug["tested_positions"] += 1

            if not plate_poly.covers(candidate):
                debug["outside_plate"] += 1
                continue

            if has_real_overlap(candidate, placed_parts):
                debug["overlap"] += 1
                continue

            debug["valid_positions"] += 1
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
                    "poly": candidate,
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                }
                best_score = score

    return best, debug


def place_item_generic(item, placed_parts, plate_poly):
    if item["shape"] == "trapezoid":
        return place_item_bottom_left(item, placed_parts, plate_poly)
    return place_item_top_left(item, placed_parts, plate_poly)


# -----------------------------
# Free-region helpers
# -----------------------------
def get_polygon_regions(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty and g.area > 1e-6]
    return [geom] if geom.area > 1e-6 else []


def get_free_geom(plate_poly, placed_parts):
    if not placed_parts:
        return plate_poly
    occupied = unary_union([p["poly"] for p in placed_parts])
    return plate_poly.difference(occupied)


# -----------------------------
# Circle helpers
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


def get_circle_pack_orders():
    return [
        ("hex", "bl"),
        ("hex", "br"),
        ("hex", "tl"),
        ("hex", "tr"),
        ("grid", "bl"),
        ("grid", "br"),
        ("grid", "tl"),
        ("grid", "tr"),
    ]


def get_center_regions_for_circles(plate_poly, placed_parts, radius, eps=1e-6):
    free_geom = get_free_geom(plate_poly, placed_parts)
    center_geom = free_geom.buffer(-(radius - eps))
    center_regions = get_polygon_regions(center_geom)
    center_regions.sort(key=lambda g: g.area, reverse=True)
    return center_regions


def generate_centers_in_center_region(center_region, radius, mode="hex", anchor="bl"):
    minx, miny, maxx, maxy = center_region.bounds
    centers = []
    eps = 1e-9

    if mode == "hex":
        step_x = 2.0 * radius
        step_y = math.sqrt(3.0) * radius

        row = 0
        y = miny
        while y <= maxy + eps:
            x_offset = radius if row % 2 == 1 else 0.0
            x = minx + x_offset
            while x <= maxx + eps:
                if center_region.covers(Point(x, y)):
                    centers.append((x, y))
                x += step_x
            y += step_y
            row += 1
    else:
        step = 2.0 * radius
        y = miny
        while y <= maxy + eps:
            x = minx
            while x <= maxx + eps:
                if center_region.covers(Point(x, y)):
                    centers.append((x, y))
                x += step
            y += step

    return sort_centers_by_anchor(centers, anchor)


def pack_circles_in_remaining_space(parts, plate_poly, placed_parts, circle_count):
    circle_poly_norm, radius, diameter, circle_pid = get_circle_template(parts)
    if circle_poly_norm is None or circle_count <= 0:
        return placed_parts, 0

    free_geom = get_free_geom(plate_poly, placed_parts)
    center_regions = get_center_regions_for_circles(plate_poly, placed_parts, radius)

    if not center_regions:
        return placed_parts, 0

    best_added = []
    best_count = -1

    for mode, anchor in get_circle_pack_orders():
        added = []

        for center_region in center_regions:
            centers = generate_centers_in_center_region(center_region, radius, mode=mode, anchor=anchor)

            for cx, cy in centers:
                candidate = make_circle_at(circle_poly_norm, radius, cx, cy)

                if not free_geom.covers(candidate):
                    continue

                overlap_new = False
                for poly in added:
                    if candidate.intersection(poly).area > 1e-6:
                        overlap_new = True
                        break
                if overlap_new:
                    continue

                added.append(candidate)

                if len(added) >= circle_count:
                    break

            if len(added) >= circle_count:
                break

        if len(added) > best_count:
            best_count = len(added)
            best_added = added

        if best_count >= circle_count:
            break

    for poly in best_added:
        minx, miny, maxx, maxy = poly.bounds
        placed_parts.append(
            {
                "id": circle_pid,
                "poly": poly,
                "x": minx,
                "y": miny,
                "angle": 0,
                "shape": "circle",
            }
        )
        print(f"Placed part {circle_pid} (shape=circle) at x={minx}, y={miny}, angle=0")

    return placed_parts, len(best_added)


# -----------------------------
# Role inference from geometry
# -----------------------------
def infer_layout_roles(parts_list):
    circles = [p for p in parts_list if p["shape"] == "circle"]
    pair_trapezoids = [p for p in parts_list if p["shape"] == "trapezoid" and p["id"].endswith("_pair")]

    remaining = [p for p in parts_list if p["shape"] != "circle" and p not in pair_trapezoids]
    rect_like = [p for p in remaining if p["shape"] == "rectangle"]

    top_shelf = None
    if rect_like:
        top_shelf = max(rect_like, key=lambda item: orientation_profile(item, "horizontal")["score"])
        remaining.remove(top_shelf)

    rect_like_remaining = [p for p in remaining if p["shape"] == "rectangle"]
    side_bars = sorted(
        rect_like_remaining,
        key=lambda item: orientation_profile(item, "vertical")["score"],
        reverse=True,
    )

    for item in side_bars:
        if item in remaining:
            remaining.remove(item)

    fillers = sorted(remaining, key=lambda p: (p["priority"], -p["area"]))
    pair_trapezoids.sort(key=lambda p: -p["area"])

    return circles, pair_trapezoids, top_shelf, side_bars, fillers


# -----------------------------
# Main nesting
# -----------------------------
def nest_parts_with_full_fit(parts, plate_poly):
    parts_list = build_parts_list(parts)
    circles, pair_trapezoids, top_shelf, side_bars, fillers = infer_layout_roles(parts_list)

    placed_parts = []

    # Step 1: trapezoid pairs
    for item in pair_trapezoids:
        candidate, dbg = place_item_bottom_left(
            item=item,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            preferred_angles=[0, 90, 180, 270],
        )
        if candidate is not None:
            placed_parts.append(candidate)
            print(
                f"Placed part {candidate['id']} "
                f"(shape={candidate['shape']}) "
                f"at x={candidate['x']}, y={candidate['y']}, angle={candidate['angle']}"
            )
        else:
            print(format_place_debug(dbg))

    # Step 2: top shelf
    if top_shelf is not None:
        top_profile = orientation_profile(top_shelf, "horizontal")
        candidate, dbg = place_item_top_left(
            item=top_shelf,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            preferred_angles=[top_profile["angle"]],
        )
        if candidate is not None:
            placed_parts.append(candidate)
            print(
                f"Placed part {candidate['id']} "
                f"(shape={candidate['shape']}) "
                f"at x={candidate['x']}, y={candidate['y']}, angle={candidate['angle']}"
            )
        else:
            print(format_place_debug(dbg))

    # Step 3: side bars
    for item in side_bars:
        side_profile = orientation_profile(item, "vertical")
        ref_bounds = cluster_bounds(placed_parts, plate_poly)

        candidate, dbg = place_item_near_reference_vertical(
            item=item,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            ref_bounds=ref_bounds,
        )

        if candidate is None:
            candidate, dbg2 = place_item_bottom_left(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                preferred_angles=[side_profile["angle"], 0, 180],
            )
            if candidate is None:
                print(format_place_debug(dbg))
                print("  fallback -> " + format_place_debug(dbg2))
            else:
                placed_parts.append(candidate)
                print(
                    f"Placed part {candidate['id']} "
                    f"(shape={candidate['shape']}) "
                    f"at x={candidate['x']}, y={candidate['y']}, angle={candidate['angle']}"
                )
        else:
            placed_parts.append(candidate)
            print(
                f"Placed part {candidate['id']} "
                f"(shape={candidate['shape']}) "
                f"at x={candidate['x']}, y={candidate['y']}, angle={candidate['angle']}"
            )

    # Step 4: fillers
    for item in fillers:
        candidate, dbg = place_item_generic(item, placed_parts, plate_poly)
        if candidate is not None:
            placed_parts.append(candidate)
            print(
                f"Placed part {candidate['id']} "
                f"(shape={candidate['shape']}) "
                f"at x={candidate['x']}, y={candidate['y']}, angle={candidate['angle']}"
            )
        else:
            print(format_place_debug(dbg))

    # Step 5: circles in true remaining space
    placed_parts, circle_count = pack_circles_in_remaining_space(
        parts=parts,
        plate_poly=plate_poly,
        placed_parts=placed_parts,
        circle_count=len(circles),
    )

    if circle_count < len(circles):
        circle_pid = str(circles[0]["base_id"]) if circles else "circle"
        for _ in range(len(circles) - circle_count):
            print(f"Could not place part {circle_pid}, not enough space")

    print(f"Total placed parts: {len(placed_parts)}")
    return placed_parts


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

        if isinstance(poly, MultiPolygon):
            polys = poly.geoms
        else:
            polys = [poly]

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

    ax.set_title("Layout Filled - Circles Packed In True Remaining Space")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# -----------------------------
# Summary table
# -----------------------------
def generate_table(placed_parts):
    summary = {}
    for p in placed_parts:
        summary[p["id"]] = summary.get(p["id"], 0) + 1

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
        placed_parts = nest_parts_with_full_fit(self.parts, self.plate_poly)
        plot_arrangement(placed_parts, self.plate_poly)
        generate_table(placed_parts)


if __name__ == "__main__":
    root = tk.Tk()
    app = NestingApp(root)
    root.mainloop()