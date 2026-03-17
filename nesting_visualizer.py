import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box


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
    """
    Normalized parallel check.
    Much more stable than raw cross-product threshold for large coordinates.
    """
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
    """
    Detect 6-point / 8-point bar-like shapes that are basically rectangles
    with chamfered corners.
    """
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) not in (6, 8):
        return False

    mrr = poly.minimum_rotated_rectangle
    if mrr.area <= 1e-9:
        return False

    ratio = poly.area / mrr.area
    return ratio >= area_ratio_tol


def is_trapezoid(poly, tol=1e-4):
    """
    General trapezoid:
    - 4 corners
    - not a rectangle
    - at least one pair of opposite sides parallel
    """
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
    This is still the slowest part of the script.
    """
    best_area = float("inf")
    best_union = None

    rotations = [0, 90, 180, 270]

    # Reduced grid for speed
    for angle1 in rotations:
        for angle2 in rotations:
            p1 = rotate(poly, angle1, origin=(0, 0), use_radians=False)
            p2 = rotate(poly, angle2, origin=(0, 0), use_radians=False)

            for dx in np.linspace(-1000, 1000, 70):
                for dy in np.linspace(-1000, 1000, 70):
                    p2_moved = translate(p2, xoff=dx, yoff=dy)

                    # allow touching, forbid real overlap
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


# -----------------------------
# Collision helpers
# -----------------------------
def has_real_overlap(candidate, placed_parts, eps=1e-6):
    for p in placed_parts:
        if candidate.intersection(p["poly"]).area > eps:
            return True
    return False


def valid_candidate(candidate, placed_parts, plate_poly):
    return plate_poly.contains(candidate) and not has_real_overlap(candidate, placed_parts)


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
        preferred_angles = [0, 90, 180, 270]

    best = None
    best_score = None

    for angle in preferred_angles:
        rotated = rotate_normalize(item["poly"], angle)
        rw, rh = poly_size(rotated)

        for x, y in generate_candidate_positions_for_part(placed_parts, plate_poly, rw, rh):
            candidate = translate(rotated, xoff=x, yoff=y)
            if not valid_candidate(candidate, placed_parts, plate_poly):
                continue

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

    return best


def place_item_top_left(item, placed_parts, plate_poly, preferred_angles=None):
    if preferred_angles is None:
        preferred_angles = [0, 90, 180, 270]

    _, _, _, plate_top = plate_poly.bounds
    best = None
    best_score = None

    for angle in preferred_angles:
        rotated = rotate_normalize(item["poly"], angle)
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
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                }
                best_score = score

    return best


def cluster_bounds(placed_parts, fallback_poly):
    if not placed_parts:
        return fallback_poly.bounds

    minx = min(p["poly"].bounds[0] for p in placed_parts)
    miny = min(p["poly"].bounds[1] for p in placed_parts)
    maxx = max(p["poly"].bounds[2] for p in placed_parts)
    maxy = max(p["poly"].bounds[3] for p in placed_parts)
    return (minx, miny, maxx, maxy)


def place_item_near_reference_vertical(item, placed_parts, plate_poly, ref_bounds):
    """
    Prefer vertical placement near the current cluster right edge.
    """
    ref_right = ref_bounds[2]
    best = None
    best_score = None

    for angle in [90, 270]:
        rotated = rotate_normalize(item["poly"], angle)
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
                    "poly": candidate,
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "shape": item["shape"],
                }
                best_score = score

    return best


def place_item_generic(item, placed_parts, plate_poly):
    """
    Fallback:
    - trapezoid-like parts -> bottom-left
    - others -> top-left
    """
    if item["shape"] == "trapezoid":
        return place_item_bottom_left(item, placed_parts, plate_poly)
    return place_item_top_left(item, placed_parts, plate_poly)


# -----------------------------
# Circle reservation / packing
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


def reserve_top_hex_circle_band(parts, plate_poly, circle_count):
    """
    Reserve enough top area for all circles in a hex band.
    """
    circle_poly_norm, radius, diameter, circle_pid = get_circle_template(parts)
    if circle_poly_norm is None or circle_count <= 0:
        return [], plate_poly

    minx, miny, maxx, maxy = plate_poly.bounds
    width = maxx - minx

    row1_cap = int(width // (2.0 * radius))
    row2_cap = max(0, int((width - radius) // (2.0 * radius)))

    if row1_cap <= 0:
        return [], plate_poly

    reserved = []
    remaining = circle_count
    rows = []
    row_index = 0

    while remaining > 0:
        cap = row1_cap if row_index % 2 == 0 else row2_cap
        if cap <= 0:
            break
        take = min(cap, remaining)
        rows.append((row_index, take))
        remaining -= take
        row_index += 1

    if remaining > 0:
        return [], plate_poly

    step_y = math.sqrt(3.0) * radius

    for row_index, take in rows:
        cy = maxy - radius - row_index * step_y
        x_offset = 0.0 if row_index % 2 == 0 else radius
        cx = minx + radius + x_offset

        for _ in range(take):
            candidate = make_circle_at(circle_poly_norm, radius, cx, cy)
            if not plate_poly.contains(candidate):
                return [], plate_poly

            reserved.append(
                {
                    "id": circle_pid,
                    "poly": candidate,
                    "x": cx - radius,
                    "y": cy - radius,
                    "angle": 0,
                    "shape": "circle",
                }
            )
            cx += 2.0 * radius

    band_bottom = min(p["poly"].bounds[1] for p in reserved)
    working_plate = box(minx, miny, maxx, band_bottom)
    return reserved, working_plate


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


def generate_circle_centers(plate_poly, radius, mode="hex", anchor="bl"):
    minx, miny, maxx, maxy = plate_poly.bounds
    centers = []
    eps = 1e-9

    if mode == "hex":
        step_x = 2.0 * radius
        step_y = math.sqrt(3.0) * radius

        row = 0
        y = miny + radius
        while y <= maxy - radius + eps:
            x_offset = radius if row % 2 == 1 else 0.0
            x = minx + radius + x_offset
            while x <= maxx - radius + eps:
                centers.append((x, y))
                x += step_x
            y += step_y
            row += 1
    else:
        step = 2.0 * radius
        y = miny + radius
        while y <= maxy - radius + eps:
            x = minx + radius
            while x <= maxx - radius + eps:
                centers.append((x, y))
                x += step
            y += step

    return sort_centers_by_anchor(centers, anchor)


def get_circle_pack_orders():
    return [
        ("hex", "tr"),
        ("hex", "tl"),
        ("hex", "br"),
        ("hex", "bl"),
        ("grid", "tr"),
        ("grid", "tl"),
        ("grid", "br"),
        ("grid", "bl"),
    ]


def place_circles_best_pattern(parts, plate_poly, placed_parts, circle_count):
    circle_poly_norm, radius, diameter, circle_pid = get_circle_template(parts)
    if circle_poly_norm is None or circle_count <= 0:
        return placed_parts, 0

    best_layout = None
    best_count = -1

    for mode, anchor in get_circle_pack_orders():
        temp_parts = list(placed_parts)
        count = 0

        centers = generate_circle_centers(plate_poly, radius, mode=mode, anchor=anchor)

        for cx, cy in centers:
            candidate = make_circle_at(circle_poly_norm, radius, cx, cy)

            if not plate_poly.contains(candidate):
                continue

            if has_real_overlap(candidate, temp_parts):
                continue

            temp_parts.append(
                {
                    "id": circle_pid,
                    "poly": candidate,
                    "x": cx - radius,
                    "y": cy - radius,
                    "angle": 0,
                    "shape": "circle",
                }
            )
            count += 1

            if count >= circle_count:
                break

        if count > best_count:
            best_count = count
            best_layout = temp_parts

        if best_count >= circle_count:
            break

    original_count = len(placed_parts)
    final_parts = best_layout if best_layout is not None else placed_parts

    for p in final_parts[original_count:]:
        print(f"Placed part {p['id']} (shape=circle) at x={p['x']}, y={p['y']}, angle=0")

    return final_parts, max(0, best_count)


# -----------------------------
# Role inference from geometry
# -----------------------------
def orientation_profile(item, target="horizontal"):
    """
    Choose the best rotation for a layout role:
    - horizontal shelf
    - vertical side bar
    """
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


def infer_layout_roles(parts_list):
    """
    Infer roles from geometry only, not IDs.

    Roles:
    - circles
    - paired trapezoids
    - one top horizontal shelf
    - remaining bar-like rectangles as vertical bars
    - fillers
    """
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
# -----------------------------
# Main nesting: circles in remaining free space
# -----------------------------
def nest_parts_with_full_fit(parts, plate_poly):
    """
    Nest all parts and place circles only in remaining space after other parts are placed.
    """
    parts_list = build_parts_list(parts)
    circles, pair_trapezoids, top_shelf, side_bars, fillers = infer_layout_roles(parts_list)

    placed_parts = []

    # Step 1: paired trapezoids -> bottom-left
    for item in pair_trapezoids:
        candidate = place_item_bottom_left(
            item=item,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
        )
        if candidate:
            placed_parts.append(candidate)

    # Step 2: best horizontal shelf -> top-left
    if top_shelf:
        top_profile = orientation_profile(top_shelf, "horizontal")
        candidate = place_item_top_left(
            item=top_shelf,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            preferred_angles=[top_profile["angle"]],
        )
        if candidate:
            placed_parts.append(candidate)

    # Step 3: vertical bars
    for item in side_bars:
        side_profile = orientation_profile(item, "vertical")
        ref_bounds = cluster_bounds(placed_parts, plate_poly)
        candidate = place_item_near_reference_vertical(
            item=item,
            placed_parts=placed_parts,
            plate_poly=plate_poly,
            ref_bounds=ref_bounds,
        )
        if candidate is None:
            candidate = place_item_bottom_left(
                item=item,
                placed_parts=placed_parts,
                plate_poly=plate_poly,
                preferred_angles=[side_profile["angle"], 0, 180],
            )
        if candidate:
            placed_parts.append(candidate)

    # Step 4: fillers
    for item in fillers:
        candidate = place_item_generic(item, placed_parts, plate_poly)
        if candidate:
            placed_parts.append(candidate)

    # Step 5: circles -> fit into remaining free space
    placed_parts, circle_count = place_circles_best_pattern(
        parts=parts,
        plate_poly=plate_poly,
        placed_parts=placed_parts,
        circle_count=len(circles),
    )

    if circle_count < len(circles):
        missing = len(circles) - circle_count
        circle_pid = str(circles[0]["base_id"]) if circles else "circle"
        for _ in range(missing):
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