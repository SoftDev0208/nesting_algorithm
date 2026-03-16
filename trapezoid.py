import json
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union


# -----------------------------
# Load JSON
# -----------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


# -----------------------------
# Convert JSON to polygon
# -----------------------------
def part_to_polygon(part):

    pts = part['contours'][0]['points']
    poly = Polygon([(p['x'], p['y']) for p in pts])

    # center polygon at origin
    cx, cy = poly.centroid.x, poly.centroid.y
    poly = translate(poly, xoff=-cx, yoff=-cy)

    return poly


# -----------------------------
# Rectangle area
# -----------------------------
def minimal_rect_area(polys):

    union = unary_union(polys)
    rect = union.minimum_rotated_rectangle

    return rect.area, rect


# -----------------------------
# Try placement
# -----------------------------
def try_place(base_polys, poly):

    best = None
    best_area = float("inf")

    rotations = [0, 90, 180, 270]

    for angle in rotations:

        p = rotate(poly, angle, origin=(0,0))

        # coarse search
        for dx in np.arange(-800,800,40):
            for dy in np.arange(-800,800,40):

                moved = translate(p,dx,dy)

                if any(moved.intersects(b) for b in base_polys):
                    continue

                area,_ = minimal_rect_area(base_polys+[moved])

                if area < best_area:
                    best_area = area
                    best = moved

    return best


# -----------------------------
# Best arrangement
# -----------------------------
def best_arrangement(poly, quantity):

    placed = [poly]

    for i in range(quantity-1):

        best = try_place(placed, poly)

        if best is None:
            break

        placed.append(best)

    return placed


# -----------------------------
# Plot
# -----------------------------
def plot(polys):

    fig, ax = plt.subplots()

    for poly in polys:
        x,y = poly.exterior.xy
        ax.plot(x,y,'b')

    union = unary_union(polys)
    rect = union.minimum_rotated_rectangle

    x,y = rect.exterior.xy
    ax.plot(x,y,'g',linewidth=2)

    ax.set_aspect("equal")
    plt.title("Minimal rotated sheet")
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
data = load_json("input.json")

part = data["parts"][0]

poly = part_to_polygon(part)

qty = part["quantity"]

best_polys = best_arrangement(poly, qty)

plot(best_polys)