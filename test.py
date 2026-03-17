import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint, box, Point
from shapely.ops import triangulate, unary_union
from shapely.affinity import rotate, scale, translate
from shapely.validation import make_valid

EPS = 1e-7
DEFAULT_CIRCLE_QUAD_SEGS = 12
DEFAULT_BUFFER_QUAD_SEGS = 8

# ----------------------------
# Data classes
# ----------------------------
@dataclass(frozen=True)
class Item:
    template_id: str
    sequence: int

@dataclass
class ShapeVariant:
    key: str
    part_id: str
    angle: float
    mirrored: bool
    geom: Polygon
    gap_geom: Polygon
    neg_gap_geom: Polygon
    gap_triangles: Tuple[Polygon, ...]
    neg_gap_triangles: Tuple[Polygon, ...]
    area: float
    bounds: Tuple[float, float, float, float]
    width: float
    height: float
    gap_convex: bool
    neg_gap_convex: bool

@dataclass
class SearchState:
    placements: Tuple[Item, ...]
    remaining: Tuple[Item, ...]
    placed_area: float
    used_maxx: float
    used_maxy: float

# ----------------------------
# Geometry helpers
# ----------------------------
def clean_geometry(geom):
    if geom is None or geom.is_empty:
        return Polygon()
    geom = make_valid(geom)
    if geom.is_empty:
        return Polygon()
    return geom

def geometry_triangles(geom) -> Tuple[Polygon, ...]:
    tris = []
    for tri in triangulate(geom):
        if tri.area > EPS:
            tris.append(clean_geometry(tri))
    return tuple(tris)

def dedupe_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out = []
    for x, y in points:
        if not out or abs(x - out[-1][0]) > EPS or abs(y - out[-1][1]) > EPS:
            out.append((x, y))
    if len(out) > 1 and abs(out[0][0]-out[-1][0]) <= EPS and abs(out[0][1]-out[-1][1]) <= EPS:
        out.pop()
    return out

def contour_to_polygon(contour, basepoint) -> Polygon:
    bx = float(basepoint.get("x", 0))
    by = float(basepoint.get("y", 0))
    pts = contour.get("points", [])
    coords = []
    for p in pts:
        x = float(p.get("x", 0)) - bx
        y = float(p.get("y", 0)) - by
        coords.append((x, y))
    coords = dedupe_points(coords)
    if len(coords) < 3:
        return Polygon()
    return clean_geometry(Polygon(coords))

def part_to_geometry(part) -> Polygon:
    geom = None
    basepoint = part.get("basepoint", {})
    for contour in part.get("contours", []):
        poly = contour_to_polygon(contour, basepoint)
        if geom is None:
            geom = poly
        else:
            geom = geom.symmetric_difference(poly)
    if geom is None:
        geom = Polygon()
    return clean_geometry(geom)

# ----------------------------
# Polygon Nesting Engine
# ----------------------------
class PolygonNestingEngine:
    def __init__(self, parts, plate_poly, options=None, beam_width=3, part_window=3, branch_factor=3):
        self.parts = parts
        self.plate_poly = clean_geometry(plate_poly)
        self.options = options or {}
        self.gap = max(0.0, float(self.options.get("PartGap", 0.0) or 0.0) - 1e-3)
        self.items = []
        self._build_items()

    def _build_items(self):
        seq = 1
        for part in self.parts:
            quantity = int(part.get("quantity", 1))
            for _ in range(quantity):
                self.items.append(Item(template_id=part["id"], sequence=seq))
                seq += 1

    def nest(self):
        # Simplified greedy shape-aware placement
        placements = []
        x_cursor = 0
        y_cursor = 0
        max_height_in_row = 0
        for item in self.items:
            part = next(p for p in self.parts if p["id"] == item.template_id)
            geom = part_to_geometry(part)
            minx, miny, maxx, maxy = geom.bounds
            w, h = maxx - minx, maxy - miny
            if x_cursor + w > 12000:
                x_cursor = 0
                y_cursor += max_height_in_row
                max_height_in_row = 0
            if y_cursor + h > 1500:
                continue
            placements.append({"id": item.template_id, "x": x_cursor, "y": y_cursor, "geom": geom})
            x_cursor += w + self.gap
            max_height_in_row = max(max_height_in_row, h)
        return placements

# ----------------------------
# Run Nesting
# ----------------------------
with open("test1.json", "r") as f:
    data = json.load(f)

plate = box(0,0,12000,1500)
engine = PolygonNestingEngine(data.get("parts", []), plate)
placements = engine.nest()

# ----------------------------
# Summary
# ----------------------------
counts = {}
for p in placements:
    counts[p["id"]] = counts.get(p["id"], 0) + 1

print(f"{'ID':<8}{'Placed':<10}")
for pid, qty in counts.items():
    print(f"{pid:<8}{qty:<10}")

total_parts = sum(part['quantity'] for part in data['parts'])
print(f"\nTotal placed parts: {len(placements)} / {total_parts}")
print(f"Total unplaced parts: {total_parts - len(placements)}")

# ----------------------------
# Plot placements
# ----------------------------
fig, ax = plt.subplots(figsize=(16,6))
ax.set_xlim(0,12000)
ax.set_ylim(0,1500)
ax.set_aspect('equal')

for p in placements:
    geom = p["geom"]
    if geom.is_empty:  # skip empty polygons
        continue
    # Draw exterior
    x, y = geom.exterior.xy
    ax.plot(x, y, color='blue')
    # Draw label at representative point
    rp = geom.representative_point()
    ax.text(rp.x, rp.y, p["id"], ha='center', va='center', fontsize=8, color='green')

ax.set_title("2D Polygon Nesting - All Parts")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.tight_layout()
plt.show()