from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate, scale, translate
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box
from shapely.ops import triangulate, unary_union
from shapely.validation import make_valid

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:  # pragma: no cover - optional in headless environments
    tk = None
    filedialog = None
    messagebox = None

EPS = 1e-7
DEFAULT_ARC_STEP_DEG = 8.0
DEFAULT_CIRCLE_QUAD_SEGS = 12
DEFAULT_BUFFER_QUAD_SEGS = 8


@dataclass(frozen=True, slots=True)
class Item:
    template_id: str
    sequence: int


@dataclass(slots=True)
class ShapeVariant:
    key: str
    part_id: str
    angle: float
    mirrored: bool
    geom: object
    gap_geom: object
    neg_gap_geom: object
    gap_triangles: Tuple[object, ...]
    neg_gap_triangles: Tuple[object, ...]
    area: float
    bounds: Tuple[float, float, float, float]
    width: float
    height: float
    gap_convex: bool
    neg_gap_convex: bool


@dataclass(slots=True)
class PartTemplate:
    part_id: str
    priority: int
    quantity: int
    base_geom: object
    variants: Tuple[ShapeVariant, ...]
    nominal_area: float
    width: float
    height: float
    max_dim: float
    bbox_fill: float
    concavity: float


@dataclass(slots=True)
class Placement:
    item: Item
    variant: ShapeVariant
    tx: float
    ty: float
    geom: object
    gap_geom: object
    score: Tuple[float, ...]
    component_area: float


@dataclass(slots=True)
class SearchState:
    placements: Tuple[Placement, ...]
    remaining: Tuple[Item, ...]
    placed_area: float
    used_maxx: float
    used_maxy: float

    @property
    def placed_count(self) -> int:
        return len(self.placements)

    @property
    def bbox_area(self) -> float:
        if self.placed_count == 0:
            return 0.0
        return max(0.0, self.used_maxx) * max(0.0, self.used_maxy)


# ----------------------------
# Geometry helpers
# ----------------------------
def load_json(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def iter_polygonal(geom) -> Iterable[object]:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for sub in geom.geoms:
            if not sub.is_empty and sub.area > EPS:
                yield sub
    elif geom.geom_type == "GeometryCollection":
        for sub in geom.geoms:
            yield from iter_polygonal(sub)


def clean_geometry(geom):
    if geom is None:
        return ShapelyPolygon()
    if geom.is_empty:
        return geom
    geom = make_valid(geom)
    polys = [g for g in iter_polygonal(geom) if g.area > EPS]
    if not polys:
        return ShapelyPolygon()
    geom = unary_union(polys)
    if geom.is_empty:
        return geom
    return geom


def dedupe_points(points: Sequence[Tuple[float, float]], tol: float = 1e-9) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for x, y in points:
        if not out or abs(x - out[-1][0]) > tol or abs(y - out[-1][1]) > tol:
            out.append((float(x), float(y)))
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= tol and abs(out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def arc_points(cx: float, cy: float, radius: float, angle1: float, angle2: float,
               step_deg: float = DEFAULT_ARC_STEP_DEG) -> List[Tuple[float, float]]:
    delta = angle2 - angle1
    if abs(delta) <= EPS:
        return [(cx + radius * math.cos(angle1), cy + radius * math.sin(angle1))]
    max_step = math.radians(max(1.0, step_deg))
    num = max(8, int(math.ceil(abs(delta) / max_step)) + 1)
    values = np.linspace(angle1, angle2, num=num, endpoint=True)
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in values]


def contour_to_polygon(contour: dict, basepoint: dict) -> object:
    bx = float(basepoint.get("x", 0.0) or 0.0)
    by = float(basepoint.get("y", 0.0) or 0.0)
    pts = contour.get("points", [])
    if not pts:
        return ShapelyPolygon()

    # Full circle contour: a single center point with a radius.
    if len(pts) == 1 and float(pts[0].get("radius", 0.0) or 0.0) > EPS:
        p = pts[0]
        cx = float(p.get("x", 0.0) or 0.0) - bx
        cy = float(p.get("y", 0.0) or 0.0) - by
        radius = float(p.get("radius", 0.0) or 0.0)
        return Point(cx, cy).buffer(radius, quad_segs=DEFAULT_CIRCLE_QUAD_SEGS)

    coords: List[Tuple[float, float]] = []
    for p in pts:
        x = float(p.get("x", 0.0) or 0.0) - bx
        y = float(p.get("y", 0.0) or 0.0) - by
        radius = float(p.get("radius", 0.0) or 0.0)
        angle1 = float(p.get("angle1", 0.0) or 0.0)
        angle2 = float(p.get("angle2", 0.0) or 0.0)

        if radius > EPS and abs(angle2 - angle1) > EPS:
            arc = arc_points(x, y, radius, angle1, angle2)
            if coords and arc:
                if abs(coords[-1][0] - arc[0][0]) <= 1e-9 and abs(coords[-1][1] - arc[0][1]) <= 1e-9:
                    arc = arc[1:]
            coords.extend(arc)
        else:
            coords.append((x, y))

    coords = dedupe_points(coords)
    if len(coords) < 3:
        return ShapelyPolygon()

    poly = ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = clean_geometry(poly.buffer(0))
    return clean_geometry(poly)


def part_to_geometry(part: dict):
    contours = part.get("contours", [])
    basepoint = part.get("basepoint", {}) or {}
    geom = None
    for contour in contours:
        cgeom = contour_to_polygon(contour, basepoint)
        if cgeom.is_empty:
            continue
        if geom is None:
            geom = cgeom
        else:
            # Even-odd fill rule handles shells, holes, and nested contours robustly.
            geom = geom.symmetric_difference(cgeom)
    if geom is None:
        geom = ShapelyPolygon()
    return clean_geometry(geom)




def is_convex_polygonal(geom) -> bool:
    polys = list(iter_polygonal(geom))
    if len(polys) != 1:
        return False
    poly = polys[0]
    if len(poly.interiors) > 0:
        return False
    return abs(poly.area - poly.convex_hull.area) <= max(1e-6, poly.area * 1e-9)

def geometry_triangles(geom) -> Tuple[object, ...]:
    triangles: List[object] = []
    for poly in iter_polygonal(geom):
        for tri in triangulate(poly):
            if tri.area <= EPS:
                continue
            rp = tri.representative_point()
            if rp.covered_by(poly):
                triangles.append(clean_geometry(tri))
    return tuple(t for t in triangles if not t.is_empty and t.area > EPS)


def convex_minkowski_sum(poly_a, poly_b):
    pts_a = list(poly_a.exterior.coords)[:-1]
    pts_b = list(poly_b.exterior.coords)[:-1]
    sums = [(ax + bx, ay + by) for ax, ay in pts_a for bx, by in pts_b]
    hull = MultiPoint(sums).convex_hull
    return clean_geometry(hull)


# ----------------------------
# Packing engine
# ----------------------------
class PolygonNestingEngine:
    def __init__(self, parts: Sequence[dict], plate_poly, options: Optional[dict] = None,
                 beam_width: int = 2, part_window: int = 2, branch_factor: int = 2,
                 candidate_limit: int = 24, per_variant_keep: int = 1, per_item_keep: int = 2,
                 verbose: bool = False):
        self.parts = list(parts)
        self.plate_poly = clean_geometry(plate_poly)
        self.options = options or {}
        self.gap = float(self.options.get("PartGap", 0.0) or 0.0)
        self.left_margin = float(self.options.get("LeftDist", 0.0) or 0.0)
        self.right_margin = float(self.options.get("RightDist", 0.0) or 0.0)
        self.bottom_margin = float(self.options.get("BottomDist", 0.0) or 0.0)
        self.top_margin = float(self.options.get("TopDist", 0.0) or 0.0)

        plate_minx, plate_miny, plate_maxx, plate_maxy = self.plate_poly.bounds
        self.usable_plate = box(
            plate_minx + self.left_margin,
            plate_miny + self.bottom_margin,
            plate_maxx - self.right_margin,
            plate_maxy - self.top_margin,
        )
        self.usable_bounds = self.usable_plate.bounds
        self.plate_width = self.usable_bounds[2] - self.usable_bounds[0]
        self.plate_height = self.usable_bounds[3] - self.usable_bounds[1]

        self.beam_width = max(1, int(beam_width))
        self.part_window = max(1, int(part_window))
        self.branch_factor = max(1, int(branch_factor))
        self.candidate_limit = max(20, int(candidate_limit))
        self.per_variant_keep = max(1, int(per_variant_keep))
        self.per_item_keep = max(1, int(per_item_keep))
        self.verbose = bool(verbose)

        self.templates: Dict[str, PartTemplate] = {}
        self.items: Tuple[Item, ...] = tuple()
        self.ifp_cache: Dict[str, object] = {}
        self.nfp_cache: Dict[Tuple[str, str], object] = {}

        self._build_templates_and_items()


    def _log(self, message: str):
        if self.verbose:
            print(message, flush=True)

    def _log_placement(self, phase: str, strategy: str, ordering_name: str,
                       state_before: SearchState, placement: Placement, total_items: int):
        if not self.verbose:
            return
        count_after = state_before.placed_count + 1
        minx, miny, maxx, maxy = placement.geom.bounds
        width = maxx - minx
        height = maxy - miny
        mode = []
        if placement.variant.mirrored:
            mode.append('mirrored')
        if abs(placement.variant.angle) > EPS:
            mode.append(f'rot={placement.variant.angle:g}')
        mode_text = f" ({', '.join(mode)})" if mode else ''
        self._log(
            f"[{phase}] order={ordering_name} strategy={strategy} "
            f"placed {count_after}/{total_items}: id={placement.item.template_id}#{placement.item.sequence}{mode_text} "
            f"at x={placement.tx:.2f}, y={placement.ty:.2f}, w={width:.2f}, h={height:.2f}"
        )

    def _build_templates_and_items(self):
        items: List[Item] = []
        for part in self.parts:
            geom = part_to_geometry(part)
            if geom.is_empty or geom.area <= EPS:
                continue

            part_id = str(part.get("id", "?"))
            quantity = int(part.get("quantity", 1) or 1)
            priority = int(float((part.get("partinfs", {}) or {}).get("priority", 50) or 50))
            variants = self._build_variants(part_id, geom, part)
            if not variants:
                continue
            minx, miny, maxx, maxy = geom.bounds
            width = maxx - minx
            height = maxy - miny
            bbox_area = max(width * height, EPS)
            bbox_fill = geom.area / bbox_area
            concavity = 1.0 - geom.area / max(geom.convex_hull.area, EPS)
            template = PartTemplate(
                part_id=part_id,
                priority=priority,
                quantity=quantity,
                base_geom=geom,
                variants=tuple(variants),
                nominal_area=geom.area,
                width=width,
                height=height,
                max_dim=max(width, height),
                bbox_fill=bbox_fill,
                concavity=concavity,
            )
            self.templates[part_id] = template
            for seq in range(1, quantity + 1):
                items.append(Item(template_id=part_id, sequence=seq))
        self.items = tuple(items)
        if self.templates:
            dominant = max(
                self.templates.values(),
                key=lambda t: (t.quantity * t.nominal_area, t.quantity, t.nominal_area, t.max_dim),
            )
            self.dominant_template_id = dominant.part_id
        else:
            self.dominant_template_id = ""

    def _build_variants(self, part_id: str, geom, part: dict) -> List[ShapeVariant]:
        info = part.get("partinfs", {}) or {}
        can_mirror = str(info.get("canmirror", "0")).strip().lower() in {"1", "true", "yes"}
        allow_rotation = str(info.get("rotation", "1")).strip().lower() not in {"0", "false", "no"}

        angle_step = float(self.options.get("ComAngle", 90.0) or 90.0)
        if angle_step <= 0:
            angle_step = 90.0
        if allow_rotation:
            count = max(1, int(round(360.0 / angle_step)))
            angles = [round(i * angle_step, 8) for i in range(count)]
        else:
            angles = [0.0]

        variants: List[ShapeVariant] = []
        seen = set()
        mirror_flags = [False, True] if can_mirror else [False]
        for mirrored in mirror_flags:
            base = scale(geom, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) if mirrored else geom
            for angle in angles:
                g = clean_geometry(rotate(base, angle, origin=(0.0, 0.0), use_radians=False))
                if g.is_empty or g.area <= EPS:
                    continue
                rounded_bounds = tuple(round(v, 5) for v in g.bounds)
                dedupe_key = (rounded_bounds, round(g.area, 5), mirrored, round(angle % 360.0, 8))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                gap_geom = g.buffer(self.gap / 2.0, quad_segs=DEFAULT_BUFFER_QUAD_SEGS) if self.gap > EPS else g
                gap_geom = clean_geometry(gap_geom)
                neg_gap_geom = clean_geometry(scale(gap_geom, xfact=-1.0, yfact=-1.0, origin=(0.0, 0.0)))
                minx, miny, maxx, maxy = g.bounds
                variants.append(
                    ShapeVariant(
                        key=f"{part_id}|a={angle:.8f}|m={int(mirrored)}",
                        part_id=part_id,
                        angle=angle,
                        mirrored=mirrored,
                        geom=g,
                        gap_geom=gap_geom,
                        neg_gap_geom=neg_gap_geom,
                        gap_triangles=geometry_triangles(gap_geom),
                        neg_gap_triangles=geometry_triangles(neg_gap_geom),
                        area=g.area,
                        bounds=g.bounds,
                        width=maxx - minx,
                        height=maxy - miny,
                        gap_convex=is_convex_polygonal(gap_geom),
                        neg_gap_convex=is_convex_polygonal(neg_gap_geom),
                    )
                )
        return variants

    def local_nfp(self, stationary: ShapeVariant, moving: ShapeVariant):
        key = (stationary.key, moving.key)
        if key in self.nfp_cache:
            return self.nfp_cache[key]

        if stationary.gap_convex and moving.neg_gap_convex:
            nfp = convex_minkowski_sum(stationary.gap_geom, moving.neg_gap_geom)
            self.nfp_cache[key] = nfp
            return nfp

        pieces: List[object] = []
        for tri_a in stationary.gap_triangles:
            for tri_b in moving.neg_gap_triangles:
                piece = convex_minkowski_sum(tri_a, tri_b)
                if not piece.is_empty and piece.area > EPS:
                    pieces.append(piece)
        nfp = clean_geometry(unary_union(pieces)) if pieces else ShapelyPolygon()
        self.nfp_cache[key] = nfp
        return nfp

    def inner_fit_region(self, variant: ShapeVariant):
        if variant.key in self.ifp_cache:
            return self.ifp_cache[variant.key]
        minx, miny, maxx, maxy = variant.bounds
        left = self.usable_bounds[0] - minx
        bottom = self.usable_bounds[1] - miny
        right = self.usable_bounds[2] - maxx
        top = self.usable_bounds[3] - maxy
        if left > right + EPS or bottom > top + EPS:
            region = ShapelyPolygon()
        else:
            region = box(left, bottom, right, top)
        self.ifp_cache[variant.key] = region
        return region

    def feasible_translation_region(self, state: SearchState, variant: ShapeVariant):
        ifp = self.inner_fit_region(variant)
        if ifp.is_empty:
            return ifp
        if not state.placements:
            return ifp
        forbidden: List[object] = []
        for placed in state.placements:
            local = self.local_nfp(placed.variant, variant)
            if local.is_empty:
                continue
            forbidden.append(translate(local, xoff=placed.tx, yoff=placed.ty))
        if not forbidden:
            return ifp
        forbidden_union = clean_geometry(unary_union(forbidden))
        feasible = clean_geometry(ifp.difference(forbidden_union))
        return feasible

    def extract_candidate_points(self, feasible_region) -> List[Tuple[float, float, float]]:
        points: Dict[Tuple[float, float], float] = {}
        for poly in iter_polygonal(feasible_region):
            component_area = poly.area
            rings = [poly.exterior, *poly.interiors]
            for ring in rings:
                for x, y in list(ring.coords)[:-1]:
                    key = (round(float(x), 6), round(float(y), 6))
                    if key not in points or component_area < points[key]:
                        points[key] = component_area
        ordered = sorted(
            [(x, y, points[(x, y)]) for x, y in points],
            key=lambda p: (round(p[1], 6), round(p[0], 6), p[2]),
        )
        return ordered[: self.candidate_limit]

    def candidate_is_valid(self, candidate_geom, state: SearchState) -> bool:
        if candidate_geom.is_empty:
            return False
        if not candidate_geom.covered_by(self.usable_plate):
            return False
        if self.gap <= EPS:
            return all(candidate_geom.intersection(p.geom).area <= 1e-6 for p in state.placements)
        return all(candidate_geom.distance(p.geom) >= self.gap - 1e-6 for p in state.placements)

    def placement_rank_key(self, placement: Placement, mode: str) -> Tuple[float, ...]:
        minx, miny, maxx, maxy = placement.geom.bounds
        if mode == "south_east":
            return (round(miny, 6), -round(maxx, 6), placement.score[1], placement.score[2], placement.score[3], placement.score[4])
        if mode == "north_west":
            return (-round(maxy, 6), round(minx, 6), placement.score[1], placement.score[2], placement.score[3], placement.score[4])
        if mode == "north_east":
            return (-round(maxy, 6), -round(maxx, 6), placement.score[1], placement.score[2], placement.score[3], placement.score[4])
        if mode == "min_waste":
            return (placement.score[1], placement.score[0], placement.score[2], placement.score[3], round(miny, 6), round(minx, 6))
        if mode == "max_contact":
            return (placement.score[4], placement.score[1], placement.score[0], placement.score[2], round(miny, 6), round(minx, 6))
        return (round(miny, 6), round(minx, 6), placement.score[1], placement.score[2], placement.score[3], placement.score[4])

    def select_diverse_placements(self, placements: Sequence[Placement], modes: Sequence[str],
                                  limit: int) -> List[Placement]:
        if not placements or limit <= 0:
            return []
        ordered_modes: List[str] = []
        for mode in [*modes, "south_west", "south_east", "north_west", "north_east", "min_waste", "max_contact"]:
            if mode not in ordered_modes:
                ordered_modes.append(mode)

        chosen: List[Placement] = []
        used = set()
        for mode in ordered_modes:
            best = min(placements, key=lambda p: self.placement_rank_key(p, mode))
            key = (best.variant.key, round(best.tx, 6), round(best.ty, 6))
            if key in used:
                continue
            used.add(key)
            chosen.append(best)
            if len(chosen) >= limit:
                return chosen

        primary_mode = ordered_modes[0] if ordered_modes else "south_west"
        for placement in sorted(placements, key=lambda p: self.placement_rank_key(p, primary_mode)):
            key = (placement.variant.key, round(placement.tx, 6), round(placement.ty, 6))
            if key in used:
                continue
            used.add(key)
            chosen.append(placement)
            if len(chosen) >= limit:
                break
        return chosen

    def primary_mode_for_item(self, item: Item, strategy: str, gap_fill_mode: bool = False) -> str:
        if gap_fill_mode:
            return "min_waste"
        if strategy == "dominant_sw_others_se":
            return "south_west" if item.template_id == self.dominant_template_id else "south_east"
        if strategy == "dominant_sw_others_ne":
            return "south_west" if item.template_id == self.dominant_template_id else "north_east"
        if strategy == "all_se":
            return "south_east"
        if strategy == "all_ne":
            return "north_east"
        if strategy == "all_nw":
            return "north_west"
        return "south_west"

    def candidate_modes_for_item(self, item: Item, strategy: str, gap_fill_mode: bool = False) -> Tuple[str, ...]:
        primary = self.primary_mode_for_item(item, strategy, gap_fill_mode)
        if gap_fill_mode:
            return (primary, "south_west", "south_east", "max_contact")
        if strategy == "dominant_sw_others_se" and item.template_id != self.dominant_template_id:
            return (primary, "south_west", "min_waste", "max_contact")
        return (primary, "south_east", "north_east", "min_waste", "max_contact")

    def score_candidate(self, state: SearchState, candidate_geom, candidate_gap_geom,
                        component_area: float, gap_fill_mode: bool = False) -> Tuple[float, ...]:
        _, _, maxx, maxy = candidate_geom.bounds
        new_used_maxx = max(state.used_maxx, maxx) if state.placements else maxx
        new_used_maxy = max(state.used_maxy, maxy) if state.placements else maxy
        used_width = max(0.0, new_used_maxx - self.usable_bounds[0])
        used_height = max(0.0, new_used_maxy - self.usable_bounds[1])
        bbox_area = used_width * used_height
        new_area = state.placed_area + candidate_geom.area
        waste = bbox_area - new_area

        plate_contact = candidate_geom.boundary.intersection(self.usable_plate.boundary).length
        part_contact = 0.0
        for placement in state.placements:
            if candidate_geom.distance(placement.geom) <= self.gap + 1e-5:
                part_contact += candidate_gap_geom.boundary.intersection(placement.gap_geom.boundary).length
        contact_bonus = plate_contact + part_contact

        # gap_fill_mode prioritizes filling the smallest remaining cavity first.
        if gap_fill_mode:
            return (
                component_area,
                waste,
                used_height,
                used_width,
                -contact_bonus,
                round(candidate_geom.bounds[1], 6),
                round(candidate_geom.bounds[0], 6),
            )
        return (
            used_height,
            waste,
            component_area,
            used_width,
            -contact_bonus,
            round(candidate_geom.bounds[1], 6),
            round(candidate_geom.bounds[0], 6),
        )

    def generate_candidate_placements(self, item: Item, state: SearchState,
                                      gap_fill_mode: bool = False,
                                      strategy: str = "all_sw") -> List[Placement]:
        template = self.templates[item.template_id]
        placements: List[Placement] = []
        seen_positions = set()
        modes = self.candidate_modes_for_item(item, strategy, gap_fill_mode)

        for variant in template.variants:
            if variant.width > self.plate_width + EPS or variant.height > self.plate_height + EPS:
                continue
            feasible = self.feasible_translation_region(state, variant)
            if feasible.is_empty:
                continue
            pts = self.extract_candidate_points(feasible)
            local_candidates: List[Placement] = []
            for tx, ty, component_area in pts:
                key = (variant.key, round(tx, 6), round(ty, 6))
                if key in seen_positions:
                    continue
                candidate_geom = translate(variant.geom, xoff=tx, yoff=ty)
                if not self.candidate_is_valid(candidate_geom, state):
                    continue
                candidate_gap = translate(variant.gap_geom, xoff=tx, yoff=ty)
                score = self.score_candidate(state, candidate_geom, candidate_gap, component_area, gap_fill_mode)
                local_candidates.append(
                    Placement(
                        item=item,
                        variant=variant,
                        tx=tx,
                        ty=ty,
                        geom=candidate_geom,
                        gap_geom=candidate_gap,
                        score=score,
                        component_area=component_area,
                    )
                )
                seen_positions.add(key)
            placements.extend(self.select_diverse_placements(local_candidates, modes, self.per_variant_keep))

        return self.select_diverse_placements(placements, modes, self.per_item_keep)

    def state_rank_key(self, state: SearchState) -> Tuple[float, ...]:
        used_width = max(0.0, state.used_maxx - self.usable_bounds[0]) if state.placements else 0.0
        used_height = max(0.0, state.used_maxy - self.usable_bounds[1]) if state.placements else 0.0
        waste = used_width * used_height - state.placed_area
        return (
            -state.placed_count,
            -state.placed_area,
            used_height,
            waste,
            used_width,
            len(state.remaining),
        )

    def place(self, state: SearchState, placement: Placement, remove_index: int) -> SearchState:
        _, _, maxx, maxy = placement.geom.bounds
        new_used_maxx = max(state.used_maxx, maxx) if state.placements else maxx
        new_used_maxy = max(state.used_maxy, maxy) if state.placements else maxy
        remaining = state.remaining[:remove_index] + state.remaining[remove_index + 1 :]
        return SearchState(
            placements=state.placements + (placement,),
            remaining=remaining,
            placed_area=state.placed_area + placement.geom.area,
            used_maxx=new_used_maxx,
            used_maxy=new_used_maxy,
        )

    def template_items_by_id(self) -> Dict[str, List[Item]]:
        grouped: Dict[str, List[Item]] = {template_id: [] for template_id in self.templates}
        for item in self.items:
            grouped[item.template_id].append(item)
        return grouped

    def grouped_template_order(self, template_ids: Sequence[str], items_by_template: Dict[str, List[Item]]) -> Tuple[Item, ...]:
        return tuple(item for template_id in template_ids for item in items_by_template.get(template_id, []))

    def round_robin_template_order(self, template_ids: Sequence[str], items_by_template: Dict[str, List[Item]]) -> Tuple[Item, ...]:
        order: List[Item] = []
        counters = {template_id: 0 for template_id in template_ids}
        while True:
            progress = False
            for template_id in template_ids:
                idx = counters[template_id]
                template_items = items_by_template.get(template_id, [])
                if idx >= len(template_items):
                    continue
                order.append(template_items[idx])
                counters[template_id] = idx + 1
                progress = True
            if not progress:
                break
        return tuple(order)

    def prefix_group_then_round_robin(self, prefix_ids: Sequence[str], suffix_ids: Sequence[str],
                                      items_by_template: Dict[str, List[Item]]) -> Tuple[Item, ...]:
        grouped_prefix = list(self.grouped_template_order(prefix_ids, items_by_template))
        suffix_round_robin = list(self.round_robin_template_order(suffix_ids, items_by_template))
        return tuple(grouped_prefix + suffix_round_robin)

    def initial_orderings(self) -> List[Tuple[str, Tuple[Item, ...]]]:
        items = list(self.items)
        items_by_template = self.template_items_by_id()

        def key_area(item: Item):
            t = self.templates[item.template_id]
            return (-t.priority, -t.nominal_area, -t.max_dim, len(t.variants), -t.concavity)

        def key_hard(item: Item):
            t = self.templates[item.template_id]
            return (-t.priority, len(t.variants), -t.max_dim, -t.concavity, t.bbox_fill, -t.nominal_area)

        def key_wide(item: Item):
            t = self.templates[item.template_id]
            return (-t.priority, -t.width, -t.height, -t.nominal_area, len(t.variants))

        def key_tall(item: Item):
            t = self.templates[item.template_id]
            return (-t.priority, -t.height, -t.width, -t.nominal_area, len(t.variants))

        def key_template_easy(template_id: str):
            t = self.templates[template_id]
            return (-t.priority, -t.bbox_fill, t.concavity, len(t.variants), -t.max_dim, -t.nominal_area)

        def key_template_hard(template_id: str):
            t = self.templates[template_id]
            return (-t.priority, -t.concavity, -len(t.variants), -t.max_dim, t.bbox_fill, -t.nominal_area)

        def key_template_area_desc(template_id: str):
            t = self.templates[template_id]
            return (-t.priority, -t.nominal_area, -t.max_dim, -t.quantity)

        def key_template_area_asc(template_id: str):
            t = self.templates[template_id]
            return (-t.priority, t.nominal_area, t.max_dim, -t.quantity)

        raw = tuple(items)
        by_hard = tuple(sorted(items, key=key_hard))
        by_area = tuple(sorted(items, key=key_area))
        by_wide = tuple(sorted(items, key=key_wide))
        by_tall = tuple(sorted(items, key=key_tall))

        template_easy = sorted(self.templates, key=key_template_easy)
        template_hard = sorted(self.templates, key=key_template_hard)
        template_area_desc = sorted(self.templates, key=key_template_area_desc)
        template_area_asc = sorted(self.templates, key=key_template_area_asc)
        prefix_count = max(1, (len(template_easy) + 1) // 2)
        easy_prefix = template_easy[:prefix_count]
        suffix_area_desc = [template_id for template_id in template_area_desc if template_id not in easy_prefix]
        suffix_area_asc = [template_id for template_id in template_area_asc if template_id not in easy_prefix]

        candidates: List[Tuple[str, Tuple[Item, ...]]] = [
            ('easy_prefix_rr_area_desc', self.prefix_group_then_round_robin(easy_prefix, suffix_area_desc, items_by_template)),
            ('easy_prefix_rr_area_asc', self.prefix_group_then_round_robin(easy_prefix, suffix_area_asc, items_by_template)),
            ('raw', raw),
            ('by_hard', by_hard),
            ('by_area', by_area),
            ('grouped_easy', self.grouped_template_order(template_easy, items_by_template)),
            ('rr_easy', self.round_robin_template_order(template_easy, items_by_template)),
            ('grouped_hard', self.grouped_template_order(template_hard, items_by_template)),
            ('by_wide', by_wide),
            ('by_tall', by_tall),
        ]

        unique: List[Tuple[str, Tuple[Item, ...]]] = []
        seen = set()
        for name, order in candidates:
            sig = tuple((item.template_id, item.sequence) for item in order)
            if sig in seen:
                continue
            seen.add(sig)
            unique.append((name, order))
        return unique

    def beam_search_pack(self, ordered_items: Tuple[Item, ...], ordering_name: str = "beam") -> SearchState:
        states = [SearchState(placements=tuple(), remaining=ordered_items, placed_area=0.0, used_maxx=0.0, used_maxy=0.0)]
        best = states[0]

        while states:
            next_states: List[SearchState] = []
            progress_made = False
            self._log(f"[beam] order={ordering_name} frontier={len(states)} best={best.placed_count}/{len(ordered_items)}")
            for state in states:
                if self.state_rank_key(state) < self.state_rank_key(best):
                    best = state
                if not state.remaining:
                    next_states.append(state)
                    continue

                choices: List[Tuple[int, Item]] = []
                seen_templates = set()
                for idx, item in enumerate(state.remaining):
                    if item.template_id in seen_templates:
                        continue
                    seen_templates.add(item.template_id)
                    choices.append((idx, item))
                    if len(choices) >= self.part_window:
                        break

                expanded = False
                for idx, item in choices:
                    candidates = self.generate_candidate_placements(item, state)
                    if not candidates:
                        continue
                    expanded = True
                    progress_made = True
                    for placement in candidates[: self.branch_factor]:
                        next_states.append(self.place(state, placement, idx))

                # Limited backtracking: if the front window dead-ends, look deeper in remaining items.
                if not expanded:
                    for idx, item in enumerate(state.remaining[len(choices):], start=len(choices)):
                        candidates = self.generate_candidate_placements(item, state)
                        if not candidates:
                            continue
                        expanded = True
                        progress_made = True
                        for placement in candidates[: max(1, self.branch_factor - 1)]:
                            next_states.append(self.place(state, placement, idx))
                        break

                if not expanded:
                    next_states.append(state)

            if not next_states:
                break

            next_states.sort(key=self.state_rank_key)
            states = next_states[: self.beam_width]
            if not progress_made:
                break

        states.sort(key=self.state_rank_key)
        return states[0] if states else best

    def guided_pack(self, ordered_items: Tuple[Item, ...], strategy: str = "all_sw",
                    ordering_name: str = "guided") -> SearchState:
        state = SearchState(placements=tuple(), remaining=ordered_items, placed_area=0.0, used_maxx=0.0, used_maxy=0.0)
        total_items = len(ordered_items)
        while state.remaining:
            placed_any = False
            for idx, item in enumerate(state.remaining):
                candidates = self.generate_candidate_placements(item, state, strategy=strategy)
                if not candidates:
                    continue
                placement = candidates[0]
                self._log_placement('guided', strategy, ordering_name, state, placement, total_items)
                state = self.place(state, placement, idx)
                placed_any = True
                break
            if not placed_any:
                self._log(f"[guided] order={ordering_name} strategy={strategy} stalled after placing {state.placed_count}/{total_items}")
                break
        return state

    def gap_fill_pass(self, state: SearchState, strategy: str = "all_sw",
                      ordering_name: str = "gap_fill") -> SearchState:
        remaining = list(state.remaining)
        remaining.sort(key=lambda item: (self.templates[item.template_id].nominal_area, self.templates[item.template_id].max_dim))
        current = state
        total_items = len(state.placements) + len(state.remaining)
        for item in list(remaining):
            idx = next((i for i, it in enumerate(current.remaining) if it == item), None)
            if idx is None:
                continue
            candidates = self.generate_candidate_placements(item, current, gap_fill_mode=True, strategy=strategy)
            if not candidates:
                continue
            placement = candidates[0]
            self._log_placement('gap-fill', strategy, ordering_name, current, placement, total_items)
            current = self.place(current, placement, idx)
        return current

    def nest(self) -> SearchState:
        best_state = SearchState(placements=tuple(), remaining=self.items, placed_area=0.0, used_maxx=0.0, used_maxy=0.0)
        target_count = len(self.items)
        search_plan: List[Tuple[str, Sequence[str], Tuple[Item, ...]]] = []
        orderings = self.initial_orderings()
        ordering_map = {name: order for name, order in orderings}

        if target_count >= 80:
            preferred_order_names = [
                'easy_prefix_rr_area_desc',
                'easy_prefix_rr_area_asc',
                'grouped_easy',
                'rr_easy',
                'raw',
                'by_hard',
                'by_area',
                'by_wide',
                'by_tall',
            ]
            strategy_map = {
                'easy_prefix_rr_area_desc': ('all_nw', 'all_ne', 'all_se'),
                'easy_prefix_rr_area_asc': ('all_nw', 'all_ne', 'all_se'),
                'grouped_easy': ('all_nw', 'all_sw'),
                'rr_easy': ('all_nw', 'all_ne', 'all_sw'),
                'raw': ('all_sw', 'all_se', 'dominant_sw_others_ne'),
                'by_hard': ('all_sw', 'all_se', 'all_nw'),
                'by_area': ('all_se', 'all_sw', 'all_nw'),
                'by_wide': ('all_se', 'all_sw'),
                'by_tall': ('all_nw', 'all_ne'),
            }
        else:
            preferred_order_names = [
                'raw',
                'by_hard',
                'by_area',
                'by_wide',
                'by_tall',
            ]
            strategy_map = {
                'raw': ('dominant_sw_others_se', 'all_sw', 'all_se', 'dominant_sw_others_ne', 'all_nw'),
                'by_hard': ('dominant_sw_others_se', 'all_sw', 'all_se', 'dominant_sw_others_ne', 'all_nw'),
                'by_area': ('dominant_sw_others_se', 'all_sw', 'all_se', 'dominant_sw_others_ne', 'all_nw'),
                'by_wide': ('all_sw', 'all_se', 'all_nw'),
                'by_tall': ('all_sw', 'all_se', 'all_nw'),
            }
        for name in preferred_order_names:
            if name not in ordering_map:
                continue
            search_plan.append((name, strategy_map.get(name, ('all_sw', 'all_se')), ordering_map[name]))

        # Fast multi-start greedy passes. The template-level hybrid orderings are tried first because they
        # preserve strip-friendly parts as blocks while interleaving complementary large/small templates later.
        for ordering_name, strategies, order in search_plan:
            for strategy in strategies:
                self._log(f"[run] order={ordering_name} strategy={strategy} starting")
                guided = self.guided_pack(order, strategy=strategy, ordering_name=ordering_name)
                guided = self.gap_fill_pass(guided, strategy=strategy, ordering_name=ordering_name)
                self._log(
                    f"[run] order={ordering_name} strategy={strategy} done: "
                    f"placed {guided.placed_count}/{target_count}, remaining {len(guided.remaining)}"
                )
                if self.state_rank_key(guided) < self.state_rank_key(best_state):
                    best_state = guided
                    self._log(f"[best] order={ordering_name} strategy={strategy} improved best to {best_state.placed_count}/{target_count}")
                if guided.placed_count >= target_count:
                    return guided
                # On very dense, high-part-count jobs the hybrid ordering usually dominates quickly.
                # For smaller jobs we keep searching because a later strategy can still hit a full placement.
                if target_count >= 80 and best_state.placed_count >= max(1, math.ceil(target_count * 0.90)) and ordering_name.startswith('easy_prefix_rr_area'):
                    return best_state

        # The guided passes are the main workhorse. When they are already within one part of full nesting,
        # returning immediately avoids an expensive beam step that rarely improves the result.
        if best_state.placed_count >= max(0, target_count - 1):
            return best_state

        # Fallback beam search for harder cases when the guided passes leave a meaningful gap.
        for ordering_name, _, order in search_plan[: min(2, len(search_plan))]:
            beam_state = self.beam_search_pack(order, ordering_name=ordering_name)
            beam_state = self.gap_fill_pass(beam_state, strategy='all_sw', ordering_name=f'{ordering_name}:beam')
            if self.state_rank_key(beam_state) < self.state_rank_key(best_state):
                best_state = beam_state
                self._log(f"[best] order={ordering_name} beam improved best to {best_state.placed_count}/{target_count}")
            if beam_state.placed_count >= target_count:
                return beam_state
        return best_state


# ----------------------------
# Reporting / plotting
# ----------------------------
def draw_geometry_outline(ax, geom, color: str = "b", linestyle: str = "-"):
    for poly in iter_polygonal(geom):
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, linestyle=linestyle)
        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.plot(hx, hy, color=color, linestyle="--")


def plot_arrangement(placements: Sequence[Placement], plate_poly, title: str = "Layout"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect("equal")
    draw_geometry_outline(ax, plate_poly, color="black")
    for placement in placements:
        draw_geometry_outline(ax, placement.geom, color="b")
        rp = placement.geom.representative_point()
        label = placement.item.template_id
        if placement.variant.mirrored:
            label += "M"
        if abs(placement.variant.angle) > EPS:
            label += f"\n{placement.variant.angle:g}°"
        ax.text(rp.x, rp.y, label, ha="center", va="center", fontsize=8, color="green")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    plt.show()


def summarize_result(final_state: SearchState):
    placed_counts: Dict[str, int] = {}
    for placement in final_state.placements:
        placed_counts[placement.item.template_id] = placed_counts.get(placement.item.template_id, 0) + 1
    remaining_counts: Dict[str, int] = {}
    for item in final_state.remaining:
        remaining_counts[item.template_id] = remaining_counts.get(item.template_id, 0) + 1

    print(f"{'ID':<8} {'Placed':<8} {'Unplaced':<8}")
    all_ids = sorted(set(placed_counts) | set(remaining_counts), key=lambda s: (len(s), s))
    for pid in all_ids:
        print(f"{pid:<8} {placed_counts.get(pid, 0):<8} {remaining_counts.get(pid, 0):<8}")
    print(f"\nTotal placed parts: {len(final_state.placements)}")
    print(f"Total unplaced parts: {len(final_state.remaining)}")


def run_nesting(data: dict, width: float, height: float, plot: bool = True,
                beam_width: int = 2, part_window: int = 2, branch_factor: int = 2,
                verbose: bool = True):
    plate_poly = box(0.0, 0.0, float(width), float(height))
    engine = PolygonNestingEngine(
        parts=data.get("parts", []),
        plate_poly=plate_poly,
        options=data.get("options", {}),
        beam_width=beam_width,
        part_window=part_window,
        branch_factor=branch_factor,
        verbose=verbose,
    )
    state = engine.nest()
    summarize_result(state)
    if plot:
        plot_arrangement(
            state.placements,
            plate_poly,
            title="Advanced polygon nesting (true NFP + beam search + hybrid ordering + gap filling)",
        )
    return state


# ----------------------------
# GUI
# ----------------------------
class NestingApp:
    def __init__(self, root):
        if tk is None:
            raise RuntimeError("Tkinter is not available in this environment")
        self.root = root
        self.root.title("2D Polygon Nesting - true NFP + backtracking + hybrid ordering")

        tk.Label(root, text="Layout Width:").grid(row=0, column=0, sticky="w")
        tk.Label(root, text="Layout Height:").grid(row=1, column=0, sticky="w")
        tk.Label(root, text="Beam Width:").grid(row=2, column=0, sticky="w")
        tk.Label(root, text="Part Window:").grid(row=3, column=0, sticky="w")
        tk.Label(root, text="Branch Factor:").grid(row=4, column=0, sticky="w")

        self.width_entry = tk.Entry(root)
        self.height_entry = tk.Entry(root)
        self.beam_entry = tk.Entry(root)
        self.window_entry = tk.Entry(root)
        self.branch_entry = tk.Entry(root)

        self.width_entry.grid(row=0, column=1)
        self.height_entry.grid(row=1, column=1)
        self.beam_entry.grid(row=2, column=1)
        self.window_entry.grid(row=3, column=1)
        self.branch_entry.grid(row=4, column=1)

        self.beam_entry.insert(0, "2")
        self.window_entry.insert(0, "2")
        self.branch_entry.insert(0, "2")

        self.load_button = tk.Button(root, text="Load JSON", command=self.load_json_file)
        self.load_button.grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")

        self.run_button = tk.Button(root, text="Run Nesting", command=self.run_nesting)
        self.run_button.grid(row=6, column=0, columnspan=2, pady=5, sticky="ew")

        self.data: Optional[dict] = None

    def load_json_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            self.data = load_json(file_path)
            parts = self.data.get("parts", [])
            messagebox.showinfo("Success", f"Loaded {len(parts)} part definitions")

    def run_nesting(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load a JSON file first")
            return
        try:
            width = float(self.width_entry.get())
            height = float(self.height_entry.get())
            beam_width = int(self.beam_entry.get())
            part_window = int(self.window_entry.get())
            branch_factor = int(self.branch_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
            return

        state = run_nesting(
            self.data,
            width=width,
            height=height,
            plot=True,
            beam_width=beam_width,
            part_window=part_window,
            branch_factor=branch_factor,
            verbose=False,
        )
        messagebox.showinfo(
            "Done",
            f"Placed {len(state.placements)} parts, unplaced {len(state.remaining)} parts.",
        )


# ----------------------------
# CLI entrypoint
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Advanced 2D polygon nesting with true NFP, beam-search backtracking, hybrid template ordering, and progress logging")
    parser.add_argument("--input", help="Path to input JSON file")
    parser.add_argument("--width", type=float, help="Plate width")
    parser.add_argument("--height", type=float, help="Plate height")
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib plotting")
    parser.add_argument("--beam-width", type=int, default=2, help="Beam width for backtracking search")
    parser.add_argument("--part-window", type=int, default=2, help="How many distinct parts to consider before branching")
    parser.add_argument("--branch-factor", type=int, default=2, help="How many candidate placements to keep per state")
    parser.add_argument("--quiet", action="store_true", help="Disable per-placement progress logs")
    args = parser.parse_args()

    if args.input and args.width is not None and args.height is not None:
        data = load_json(args.input)
        run_nesting(
            data,
            width=args.width,
            height=args.height,
            plot=not args.no_plot,
            beam_width=args.beam_width,
            part_window=args.part_window,
            branch_factor=args.branch_factor,
            verbose=not args.quiet,
        )
        return

    if tk is None:
        raise RuntimeError("Tkinter is unavailable. Use CLI mode with --input/--width/--height.")

    root = tk.Tk()
    app = NestingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
