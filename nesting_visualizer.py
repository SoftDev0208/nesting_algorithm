import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon, box
from shapely.affinity import translate, rotate
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# Load JSON
def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)

# Convert part to shapely polygon
# Circle -> approximate as polygon
# Rect / poly -> use points
def part_to_polygon(part):
    pts = part['contours'][0]['points']
    if len(pts) == 1 and pts[0].get('radius',0) > 0:
        angle = np.linspace(0, 2*np.pi, 30)
        r = pts[0]['radius']
        cx, cy = pts[0]['x'], pts[0]['y']
        return ShapelyPolygon([(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angle])
    else:
        return ShapelyPolygon([(p['x'], p['y']) for p in pts])

# Calculate bounding box width and height
def poly_size(poly):
    minx, miny, maxx, maxy = poly.bounds
    return maxx-minx, maxy-miny

# Nesting with quantities, no overlaps, full-fit validation
def nest_parts_with_full_fit(parts, plate_poly):
    parts_list = []
    for part in parts:
        qty = part.get('quantity', 1)
        for _ in range(qty):
            parts_list.append((part['id'], part_to_polygon(part)))

    # Sort by area descending
    parts_list.sort(key=lambda x: x[1].area, reverse=True)

    placed_parts = []
    occupied = []
    plate_minx, plate_miny, plate_maxx, plate_maxy = plate_poly.bounds

    for pid, poly in parts_list:
        placed = False
        for angle in [0, 90]:
            rotated = rotate(poly, angle, origin=(0,0), use_radians=False)
            rw, rh = poly_size(rotated)
            x_positions = np.arange(plate_minx, plate_maxx - rw, 50)
            y_positions = np.arange(plate_miny, plate_maxy - rh, 50)
            for x in x_positions:
                for y in y_positions:
                    candidate = translate(rotated, xoff=x, yoff=y)
                    # Full-fit validation: candidate must be completely inside plate
                    if not plate_poly.contains(candidate):
                        continue
                    overlap = any(candidate.intersects(o) for o in occupied)
                    if not overlap:
                        placed_parts.append({'id': pid, 'poly': candidate, 'x': x, 'y': y, 'angle': angle})
                        occupied.append(candidate)
                        print(f"Placed part {pid} at x={x}, y={y}, angle={angle}")
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            print(f"Could not place part {pid}, not enough space")

    print(f"Total placed parts: {len(placed_parts)}")
    return placed_parts

# Plot arrangement
def plot_arrangement(placed_parts, plate_poly):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_aspect('equal')
    xs, ys = plate_poly.exterior.xy
    ax.plot(xs, ys, 'black')
    for p in placed_parts:
        xs, ys = p['poly'].exterior.xy
        ax.plot(xs, ys, 'b')
        cx = np.mean(xs)
        cy = np.mean(ys)
        ax.text(cx, cy, str(p['id']), ha='center', va='center', fontsize=8, color='green')
    ax.set_title('Layout Filled According to Quantities - Full Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Generate summary table
def generate_table(placed_parts):
    summary = {}
    for p in placed_parts:
        summary[p['id']] = summary.get(p['id'],0)+1
    print(f"{'ID':<5} {'Placed Count':<12}")
    for pid, count in summary.items():
        print(f"{pid:<5} {count:<12}")

# GUI
class NestingApp:
    def __init__(self, root):
        self.root = root
        self.root.title('2D Nesting with Full-Fit Validation')

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
        file_path = filedialog.askopenfilename(filetypes=[('JSON Files', '*.json')])
        if file_path:
            self.parts = load_json(file_path)['parts']
            messagebox.showinfo("Success", f"Loaded {len(self.parts)} parts")

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
        self.plate_poly = box(0,0,w,h)
        placed_parts = nest_parts_with_full_fit(self.parts, self.plate_poly)
        plot_arrangement(placed_parts, self.plate_poly)
        generate_table(placed_parts)

if __name__ == '__main__':
    root = tk.Tk()
    app = NestingApp(root)
    root.mainloop()
