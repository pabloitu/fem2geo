"""Build topography surfaces in the BEM model frame.

Produces two surfaces, both cropped to BBOX and reprojected to the same
CRS and units as the fault meshes:

- ``topo_dem.vtp``   geometry and ``elevation`` scalar from the DEM
- ``topo_rgb.vtp``   geometry from the image grid, elevation sampled
                     from the DEM, colour in a uint8 ``rgb`` array

The image is the finer of the two grids here (0.004078 deg against
0.004167), so it drives the basemap geometry and elevation is sampled
onto it. The DEM surface is kept separate because it covers a wider
extent and is the one to use where colour is not wanted.

In ParaView, colour ``topo_rgb`` by ``rgb`` and switch "Map Scalars"
off, otherwise the triplet is pushed through a colour map and the
result is meaningless.

Vertical exaggeration is left at 1. Baking it in here means the
surfaces no longer sit correctly against the fault meshes unless those
are scaled identically; a Transform filter in ParaView with scale
(1, 1, VE) applied to every dataset is the safer route.
"""

import numpy as np
import pyvista as pv
import rasterio
from rasterio.windows import from_bounds
from pyproj import CRS, Transformer

from fem2geo.internal.io import quad_faces

DEM = "merged.tif"
RGB = "basemap_1.tiff"
OUT_DEM = "topo_dem.vtp"
OUT_RGB = "topo_rgb.vtp"

SRC_CRS = "epsg:4326"
OUT_CRS = "epsg:32718"

# lon/lat crop. Latitudes are sorted, so either order works.
BBOX_LON = (-77.0, -69.0)
BBOX_LAT = (-36, -49.0)

DEM_BAND = 1
RGB_BANDS = (1, 2, 3)

# "m" matches the fault meshes written by build_lofs.py. Switching to
# "km" here means scaling those by 1/1000 as well, or nothing lines up.
UNITS = "m"

# Cap on the larger mesh dimension. The bbox crop already brings this
# to roughly 736 x 2575, so decimation is off unless the crop grows.
TARGET_DIM = 4000

EXAGGERATION = 1.0

UNIT_SCALE = {"m": 1.0, "km": 1e-3}[UNITS]
LON = tuple(sorted(BBOX_LON))
LAT = tuple(sorted(BBOX_LAT))


def read_window(path, bands, target_dim):
    """Read a cropped, optionally decimated block plus its transform.

    Returns ``(data, transform, width, height)`` with data shaped
    ``(n_bands, h, w)`` and NaN where the source was masked.
    """
    with rasterio.open(path) as ds:
        if ds.count < max(bands):
            raise ValueError(f"{path} has {ds.count} band(s), need {max(bands)}")
        win = from_bounds(LON[0], LAT[0], LON[1], LAT[1], ds.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        if win.width < 2 or win.height < 2:
            raise ValueError(f"{path}: bbox leaves {win.width}x{win.height} pixels")

        step = max(1, int(np.ceil(max(win.width, win.height) / target_dim)))
        w, h = int(win.width) // step, int(win.height) // step
        data = ds.read(list(bands), window=win, out_shape=(len(bands), h, w),
                       masked=True)
        data = np.ma.filled(data.astype(float), np.nan)
        wt = rasterio.windows.transform(win, ds.transform)
        tfm = wt * wt.scale(win.width / w, win.height / h)
        print(f"{path}: {ds.width}x{ds.height} -> window {int(win.width)}x"
              f"{int(win.height)} -> {w}x{h} (step {step})")
    return data, tfm, w, h


def pixel_centres(tfm, w, h):
    """Pixel-centre coordinates in raster row order (north first)."""
    cc, rr = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    xs = tfm.a * cc + tfm.b * rr + tfm.c
    ys = tfm.d * cc + tfm.e * rr + tfm.f
    return xs, ys


def sample_grid(values, tfm, w, h, x, y):
    """Nearest-neighbour lookup into a raster block at map coordinates.

    ``values`` must be in raster row order, matching ``tfm``. Points
    outside the block come back NaN. Pixel centres sit at half-integer
    indices, hence the 0.5 shift before rounding.
    """
    inv = ~tfm
    cols = inv.a * x + inv.b * y + inv.c
    rows = inv.d * x + inv.e * y + inv.f
    outside = (cols < 0) | (cols > w) | (rows < 0) | (rows > h)
    ci = np.clip(np.round(cols - 0.5).astype(int), 0, w - 1)
    ri = np.clip(np.round(rows - 0.5).astype(int), 0, h - 1)
    out = values[ri, ci].astype(float)
    out[outside] = np.nan
    return out


def build_surface(xs, ys, z, w):
    """Triangulate a grid, dropping quads that touch a NaN elevation.

    Rows are flipped here so j increases northward, matching VTK
    winding. Sampling must happen before this, in raster row order.
    """
    xs, ys, z = xs[::-1], ys[::-1], z[::-1]
    valid = np.isfinite(z)
    faces = quad_faces(valid, w)
    if faces.size == 0:
        raise ValueError("no valid faces; is the elevation all masked?")
    pts = np.c_[xs.ravel(), ys.ravel(), np.where(valid, z, 0.0).ravel()]
    surf = pv.PolyData(pts, faces)
    surf.point_data["valid"] = valid.astype(np.uint8).ravel()
    return surf


def to_model_frame(surf):
    """Reproject in place into the model CRS, units and exaggeration."""
    pts = np.asarray(surf.points, dtype=float)
    if CRS.from_user_input(SRC_CRS) != CRS.from_user_input(OUT_CRS):
        t = Transformer.from_crs(SRC_CRS, OUT_CRS, always_xy=True)
        x, y = t.transform(pts[:, 0], pts[:, 1])
    else:
        x, y = pts[:, 0], pts[:, 1]
    surf.points = np.c_[
        x * UNIT_SCALE, y * UNIT_SCALE, pts[:, 2] * UNIT_SCALE * EXAGGERATION
    ]
    return surf


def report(name, surf):
    b = surf.bounds
    print(f"wrote {name}: {surf.n_points} points, {surf.n_cells} triangles")
    print(f"  x: {b[0]:.0f} .. {b[1]:.0f}")
    print(f"  y: {b[2]:.0f} .. {b[3]:.0f}")
    print(f"  z: {b[4]:.0f} .. {b[5]:.0f}")


# DEM surface

dem, dem_tfm, dw, dh = read_window(DEM, (DEM_BAND,), TARGET_DIM)
dem_z = dem[0]
dem_x, dem_y = pixel_centres(dem_tfm, dw, dh)

topo = build_surface(dem_x, dem_y, dem_z, dw)
topo.point_data["elevation"] = dem_z[::-1].ravel()
to_model_frame(topo)
topo.save(OUT_DEM)
report(OUT_DEM, topo)

# RGB basemap, elevation sampled from the DEM

img, img_tfm, iw, ih = read_window(RGB, RGB_BANDS, TARGET_DIM)
img_x, img_y = pixel_centres(img_tfm, iw, ih)

base_z = sample_grid(dem_z, dem_tfm, dw, dh, img_x, img_y)
n_out = int(np.isnan(base_z).sum())
if n_out:
    print(f"  {n_out} image pixels fall outside the DEM, dropped")

base = build_surface(img_x, img_y, base_z, iw)

rgb = img[:, ::-1].reshape(len(RGB_BANDS), -1).T
if not np.all(np.isfinite(rgb)):
    rgb = np.nan_to_num(rgb, nan=128.0)
if rgb.max() > 255 or rgb.min() < 0:
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255)
base.point_data["rgb"] = rgb.astype(np.uint8)
base.point_data["elevation"] = base_z[::-1].ravel()

to_model_frame(base)
base.save(OUT_RGB)
report(OUT_RGB, base)