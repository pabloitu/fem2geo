import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyproj
import pyvista as pv
import rasterio
from rasterio.transform import from_origin

from fem2geo.jobs.project import run


def _utm_xy(lon, lat):
    tfm = pyproj.Transformer.from_crs("epsg:4326", "epsg:32719", always_xy=True)
    return tfm.transform(lon, lat)


class _JobTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_csv(self, name, text):
        p = self.tmp / name
        p.write_text(text)
        return p

    def write_mesh(self, name, mesh):
        p = self.tmp / name
        mesh.save(str(p))
        return p


class TestCatalogJob(_JobTestBase):

    def _basic_cfg(self):
        return {
            "job": "project",
            "data": {
                "file": "cat.csv",
                "catalog": {"columns": ["longitude", "latitude", "depth"]},
            },
            "src": {
                "crs": "epsg:4326",
                "xy_units": "deg",
                "z_units": "km",
                "z_positive": "down",
            },
            "dst": {
                "units": "km",
                "anchor": {
                    "data": {"lon": -71.07, "lat": -20.09, "depth_km": 15.6},
                    "model": [0, 0, -21],
                    "rotation_deg": -10,
                },
            },
            "output": {"dir": str(self.tmp), "file": "out.vtp"},
        }

    def test_anchor_lands_at_origin(self):
        self.write_csv(
            "cat.csv",
            "longitude,latitude,depth,mag\n"
            "-71.07,-20.09,15.6,5.0\n",
        )
        run(self._basic_cfg(), self.tmp)
        out = pv.read(str(self.tmp / "out.vtp"))
        self.assertEqual(out.n_points, 1)
        np.testing.assert_allclose(out.points[0], [0.0, 0.0, -21.0], atol=1e-9)

    def test_attrs_carried_to_polydata(self):
        self.write_csv(
            "cat.csv",
            "longitude,latitude,depth,mag,rms\n"
            "-71.07,-20.09,15.6,5.0,0.3\n"
            "-72.00,-20.50,30.0,4.5,0.4\n",
        )
        run(self._basic_cfg(), self.tmp)
        out = pv.read(str(self.tmp / "out.vtp"))
        self.assertIn("mag", out.point_data)
        self.assertIn("rms", out.point_data)
        np.testing.assert_array_equal(out.point_data["mag"], [5.0, 4.5])

    def test_non_numeric_columns_dropped(self):
        self.write_csv(
            "cat.csv",
            "longitude,latitude,depth,mag,magType\n"
            "-71.07,-20.09,15.6,5.0,Mw\n",
        )
        run(self._basic_cfg(), self.tmp)
        out = pv.read(str(self.tmp / "out.vtp"))
        self.assertIn("mag", out.point_data)
        self.assertNotIn("magType", out.point_data)

    def test_bbox_filters_points(self):
        self.write_csv(
            "cat.csv",
            "longitude,latitude,depth,mag\n"
            "-71.07,-20.09,15.6,5.0\n"
            "-72.00,-20.50,30.0,4.5\n"
            "-65.00,-15.00,5.0,3.0\n"
            "-60.00,-10.00,5.0,3.0\n",
        )
        cfg = self._basic_cfg()
        cfg["src"]["bbox"] = {
            "lon": [-75, -67], "lat": [-22, -19], "depth_km": [0, 60],
        }
        run(cfg, self.tmp)
        out = pv.read(str(self.tmp / "out.vtp"))
        self.assertEqual(out.n_points, 2)

    def test_bbox_empty_result_raises(self):
        self.write_csv(
            "cat.csv",
            "longitude,latitude,depth,mag\n-71,-20,5,3.0\n",
        )
        cfg = self._basic_cfg()
        cfg["src"]["bbox"] = {"lon": [10, 20]}
        with self.assertRaisesRegex(ValueError, "No points left"):
            run(cfg, self.tmp)

    def test_missing_columns_raises(self):
        self.write_csv("cat.csv", "longitude,latitude,depth\n-71,-20,5\n")
        cfg = self._basic_cfg()
        cfg["data"]["catalog"].pop("columns")
        with self.assertRaisesRegex(ValueError, "columns"):
            run(cfg, self.tmp)

    def test_missing_catalog_block_raises(self):
        self.write_csv("cat.csv", "longitude,latitude,depth\n-71,-20,5\n")
        cfg = self._basic_cfg()
        cfg["data"].pop("catalog")
        with self.assertRaisesRegex(ValueError, "columns"):
            run(cfg, self.tmp)

    def test_rotation_default(self):
        self.write_csv(
            "cat.csv",
            "longitude,latitude,depth\n"
            "-71.07,-20.09,15.6\n"
            "-70.07,-20.09,15.6\n",
        )
        cfg = self._basic_cfg()
        cfg["dst"]["anchor"].pop("rotation_deg")
        run(cfg, self.tmp)
        no_rot = pv.read(str(self.tmp / "out.vtp")).points.copy()

        cfg_rot = self._basic_cfg()
        cfg_rot["dst"]["anchor"]["rotation_deg"] = 90.0
        run(cfg_rot, self.tmp)
        with_rot = pv.read(str(self.tmp / "out.vtp")).points.copy()

        # 90° rotation around the anchor at origin: (x, y) -> (-y, x)
        np.testing.assert_allclose(with_rot[1, 0], -no_rot[1, 1], atol=1e-6)
        np.testing.assert_allclose(with_rot[1, 1],  no_rot[1, 0], atol=1e-6)


class TestCatalogJobAnchorXY(_JobTestBase):
    """Catalog job with the anchor given in projected coordinates."""

    def test_anchor_x_y_in_utm(self):
        # UTM 19S anchor for (-71.07, -20.09)
        ax, ay = _utm_xy(-71.07, -20.09)

        # Build a catalog in UTM meters with depth in km down
        import pyproj
        tfm = pyproj.Transformer.from_crs("epsg:4326", "epsg:32719", always_xy=True)
        x0, y0 = tfm.transform(-71.07, -20.09)
        p = self.tmp / "cat.csv"
        p.write_text(
            "x,y,depth\n"
            f"{x0},{y0},15.6\n"
        )

        cfg = {
            "job": "project",
            "data": {
                "file": "cat.csv",
                "catalog": {"columns": ["x", "y", "depth"]},
            },
            "src": {
                "crs": "epsg:32719",
                "xy_units": "m",
                "z_units": "km",
                "z_positive": "down",
            },
            "dst": {
                "units": "km",
                "anchor": {
                    "data": {"x": ax, "y": ay, "depth_km": 15.6},
                    "model": [0, 0, -21],
                },
            },
            "output": {"dir": str(self.tmp), "file": "out.vtp"},
        }
        run(cfg, self.tmp)
        out = pv.read(str(self.tmp / "out.vtp"))
        self.assertEqual(out.n_points, 1)
        np.testing.assert_allclose(out.points[0], [0.0, 0.0, -21.0], atol=1e-4)

    def test_xy_with_geographic_src_rejected(self):
        self.write_csv("cat.csv", "lon,lat,depth\n-71,-20,5\n")
        cfg = {
            "job": "project",
            "data": {
                "file": "cat.csv",
                "catalog": {"columns": ["lon", "lat", "depth"]},
            },
            "src": {
                "crs": "epsg:4326",
                "xy_units": "deg",
                "z_units": "km",
                "z_positive": "down",
            },
            "dst": {
                "units": "km",
                "anchor": {
                    "data": {"x": 100000.0, "y": 7000000.0, "depth_km": 15.6},
                    "model": [0, 0, -21],
                },
            },
            "output": {"dir": str(self.tmp), "file": "out.vtp"},
        }
        with self.assertRaisesRegex(ValueError, "geographic.*lon/lat"):
            run(cfg, self.tmp)

    def test_mixing_lonlat_and_xy_rejected(self):
        self.write_csv("cat.csv", "lon,lat,depth\n-71,-20,5\n")
        cfg = {
            "job": "project",
            "data": {
                "file": "cat.csv",
                "catalog": {"columns": ["lon", "lat", "depth"]},
            },
            "src": {
                "crs": "epsg:4326",
                "xy_units": "deg",
                "z_units": "km",
                "z_positive": "down",
            },
            "dst": {
                "units": "km",
                "anchor": {
                    "data": {
                        "lon": -71.07, "lat": -20.09,
                        "x": 100.0, "y": 200.0,
                        "depth_km": 15.6,
                    },
                    "model": [0, 0, -21],
                },
            },
            "output": {"dir": str(self.tmp), "file": "out.vtp"},
        }
        with self.assertRaisesRegex(ValueError, "mix lon/lat with x/y"):
            run(cfg, self.tmp)


class TestMeshJob(_JobTestBase):

    def _make_mesh(self):
        ax, ay = _utm_xy(-71.07, -20.09)
        pts = np.array([
            [ax,        ay,        -1000.0],
            [ax + 1000, ay,        -1000.0],
            [ax,        ay + 1000, -2000.0],
            [ax + 1000, ay + 1000, -2500.0],
        ])
        mesh = pv.PolyData(pts)
        mesh.point_data["elev"] = np.array([-1000., -1000., -2000., -2500.])
        return mesh

    def _basic_cfg(self):
        return {
            "job": "project",
            "data": {"file": "in.vtp"},
            "src": {
                "crs": "epsg:32719",
                "xy_units": "m",
                "z_units": "m",
                "z_positive": "up",
            },
            "dst": {
                "units": "km",
                "anchor": {
                    "data": {"lon": -71.07, "lat": -20.09, "depth_km": 1.0},
                    "model": [0, 0, -8],
                    "rotation_deg": -10,
                },
            },
            "output": {"dir": str(self.tmp), "file": "out.vtp"},
        }

    def test_anchor_lands_at_origin(self):
        self.write_mesh("in.vtp", self._make_mesh())
        run(self._basic_cfg(), self.tmp)
        out = pv.read(str(self.tmp / "out.vtp"))
        self.assertEqual(out.n_points, 4)
        np.testing.assert_allclose(out.points[0], [0.0, 0.0, -8.0], atol=1e-4)

    def test_point_data_preserved(self):
        self.write_mesh("in.vtp", self._make_mesh())
        run(self._basic_cfg(), self.tmp)
        out = pv.read(str(self.tmp / "out.vtp"))
        self.assertIn("elev", out.point_data)
        np.testing.assert_array_equal(
            out.point_data["elev"], [-1000., -1000., -2000., -2500.]
        )

    def test_bbox_filters_and_extension_corrected(self):
        ax_in, ay_in = _utm_xy(-71.07, -20.09)
        ax_out, ay_out = _utm_xy(-60.0, -10.0)
        pts = np.array([
            [ax_in,  ay_in,  -1000.0],
            [ax_out, ay_out, -1000.0],
        ])
        self.write_mesh("in.vtp", pv.PolyData(pts))
        cfg = self._basic_cfg()
        cfg["src"]["bbox"] = {"lon": [-72, -70], "lat": [-22, -19]}
        run(cfg, self.tmp)
        self.assertTrue((self.tmp / "out.vtu").exists())
        out = pv.read(str(self.tmp / "out.vtu"))
        self.assertEqual(out.n_points, 1)

    def test_bbox_cleanly_clips_cells_at_boundary(self):
        # A triangle with one vertex inside the bbox and two outside.
        # clip_box should split it, producing new vertices on the bbox edges
        # with interpolated point_data — unlike extract_points which only
        # keeps or drops whole cells.
        import pyproj
        tfm = pyproj.Transformer.from_crs(
            "epsg:4326", "epsg:32719", always_xy=True
        )
        pts_lonlat = [(-71.0, -20.0), (-65.0, -20.0), (-71.0, -15.0)]
        pts = np.array(
            [[*tfm.transform(lon, lat), -1000.0] for lon, lat in pts_lonlat]
        )
        mesh = pv.PolyData(pts).delaunay_2d()
        mesh.point_data["idx"] = np.array([0.0, 1.0, 2.0])
        self.write_mesh("in.vtp", mesh)

        cfg = self._basic_cfg()
        cfg["src"]["bbox"] = {"lon": [-72, -70], "lat": [-21, -19]}
        run(cfg, self.tmp)
        out = pv.read(str(self.tmp / "out.vtu"))

        # New vertices at the cut boundary, more than the input triangle had
        self.assertGreater(out.n_points, 3)
        # point_data interpolated at cut vertices gives non-integer values
        idx = out.point_data["idx"]
        non_integer = np.abs(idx - np.round(idx)) > 1e-6
        self.assertTrue(non_integer.any())

    def test_mesh_with_catalog_block_rejected(self):
        self.write_mesh("in.vtp", self._make_mesh())
        cfg = self._basic_cfg()
        cfg["data"]["catalog"] = {"columns": ["x", "y", "z"]}
        with self.assertRaisesRegex(ValueError, "catalog declared"):
            run(cfg, self.tmp)


class TestValidation(_JobTestBase):

    def _valid_cfg(self):
        (self.tmp / "in.csv").write_text("a,b,c\n-71,-20,5\n")
        return {
            "job": "project",
            "data": {
                "file": "in.csv",
                "catalog": {"columns": ["a", "b", "c"]},
            },
            "src": {
                "crs": "epsg:4326",
                "xy_units": "deg",
                "z_units": "km",
                "z_positive": "down",
            },
            "dst": {
                "units": "km",
                "anchor": {
                    "data": {"lon": -71, "lat": -20, "depth_km": 15},
                    "model": [0, 0, 0],
                },
            },
            "output": {"dir": str(self.tmp), "file": "out.vtp"},
        }

    def test_legacy_projector_block_rejected(self):
        cfg = self._valid_cfg()
        cfg["projector"] = {"src_crs": "epsg:4326"}
        with self.assertRaisesRegex(ValueError, "Legacy 'projector:'"):
            run(cfg, self.tmp)

    def test_legacy_input_block_rejected(self):
        cfg = self._valid_cfg()
        cfg["input"] = cfg.pop("data")
        with self.assertRaisesRegex(ValueError, "Legacy 'input:'"):
            run(cfg, self.tmp)

    def test_missing_data_file(self):
        cfg = self._valid_cfg()
        cfg["data"].pop("file")
        with self.assertRaisesRegex(ValueError, "data.file"):
            run(cfg, self.tmp)

    def test_missing_input_file_on_disk(self):
        cfg = self._valid_cfg()
        cfg["data"]["file"] = "nope.csv"
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            run(cfg, self.tmp)

    def test_missing_src_crs(self):
        cfg = self._valid_cfg()
        cfg["src"].pop("crs")
        with self.assertRaisesRegex(ValueError, "src.crs"):
            run(cfg, self.tmp)

    def test_missing_src_xy_units(self):
        cfg = self._valid_cfg()
        cfg["src"].pop("xy_units")
        with self.assertRaisesRegex(ValueError, "src.xy_units"):
            run(cfg, self.tmp)

    def test_missing_src_z_units(self):
        cfg = self._valid_cfg()
        cfg["src"].pop("z_units")
        with self.assertRaisesRegex(ValueError, "src.z_units"):
            run(cfg, self.tmp)

    def test_missing_src_z_positive(self):
        cfg = self._valid_cfg()
        cfg["src"].pop("z_positive")
        with self.assertRaisesRegex(ValueError, "src.z_positive"):
            run(cfg, self.tmp)

    def test_missing_dst_units(self):
        cfg = self._valid_cfg()
        cfg["dst"].pop("units")
        with self.assertRaisesRegex(ValueError, "dst.units"):
            run(cfg, self.tmp)

    def test_missing_dst_anchor(self):
        cfg = self._valid_cfg()
        cfg["dst"].pop("anchor")
        with self.assertRaisesRegex(ValueError, "dst.anchor"):
            run(cfg, self.tmp)

    def test_anchor_missing_data(self):
        cfg = self._valid_cfg()
        cfg["dst"]["anchor"].pop("data")
        with self.assertRaisesRegex(ValueError, "data.*model"):
            run(cfg, self.tmp)

    def test_anchor_missing_depth_km(self):
        cfg = self._valid_cfg()
        cfg["dst"]["anchor"]["data"].pop("depth_km")
        with self.assertRaisesRegex(ValueError, "depth_km"):
            run(cfg, self.tmp)

    def test_anchor_model_wrong_length(self):
        cfg = self._valid_cfg()
        cfg["dst"]["anchor"]["model"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "model must be"):
            run(cfg, self.tmp)

    def test_unsupported_extension(self):
        (self.tmp / "data.txt").write_text("nope")
        cfg = self._valid_cfg()
        cfg["data"]["file"] = "data.txt"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            run(cfg, self.tmp)


class TestRasterJob(_JobTestBase):

    def write_raster(self, name, bands, res=1000.0, nx=10, ny=8):
        tfm = pyproj.Transformer.from_crs(
            "epsg:4326", "epsg:32719", always_xy=True
        )
        cx, cy = tfm.transform(-71.07, -20.09)
        left = cx - (nx / 2) * res
        top = cy + (ny / 2) * res
        transform = from_origin(left, top, res, res)
        path = self.tmp / name
        with rasterio.open(
            str(path), "w",
            driver="GTiff", height=ny, width=nx, count=len(bands),
            dtype="float32", crs="epsg:32719", transform=transform,
        ) as ds:
            for i, arr in enumerate(bands, start=1):
                ds.write(arr.astype(np.float32), i)
        return path

    def _basic_cfg(self, fname="topo.tif", z_band=None):
        data = {"file": fname}
        if z_band is not None:
            data["raster"] = {"z_band": z_band}
        return {
            "job": "project",
            "data": data,
            "src": {
                "crs": "epsg:32719",
                "xy_units": "m",
                "z_units": "m",
                "z_positive": "up",
            },
            "dst": {
                "units": "km",
                "anchor": {
                    "data": {"lon": -71.07, "lat": -20.09, "depth_km": 15.6},
                    "model": [0, 0, -21],
                    "rotation_deg": 0,
                },
            },
            "output": {"dir": str(self.tmp), "file": "out.vtp"},
        }

    def test_flat_raster_z_is_constant(self):
        self.write_raster("topo.tif", [np.zeros((8, 10))])
        run(self._basic_cfg(), self.tmp)
        poly = pv.read(str(self.tmp / "out.vtp"))
        self.assertIsInstance(poly, pv.PolyData)
        self.assertEqual(poly.n_points, 80)
        self.assertEqual(poly.n_cells, 2 * 9 * 7)
        z = poly.points[:, 2]
        np.testing.assert_allclose(z, z[0], atol=1e-9)

    def test_relief_from_z_band(self):
        band = np.arange(80, dtype=float).reshape(8, 10) * 10.0
        self.write_raster("topo.tif", [band])
        run(self._basic_cfg(z_band=1), self.tmp)
        poly = pv.read(str(self.tmp / "out.vtp"))
        z_range = poly.points[:, 2].max() - poly.points[:, 2].min()
        np.testing.assert_allclose(z_range, 0.79, atol=1e-6)
        normals = poly.compute_normals().cell_data["Normals"]
        self.assertTrue((normals[:, 2] > 0).all())

    def test_bands_stored_as_point_data(self):
        b1 = np.ones((8, 10)) * 100.0
        b2 = np.ones((8, 10)) * 7.0
        b3 = np.arange(80).reshape(8, 10).astype(float)
        self.write_raster("topo.tif", [b1, b2, b3])
        run(self._basic_cfg(z_band=1), self.tmp)
        poly = pv.read(str(self.tmp / "out.vtp"))
        names = list(poly.point_data.keys())
        self.assertIn("band_1", names)
        self.assertIn("band_2", names)
        self.assertIn("band_3", names)
        np.testing.assert_allclose(poly.point_data["band_2"], 7.0)

    def test_bbox_crops_raster(self):
        self.write_raster("topo.tif", [np.zeros((8, 10))])
        cfg = self._basic_cfg()
        cfg["src"]["bbox"] = {
            "lon": [-71.10, -71.04],
            "lat": [-20.11, -20.07],
        }
        run(cfg, self.tmp)
        poly = pv.read(str(self.tmp / "out.vtp"))
        self.assertLess(poly.n_points, 80)
        self.assertGreater(poly.n_points, 0)
        self.assertGreater(poly.n_cells, 0)

    def test_bbox_outside_raster_raises(self):
        self.write_raster("topo.tif", [np.zeros((8, 10))])
        cfg = self._basic_cfg()
        cfg["src"]["bbox"] = {
            "lon": [-10.0, -9.0],
            "lat": [10.0, 11.0],
        }
        with self.assertRaises(ValueError):
            run(cfg, self.tmp)

    def test_z_band_out_of_range_raises(self):
        self.write_raster("topo.tif", [np.zeros((8, 10))])
        with self.assertRaises(ValueError):
            run(self._basic_cfg(z_band=2), self.tmp)

    def test_z_band_zero_rejected(self):
        self.write_raster("topo.tif", [np.zeros((8, 10))])
        with self.assertRaises(ValueError):
            run(self._basic_cfg(z_band=0), self.tmp)

    def test_extension_autocorrected_to_vtp(self):
        self.write_raster("topo.tif", [np.zeros((8, 10))])
        cfg = self._basic_cfg()
        cfg["output"]["file"] = "wrong.vtu"
        run(cfg, self.tmp)
        self.assertTrue((self.tmp / "wrong.vtp").exists())
        self.assertFalse((self.tmp / "wrong.vtu").exists())

    def test_tiff_extension_accepted(self):
        self.write_raster("topo.tiff", [np.zeros((8, 10))])
        run(self._basic_cfg(fname="topo.tiff"), self.tmp)
        self.assertTrue((self.tmp / "out.vtp").exists())

    def test_catalog_block_on_raster_rejected(self):
        self.write_raster("topo.tif", [np.zeros((8, 10))])
        cfg = self._basic_cfg()
        cfg["data"]["catalog"] = {"columns": ["x", "y", "z"]}
        with self.assertRaises(ValueError):
            run(cfg, self.tmp)

    def test_rotation_applied_to_grid(self):
        self.write_raster("topo.tif", [np.zeros((8, 10))])
        cfg = self._basic_cfg()
        cfg["dst"]["anchor"]["rotation_deg"] = 45.0
        run(cfg, self.tmp)
        poly = pv.read(str(self.tmp / "out.vtp"))
        self.assertIsInstance(poly, pv.PolyData)
        self.assertEqual(poly.n_points, 80)
        self.assertEqual(poly.n_cells, 2 * 9 * 7)

    def test_lonlat_raster_with_meters_z(self):
        nx, ny = 10, 8
        res_deg = 0.01
        left = -71.07 - (nx / 2) * res_deg
        top = -20.09 + (ny / 2) * res_deg
        tfm = from_origin(left, top, res_deg, res_deg)
        band = np.arange(nx * ny, dtype=np.float32).reshape(ny, nx) * 10.0
        path = self.tmp / "topo_ll.tif"
        with rasterio.open(
            str(path), "w",
            driver="GTiff", height=ny, width=nx, count=1,
            dtype="float32", crs="epsg:4326", transform=tfm,
        ) as ds:
            ds.write(band, 1)
        cfg = self._basic_cfg(fname="topo_ll.tif", z_band=1)
        cfg["src"] = {
            "crs": "epsg:4326", "xy_units": "deg",
            "z_units": "m", "z_positive": "up",
        }
        run(cfg, self.tmp)
        poly = pv.read(str(self.tmp / "out.vtp"))
        self.assertEqual(poly.n_points, 80)
        np.testing.assert_allclose(
            poly.points[:, 2].max() - poly.points[:, 2].min(),
            0.79, atol=1e-6,
        )

    def test_nodata_creates_holes(self):
        band = np.ones((8, 10), dtype=np.float32) * 100.0
        band[3:5, 3:5] = -9999.0
        cx, cy = _utm_xy(-71.07, -20.09)
        res, nx, ny = 1000.0, 10, 8
        tfm = from_origin(cx - (nx/2)*res, cy + (ny/2)*res, res, res)
        path = self.tmp / "holes.tif"
        with rasterio.open(
            str(path), "w",
            driver="GTiff", height=ny, width=nx, count=1,
            dtype="float32", crs="epsg:32719", transform=tfm,
            nodata=-9999.0,
        ) as ds:
            ds.write(band, 1)
        cfg = self._basic_cfg(fname="holes.tif", z_band=1)
        run(cfg, self.tmp)
        poly = pv.read(str(self.tmp / "out.vtp"))
        self.assertLess(poly.n_cells, 2 * 9 * 7)
        self.assertGreater(poly.n_cells, 0)


if __name__ == "__main__":
    unittest.main()