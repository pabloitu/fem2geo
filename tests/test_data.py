import numpy as np
import pytest

from fem2geo.data import FractureData, FaultData, CatalogData


# FractureData

def test_fracture_basic():
    fd = FractureData(planes=[[10, 30], [200, 80], [355, 5]])
    assert len(fd) == 3
    assert fd.planes.shape == (3, 2)
    np.testing.assert_array_equal(fd.planes[0], [10, 30])


def test_fracture_single_pair_promoted_to_2d():
    fd = FractureData(planes=[45, 60])
    assert fd.planes.shape == (1, 2)
    np.testing.assert_array_equal(fd.planes[0], [45, 60])
    assert len(fd) == 1


def test_fracture_cast_to_float():
    fd = FractureData(planes=[[10, 30]])
    assert fd.planes.dtype == float


def test_fracture_wrong_columns():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        FractureData(planes=[[10, 30, 50]])


def test_fracture_repr():
    fd = FractureData(planes=[[10, 30], [20, 40]])
    assert repr(fd) == "FractureData(2 measurements)"


def test_fracture_accepts_ndarray():
    arr = np.array([[10.0, 30.0], [20.0, 40.0]])
    fd = FractureData(planes=arr)
    assert len(fd) == 2


# FaultData

def test_fault_basic():
    fd = FaultData(planes=[[10, 30], [200, 80]], rakes=[45, -90])
    assert len(fd) == 2
    np.testing.assert_array_equal(fd.rakes, [45, -90])


def test_fault_single_measurement():
    fd = FaultData(planes=[45, 60], rakes=[90])
    assert fd.planes.shape == (1, 2)
    assert fd.rakes.shape == (1,)
    assert len(fd) == 1


def test_fault_cast_to_float():
    fd = FaultData(planes=[[10, 30]], rakes=[45])
    assert fd.planes.dtype == float
    assert fd.rakes.dtype == float


def test_fault_rakes_raveled():
    fd = FaultData(planes=[[10, 30], [20, 40]], rakes=[[45], [-90]])
    assert fd.rakes.shape == (2,)


def test_fault_wrong_plane_columns():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        FaultData(planes=[[10, 30, 50]], rakes=[45])


def test_fault_length_mismatch():
    with pytest.raises(ValueError, match="same number of rows"):
        FaultData(planes=[[10, 30], [20, 40]], rakes=[45])


def test_fault_rake_out_of_range_positive():
    with pytest.raises(ValueError, match="-180, 180"):
        FaultData(planes=[[10, 30]], rakes=[200])


def test_fault_rake_out_of_range_negative():
    with pytest.raises(ValueError, match="-180, 180"):
        FaultData(planes=[[10, 30]], rakes=[-181])


def test_fault_rake_at_boundary():
    fd = FaultData(planes=[[10, 30], [20, 40]], rakes=[180, -180])
    np.testing.assert_array_equal(fd.rakes, [180, -180])


def test_fault_repr():
    fd = FaultData(planes=[[10, 30]], rakes=[45])
    assert repr(fd) == "FaultData(1 measurements)"


# CatalogData

def test_catalog_basic():
    cat = CatalogData(x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9])
    assert len(cat) == 3
    np.testing.assert_array_equal(cat.x, [1, 2, 3])
    np.testing.assert_array_equal(cat.y, [4, 5, 6])
    np.testing.assert_array_equal(cat.z, [7, 8, 9])
    assert cat.attrs == {}


def test_catalog_coords_cast_to_float():
    cat = CatalogData(x=[1, 2], y=[3, 4], z=[5, 6])
    assert cat.x.dtype == float
    assert cat.y.dtype == float
    assert cat.z.dtype == float


def test_catalog_attrs_kept_and_cast():
    cat = CatalogData(
        x=[0, 1, 2], y=[0, 1, 2], z=[0, 1, 2],
        attrs={"mag": [3.5, 4.1, 2.8], "depth_km": [10, 20, 30]},
    )
    assert set(cat.attrs) == {"mag", "depth_km"}
    np.testing.assert_array_equal(cat.attrs["mag"], [3.5, 4.1, 2.8])
    assert isinstance(cat.attrs["depth_km"], np.ndarray)


def test_catalog_mismatched_xyz_lengths():
    with pytest.raises(ValueError, match="same length"):
        CatalogData(x=[1, 2, 3], y=[1, 2], z=[1, 2, 3])


def test_catalog_mismatched_attr_length():
    with pytest.raises(ValueError, match="attr 'mag'"):
        CatalogData(
            x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3],
            attrs={"mag": [1.0, 2.0]},
        )


def test_catalog_ravels_2d_input():
    cat = CatalogData(x=np.array([[1, 2, 3]]), y=[4, 5, 6], z=[7, 8, 9])
    assert cat.x.shape == (3,)


def test_catalog_empty():
    cat = CatalogData(x=[], y=[], z=[])
    assert len(cat) == 0
    assert cat.attrs == {}


def test_catalog_repr_lists_attr_names():
    cat = CatalogData(
        x=[0], y=[0], z=[0],
        attrs={"mag": [3.0], "rms": [0.5]},
    )
    r = repr(cat)
    assert "1 points" in r
    assert "mag" in r and "rms" in r


def test_catalog_default_attrs_independent():
    a = CatalogData(x=[1], y=[1], z=[1])
    b = CatalogData(x=[2], y=[2], z=[2])
    a.attrs["foo"] = np.array([1.0])
    assert "foo" not in b.attrs
