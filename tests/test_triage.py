import json
import subprocess

import h5py
import numpy as np
import pytest
import tifffile
import zarr

from glance_volume.triage import Volume, VolumeError


def test_npy_volume(tmp_path):
    path = tmp_path / "volume.npy"
    np.save(path, np.zeros((3, 4, 5), dtype=np.uint16))
    volume = Volume(path)
    assert volume.shape == (3, 4, 5)
    assert str(volume.dtype) == "uint16"
    assert volume.format_description == "NumPy .npy"


def test_npz_selects_largest_3d_array(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        image=np.zeros((3, 4, 5), dtype=np.float32),
        tiny=np.zeros((2, 2), dtype=np.uint8),
        bigger=np.zeros((4, 4, 4), dtype=np.int16),
    )
    volume = Volume(path)
    assert volume.dataset == "bigger"
    assert volume.shape == (4, 4, 4)
    assert str(volume.dtype) == "int16"


def test_npz_query_overrides_scales(tmp_path):
    path = tmp_path / "labels_and_image.npz"
    np.savez(
        path,
        labels=np.zeros((2, 3, 4), dtype=np.uint8),
        image=np.zeros((2, 3, 4), dtype=np.uint8),
    )
    volume = Volume(f"{path}?z=13&y=2")
    assert volume.dataset == "labels"
    assert volume.scales == (13.0, 2.0, 1.0)


def test_tiff_volume(tmp_path):
    path = tmp_path / "volume.tiff"
    tifffile.imwrite(path, np.zeros((2, 3, 4), dtype=np.uint8))
    volume = Volume(path)
    assert volume.shape == (2, 3, 4)
    assert volume.format_description == "TIFF"


def test_ome_tiff_scales(tmp_path):
    path = tmp_path / "ome.tiff"
    tifffile.imwrite(
        path,
        np.zeros((2, 3, 5), dtype=np.uint8),
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 2.0,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeX": 0.25,
        },
    )
    volume = Volume(path)
    assert volume.format_description == "OME-TIFF"
    assert volume.shape == (2, 3, 5)
    assert volume.scales == (2.0, 0.5, 0.25)


def test_volume_path_query_overrides_scales(tmp_path):
    path = tmp_path / "queried_ome.tiff"
    tifffile.imwrite(
        path,
        np.zeros((2, 3, 5), dtype=np.uint8),
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 2.0,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeX": 0.25,
        },
    )
    volume = Volume(f"{path}?z=10&y=2&x=0.5")
    assert volume.path == str(path)
    assert volume.scales == (10.0, 2.0, 0.5)


def test_volume_path_query_allows_partial_xyz_overrides(tmp_path):
    path = tmp_path / "queried_ome_missing_x.tiff"
    tifffile.imwrite(
        path,
        np.zeros((2, 3, 5), dtype=np.uint8),
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 2.0,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeX": 0.25,
        },
    )
    volume = Volume(f"{path}?z=10&y=2")
    assert volume.scales == (10.0, 2.0, 0.25)


def test_volume_path_query_allows_single_axis_override(tmp_path):
    path = tmp_path / "queried_ome_x_only.tiff"
    tifffile.imwrite(
        path,
        np.zeros((2, 3, 5), dtype=np.uint8),
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 2.0,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeX": 0.25,
        },
    )
    volume = Volume(f"{path}?x=0.5")
    assert volume.scales == (2.0, 0.5, 0.5)


def test_ome_zarr_scales(tmp_path):
    path = tmp_path / "ome.zarr"
    root = zarr.open_group(str(path), mode="w")
    # Use create_dataset for compatibility with installed zarr versions
    root.create_dataset("0", data=np.zeros((2, 3, 4), dtype=np.uint8))
    root.attrs["multiscales"] = [
        {
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [4.0, 2.0, 1.0]}
                    ],
                }
            ]
        }
    ]
    volume = Volume(path)
    assert volume.format_description == "OME-Zarr"
    assert volume.scales == (4.0, 2.0, 1.0)


def test_hdf5_selects_largest_volume(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("small", data=np.zeros((2, 2), dtype=np.uint8))
        handle.create_dataset("volumes/main", data=np.zeros((5, 4, 3), dtype=np.int32))
    volume = Volume(path)
    assert volume.dataset == "volumes/main"
    assert volume.shape == (5, 4, 3)
    assert str(volume.dtype) == "int32"


def test_raises_when_no_volume_data(tmp_path):
    path = tmp_path / "flat.npy"
    np.save(path, np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(VolumeError):
        Volume(path)


def test_json_is_serializable(tmp_path):
    path = tmp_path / "volume.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=np.uint8))
    payload = Volume(path).json()
    encoded = json.dumps(payload)
    assert isinstance(encoded, str)


def test_volume_triage_script(tmp_path):
    path = tmp_path / "volume.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=np.uint8))
    completed = subprocess.run(
        ["volume-triage", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["path"] == str(path)
    assert payload["shape"] == [2, 3, 4]
    assert payload["dtype"] == "uint8"
