from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit
from pathlib import Path
from typing import Any
import neuroglancer

import numpy as np

# Symbolic constants for format descriptions
FORMAT_UNKNOWN = ""
FORMAT_NUMPY_NPY = "NumPy .npy"
FORMAT_NUMPY_NPZ = "NumPy .npz"
FORMAT_OME_TIFF = "OME-TIFF"
FORMAT_TIFF = "TIFF"
FORMAT_OME_ZARR = "OME-Zarr"
FORMAT_ZARR = "Zarr"
FORMAT_HDF5 = "HDF5"
DIMENSION_NAMES = ["z", "y", "x"]

class VolumeError(ValueError):
    pass


@dataclass(frozen=True)
class _VolumeMetadata:
    format_description: str
    dataset: str | None
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    scales: tuple[float, ...] | None


def _numel(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64))


def _choose_largest(candidates: list[tuple[tuple[int, ...], np.dtype[Any], str | None, tuple[float, ...] | None]]) -> _VolumeMetadata:
    valid = [item for item in candidates if len(item[0]) >= 3]
    if not valid:
        raise VolumeError("No volume data with ndim >= 3 was found.")
    shape, dtype, dataset, scales = max(valid, key=lambda item: _numel(item[0]))
    return _VolumeMetadata(
        format_description=FORMAT_UNKNOWN,
        dataset=dataset,
        dtype=np.dtype(dtype),
        shape=shape,
        scales=scales,
    )


def _split_volume_path(path: str | Path) -> tuple[Path, tuple[float | None, float | None, float | None] | None]:
    path_text = str(path)
    parsed = urlsplit(path_text)
    actual_path = Path(parsed.path or path_text)
    if not parsed.query:
        return actual_path, None

    params = parse_qs(parsed.query, keep_blank_values=True)
    if not any(name in params for name in ("x", "y", "z")):
        return actual_path, None

    overrides_list: list[float | None] = []
    for name in ("z", "y", "x"):
        value = params.get(name)
        if not value:
            overrides_list.append(None)
            continue
        try:
            overrides_list.append(float(value[0]))
        except (TypeError, ValueError, IndexError) as exc:
            raise VolumeError(f"Invalid scale override in volume path: {path_text}") from exc
    overrides = tuple(overrides_list)
    return actual_path, overrides


def _npy_header(handle: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version in {(2, 0), (3, 0)}:
        shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
    elif hasattr(np.lib.format, "_read_array_header"):
        shape, _, dtype = np.lib.format._read_array_header(handle, version)  # type: ignore[attr-defined]
    else:
        raise VolumeError(f"Unsupported npy version {version!r}.")
    return tuple(int(i) for i in shape), np.dtype(dtype)


def _from_npy(path: Path) -> _VolumeMetadata:
    with path.open("rb") as handle:
        shape, dtype = _npy_header(handle)
    metadata = _choose_largest([(shape, dtype, None, None)])
    return _VolumeMetadata(
        format_description=FORMAT_NUMPY_NPY,
        dataset=metadata.dataset,
        dtype=metadata.dtype,
        shape=metadata.shape,
        scales=metadata.scales,
    )


def _from_npz(path: Path) -> _VolumeMetadata:
    candidates: list[tuple[tuple[int, ...], np.dtype[Any], str | None, tuple[float, ...] | None]] = []
    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name, "r") as handle:
                shape, dtype = _npy_header(handle)
                candidates.append((shape, dtype, name.removesuffix(".npy"), None))
    metadata = _choose_largest(candidates)
    return _VolumeMetadata(
        format_description=FORMAT_NUMPY_NPZ,
        dataset=metadata.dataset,
        dtype=metadata.dtype,
        shape=metadata.shape,
        scales=metadata.scales,
    )


def _extract_ome_tiff_scales(ome_metadata: str | None, axes: str | None) -> tuple[float, ...] | None:
    if not ome_metadata or not axes:
        return None
    try:
        import tifffile

        parsed = tifffile.xml2dict(ome_metadata)
    except Exception:
        return None

    ome = parsed.get("OME")
    if not isinstance(ome, dict):
        return None
    image = ome.get("Image")
    if isinstance(image, list):
        if len(image) != 1:
            return None
        image = image[0]
    if not isinstance(image, dict):
        return None
    pixels = image.get("Pixels")
    if not isinstance(pixels, dict):
        return None

    axis_sizes = {
        "Z": pixels.get("PhysicalSizeZ"),
        "Y": pixels.get("PhysicalSizeY"),
        "X": pixels.get("PhysicalSizeX"),
    }
    if not all(axis_sizes.get(a) is not None for a in axes if a in {"Z", "Y", "X"}):
        return None
    result: list[float] = []
    for axis in axes:
        if axis in axis_sizes:
            result.append(float(axis_sizes[axis]))
        else:
            return None
    return tuple(result)


def _from_tiff(path: Path) -> _VolumeMetadata:
    import tifffile

    candidates: list[tuple[tuple[int, ...], np.dtype[Any], str | None, tuple[float, ...] | None]] = []
    with tifffile.TiffFile(path) as tif:
        for index, series in enumerate(tif.series):
            shape = tuple(int(i) for i in series.shape)
            dtype = np.dtype(series.dtype)
            axes = getattr(series, "axes", None)
            scales = _extract_ome_tiff_scales(tif.ome_metadata, axes) if tif.is_ome else None
            candidates.append((shape, dtype, f"series[{index}]", scales))
        metadata = _choose_largest(candidates)
        format_description = FORMAT_OME_TIFF if tif.is_ome else FORMAT_TIFF
    return _VolumeMetadata(
        format_description=format_description,
        dataset=metadata.dataset,
        dtype=metadata.dtype,
        shape=metadata.shape,
        scales=metadata.scales,
    )


def _extract_ome_zarr_scales(multiscales: Any, dataset_name: str | None) -> tuple[float, ...] | None:
    if not dataset_name or not isinstance(multiscales, list) or len(multiscales) != 1:
        return None
    entry = multiscales[0]
    if not isinstance(entry, dict):
        return None
    datasets = entry.get("datasets")
    if not isinstance(datasets, list):
        return None
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        if dataset.get("path") != dataset_name:
            continue
        transforms = dataset.get("coordinateTransformations")
        if not isinstance(transforms, list):
            continue
        scales = [t for t in transforms if isinstance(t, dict) and t.get("type") == "scale"]
        if len(scales) != 1:
            return None
        scale_values = scales[0].get("scale")
        if not isinstance(scale_values, list):
            return None
        return tuple(float(v) for v in scale_values)
    return None


def _from_zarr(path: Path) -> _VolumeMetadata:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    candidates: list[tuple[tuple[int, ...], np.dtype[Any], str | None, tuple[float, ...] | None]] = []

    if hasattr(root, "members"):
        for name, obj in root.members(max_depth=None):
            if not isinstance(obj, zarr.Array):
                continue
            shape = tuple(int(i) for i in obj.shape)
            dtype = np.dtype(obj.dtype)
            candidates.append((shape, dtype, name, None))
    else:
        # zarr 2.x compatibility: use arrays() iterator
        for name, obj in root.arrays(recurse=True):
            shape = tuple(int(i) for i in obj.shape)
            dtype = np.dtype(obj.dtype)
            candidates.append((shape, dtype, name, None))
    metadata = _choose_largest(candidates)
    scales = _extract_ome_zarr_scales(root.attrs.get("multiscales"), metadata.dataset)
    return _VolumeMetadata(
        format_description=FORMAT_OME_ZARR if scales is not None else FORMAT_ZARR,
        dataset=metadata.dataset,
        dtype=metadata.dtype,
        shape=metadata.shape,
        scales=scales,
    )


def _from_hdf5(path: Path) -> _VolumeMetadata:
    import h5py

    candidates: list[tuple[tuple[int, ...], np.dtype[Any], str | None, tuple[float, ...] | None]] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            shape = tuple(int(i) for i in obj.shape)
            dtype = np.dtype(obj.dtype)
            candidates.append((shape, dtype, name, None))

        handle.visititems(visit)
    metadata = _choose_largest(candidates)
    return _VolumeMetadata(
        format_description=FORMAT_HDF5,
        dataset=metadata.dataset,
        dtype=metadata.dtype,
        shape=metadata.shape,
        scales=metadata.scales,
    )


class Volume:
    def __init__(self, path: str | Path, verbose: bool = False):
        self.verbose = verbose
        path, scale_overrides = _split_volume_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = str(path)

        metadata: _VolumeMetadata
        suffix = path.suffix.lower()
        if suffix == ".npy":
            metadata = _from_npy(path)
        elif suffix == ".npz":
            metadata = _from_npz(path)
        elif suffix in {".tif", ".tiff"}:
            metadata = _from_tiff(path)
        elif suffix in {".h5", ".hdf5"}:
            metadata = _from_hdf5(path)
        elif path.is_dir():
            metadata = _from_zarr(path)
        else:
            raise VolumeError(f"Unsupported file format for {path}.")
        
        if scale_overrides is not None:
            mscales = metadata.scales if metadata.scales is not None else (1.0,) * len(metadata.shape)
            if len(mscales) < 3:
                raise VolumeError("Scale overrides require a volume with at least 3 dimensions.")
            scales = list(mscales)
            for index, override in enumerate(scale_overrides):
                if override is not None:
                    scales[index] = override
            metadata = _VolumeMetadata(
                format_description=metadata.format_description,
                dataset=metadata.dataset,
                dtype=metadata.dtype,
                shape=metadata.shape,
                scales=tuple(scales),
            )

        self.format_description = metadata.format_description
        self.dataset = metadata.dataset
        self.dtype = metadata.dtype
        self.shape = metadata.shape
        self.scales = metadata.scales
        self._array: np.ndarray | None = None

    @property
    def array(self) -> np.ndarray:
        """Return the volume array, reading and caching it on first access."""
        if self._array is not None:
            return self._array

        path = Path(self.path)
        fmt = self.format_description

        if self.verbose:
            print("Reading volume")
            print(json.dumps(self.json(), indent=2, sort_keys=True))

        if fmt == FORMAT_NUMPY_NPY:
            arr = np.load(path)
        elif fmt == FORMAT_NUMPY_NPZ:
            npz = np.load(path)
            if self.dataset and self.dataset in npz:
                arr = npz[self.dataset]
            elif len(npz.files) == 1:
                arr = npz[npz.files[0]]
            else:
                raise VolumeError(f"Cannot determine dataset in {path}")
        elif fmt in {FORMAT_OME_TIFF, FORMAT_TIFF}:
            try:
                import tifffile

                with tifffile.TiffFile(path) as tif:
                    if self.dataset and self.dataset.startswith("series[") and self.dataset.endswith("]"):
                        index = int(self.dataset[len("series["):-1])
                    else:
                        index = 0
                    series = tif.series[index]
                    arr = series.asarray()
            except Exception as exc:  # pragma: no cover - platform-specific tifffile errors
                raise VolumeError(f"Error reading TIFF: {exc}")
        elif fmt == FORMAT_HDF5:
            try:
                import h5py

                with h5py.File(path, "r") as handle:
                    if not self.dataset:
                        raise VolumeError("HDF5 dataset name unknown")
                    arr = handle[self.dataset][()]
            except Exception as exc:  # pragma: no cover - h5py I/O errors
                raise VolumeError(f"Error reading HDF5: {exc}")
        elif fmt in {FORMAT_OME_ZARR, FORMAT_ZARR}:
            try:
                import zarr

                root = zarr.open_group(str(path), mode="r")
                if self.dataset and self.dataset in root:
                    arr = root[self.dataset][...]
                else:
                    # try to find the first array member
                    found = None
                    for name, obj in root.members(max_depth=None):
                        try:
                            import zarr as _z

                            if isinstance(obj, _z.Array):
                                found = name
                                break
                        except Exception:
                            continue
                    if found is None:
                        raise VolumeError(f"No array found in Zarr at {path}")
                    arr = root[found][...]
            except Exception as exc:  # pragma: no cover - zarr I/O errors
                raise VolumeError(f"Error reading Zarr: {exc}")
        else:
            # fallback to suffix-based behavior for unknown formats
            suffix = path.suffix.lower()
            if suffix == ".npy":
                arr = np.load(path)
            else:
                raise VolumeError(f"Unsupported file format for reading array: {path}")

        self._array = np.asarray(arr)
        return self._array

    def json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format_description,
            "dataset": self.dataset,
            "dtype": str(self.dtype),
            "shape": list(self.shape),
            "scales": list(self.scales) if self.scales is not None else None,
        }
    
    def neuroglancer_volume(self, scales=None) -> neuroglancer.LocalVolume:
        if scales is None:
            scales = self.scales
        dimensionscale = [1,1,1]
        if scales is not None and len(scales) != len(self.shape):
            raise ValueError("Scales length must match shape length.")
        if scales is not None:
            dimensionscale = list(scales)
        print("creating neuroglancer volume with scales:", dimensionscale)
        dimensions = neuroglancer.CoordinateSpace(
            names=DIMENSION_NAMES[:],
            units="nm", 
            scales=dimensionscale)
        return neuroglancer.LocalVolume(
            data=self.array,
            dimensions=dimensions,
        )
    
    def neuroglancer_image_layer(self, scales=None, color="white") -> neuroglancer.ImageLayer:
        image = self.array
        p1, p99 = np.percentile(image, [1, 99])
        shader = neuroglancer_colored_shader(color, p1, p99)
        result = neuroglancer.ImageLayer(
            source=self.neuroglancer_volume(scales),
            shader=shader,
        )
        return result
    
    def neuroglancer_segmentation_layer(self, scales=None, hexcolors=None) -> neuroglancer.SegmentationLayer:
        result = neuroglancer.SegmentationLayer(
            source=self.neuroglancer_volume(scales),
        )
        if hexcolors is not None:
            result.segment_colors.update(hexcolors)
        return result

    def __repr__(self) -> str:
        return json.dumps(self.json(), sort_keys=True)
    
def neuroglancer_colored_shader(color: str, min_value: float = 0.0, max_value: float = 1.0) -> str:
    return NEUROGLANCER_COLORED_GLSL % (float(min_value), float(max_value), color)

NEUROGLANCER_COLORED_GLSL = """
#uicontrol invlerp normalized(range=[%s, %s])
#uicontrol vec3 color color(default="%s")

void main() {
    emitRGB(color * normalized());
}
"""

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="volume-triage",
        description="Print JSON metadata for a volume path.",
    )
    parser.add_argument("path", help="Path to the volume file or directory.")
    args = parser.parse_args(argv)
    print(json.dumps(Volume(args.path).json(), indent=2, sort_keys=True))
    return 0
