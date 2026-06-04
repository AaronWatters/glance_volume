# glance_volume
Analyse or view volume data.

`glance_volume.triage.Volume(path)` inspects volume metadata (shape, dtype, and
scales when unambiguous) for `.npy`, `.npz`, TIFF/OME-TIFF, OME-Zarr, and HDF5
inputs.

# Scripts

```bash
$ volume-triage PATH # Print volume metadata from PATH as JSON.
```

# Development install

To install the module in development mode, clone the git repository and then run:

```bash
pip install -e .
```
