# glance_volume
Analyse or view volume data.

`glance_volume.triage.Volume(path)` inspects volume metadata (shape, dtype, and
scales when unambiguous) for `.npy`, `.npz`, TIFF/OME-TIFF, OME-Zarr, and HDF5
inputs.

# Scripts

```bash
$ # Print volume metadata from PATH as JSON.
$ volume-triage PATH
$ # Example path notation, providing scaling (x implicitly 1).
$ volume-triage labels_and_image.npz?z=13\&y=2
```

# Development install

To install the module in development mode, clone the git repository and then run:

```bash
pip install -e .
```
