# photonsfile

Pure-Python reader for Photonscore LINCam `.photons` files (the **D7** container).
numpy only, with optional numba acceleration for the varint decode.

Photonscore LINCam is a position-sensitive single-photon detector: every photon
carries an `(x, y)` detector position, a TCSPC micro time (`dt`), and a macro time
in milliseconds (`ms`). This library decodes those per-photon streams and can bin
them into an intensity image, a TCSPC decay, or a `(Y, X, H)` FLIM cube.

## The D7 format

A `.photons` file is a paged container: 16 KB pages, each carrying a 2-byte page
marker, holding a protobuf-style header, a per-dataset index, and an epilogue. Each
dataset (`/photons/x`, `/photons/y`, `/photons/dt`, `/photons/ms`, and on dual-TDC
detectors `/start/time` and `/stop/time`) is stored as a seed value followed by
zigzag-varint delta blocks. `dt` to time is calibrated from the file's
`/photons/TacChannel` attribute (picoseconds per raw unit).

The format is documented at [github.com/photonscore/d7](https://github.com/photonscore/d7)
(Apache-2.0). This reader was written from that specification and **validated
bit-exact against the Photonscore SDK**. No vendor source is redistributed.

## Install

```bash
pip install photonsfile          # numpy only
pip install photonsfile[all]     # + numba, for a faster varint decode
```

## Usage

```python
from photonsfile import PhotonsFile, imread

with PhotonsFile('sample.photons') as f:
    ph = f.photons()               # {'x', 'y', 'dt', 'ms'} arrays, one per photon
    img = f.image(pixels=512)      # (Y, X) intensity image
    decay = f.decay(bins=256)      # summed TCSPC histogram
    cube = f.flim_image(512, 256)  # (Y, X, H) FLIM cube
    dt_res = f.tcspc_resolution(256)   # seconds per bin, from TacChannel
    print(f.attributes)            # D7 file attributes

img = imread('sample.photons')     # shortcut for the intensity image
```

Low-level access to the decoder is also exported: `read_header`,
`read_attributes`, `read_photons`, `dataset_names`, `has_dual_tdc`.

numba is optional; without it the varint decode uses a vectorised numpy fallback.
`photonsfile.have_numba()` reports which path is active.

## Acknowledgement

Thank you to **Photonscore GmbH** for supporting this work: they shared the LINCam
SDK and a sample `.photons` file, and open-sourced the D7 storage format at
[github.com/photonscore/d7](https://github.com/photonscore/d7) (Apache-2.0). At
Photonscore's request, the credit here is to the company rather than to an
individual.

## Provenance

The decoder is original code, first worked out from the SDK and the sample file,
then checked field for field and bit-exact against the public D7 specification on a
284-million-photon sample. No Photonscore source is redistributed here; the reader
is pure Python (numpy, with an optional numba path) and needs none of Photonscore's
native libraries.

## Citation

If you use photonsfile, please cite it via the DOI. Archived on Zenodo:

<!-- After minting the Zenodo DOI, replace XXXXXXX in the badge and line below. -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> Hunt, A. *photonsfile: a pure-Python reader for Photonscore LINCam .photons
> (D7) files*. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub shows a
"Cite this repository" button from it.

## License

MIT - see `LICENSE`.

