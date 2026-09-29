"""Read Photonscore LINCam .photons files.

Photonscore LINCam detectors write time- and space-resolved single-photon data
to a `.photons` file, the D7 container. This is a pure-Python reader for it:
numpy only, with optional numba acceleration for the varint decode.

The D7 container is a paged file (16 KB pages, each carrying a 2-byte page
marker), holding a protobuf-style header, a per-dataset index, and an epilogue.
Each photon dataset (`/photons/x`, `/photons/y`, `/photons/dt`, `/photons/ms`,
and on dual-TDC detectors `/start/time` and `/stop/time`) is stored as a seed
value followed by zigzag varint delta blocks. `x`/`y` are detector positions,
`dt` is the TCSPC micro time, and `ms` is the macro time in milliseconds.

The D7 format is documented at https://github.com/photonscore/d7 (Apache-2.0).
This reader was written from that specification and validated bit-exact against
the Photonscore SDK. No vendor source is redistributed.

Examples
--------
>>> from photonsfile import PhotonsFile
>>> with PhotonsFile('sample.photons') as f:      # doctest: +SKIP
...     ph = f.photons()                          # dict of x, y, dt, ms arrays
...     image = f.image(pixels=512)               # (Y, X) intensity image
...     decay = f.decay(bins=256)                 # TCSPC histogram

"""

import numpy as np

from ._d7 import (
    MAGIC,
    read_header,
    read_attributes,
    read_photons,
    iter_photons,
    dataset_names,
    has_dual_tdc,
    _HAVE_NUMBA,
)

__version__ = '2026.9.29'

__all__ = [
    'PhotonsFile',
    'imread',
    'read_header',
    'read_attributes',
    'read_photons',
    'iter_photons',
    'dataset_names',
    'has_dual_tdc',
    'have_numba',
    'MAGIC',
]

def have_numba():
    """Return True if the varint decode is numba-accelerated."""
    return _HAVE_NUMBA

class PhotonsFile:
    """A Photonscore LINCam `.photons` (D7) file.

    Parameters:
        filename: Name of the `.photons` file to open.

    """

    def __init__(self, filename):
        self.filename = str(filename)
        self.header = read_header(self.filename)
        self.attributes = read_attributes(self.filename)
        self.version = self.header['version']
        self.datasets = [d['name'] for d in self.header['datasets']]
        self.dual_tdc = has_dual_tdc(self.filename)
        self.position_bits = int(self.attributes.get('/photons/PositionBits', 12))
        self.tac_bits = int(self.attributes.get('/photons/TacBits', 12))
        self.position_range = 1 << self.position_bits
        self.tac_range = 1 << self.tac_bits
        self.tac_channel = self.attributes.get('/photons/TacChannel')
        self._photons = None
        self._tac_checked = False

    def photons(self, wanted=('x', 'y', 'dt', 'ms')):
        """Return a dict of per-photon arrays for the requested datasets.

        Keys are the short names (`'x'`, `'y'`, `'dt'`, `'ms'`). On dual-TDC
        detectors `dt` is `stop - start` in picoseconds and `tac_range` is
        updated from the data.
        """
        if self._photons is None:
            self._photons = read_photons(self.filename, wanted)
            if self.dual_tdc and 'dt' in self._photons:
                dt = np.asarray(self._photons['dt'])
                dt = dt[dt >= 0]
                if dt.size:
                    self.tac_range = int(dt.max()) + 1
                self._tac_checked = True
        return self._photons

    def tcspc_resolution(self, bins):
        """Return the TCSPC bin width in seconds for a given number of bins.

        Uses the `/photons/TacChannel` attribute (picoseconds per raw dt unit),
        or 1 ps per unit for dual-TDC detectors. Returns 0.0 if unknown.
        """
        if self.dual_tdc:
            period_s = self.tac_range * 1e-12
        elif self.tac_channel:
            period_s = self.tac_range * float(self.tac_channel) * 1e-12
        else:
            return 0.0
        return period_s / bins if bins else 0.0

    def iter_photons(self, wanted=('x', 'y', 'dt'), chunk_blocks=32):
        """Yield dicts of equal-length x, y, dt arrays without loading the file."""
        return iter_photons(self.filename, wanted, chunk_blocks)

    def _dual_tac_range(self):
        if self._tac_checked == True:
            return
        self._tac_checked = True
        if self._photons is not None and 'dt' in self._photons:
            dt = np.asarray(self._photons['dt'])
            top = int(dt.max()) if dt.size else -1
        else:
            top = -1
            for ch in self.iter_photons(('dt',)):
                if ch['dt'].size:
                    top = max(top, int(ch['dt'].max()))
        if top >= 0:
            self.tac_range = top + 1

    def _binned_chunks(self, pixels, bins, binning):
        if self.dual_tdc == True:
            self._dual_tac_range()
        p = int(pixels)
        b = int(bins)
        for ch in self.iter_photons():
            x = ch['x'].astype(np.int64)
            y = ch['y'].astype(np.int64)
            dt = ch['dt'].astype(np.int64)
            valid = ((x >= 0) & (x < self.position_range)
                     & (y >= 0) & (y < self.position_range)
                     & (dt >= 0) & (dt < self.tac_range))
            xi = (x[valid] * p) // self.position_range
            yi = (y[valid] * p) // self.position_range
            di = (dt[valid] * b) // self.tac_range
            if binning > 1:
                xi //= binning
                yi //= binning
            yield xi, yi, di

    def _grid(self, pixels, binning):
        p = int(pixels)
        if binning > 1:
            p = (p + binning - 1) // binning
        return p

    def image(self, pixels=512, binning=1):
        """Return the `(Y, X)` intensity image, photon positions binned to a grid."""
        p = self._grid(pixels, binning)
        if p == 0:
            return np.zeros((0, 0), dtype=np.uint32)
        out = np.zeros(p * p, dtype=np.uint64)
        for xi, yi, _ in self._binned_chunks(pixels, 1, binning):
            out += np.bincount(yi * p + xi, minlength=p * p).astype(np.uint64)
        return out.reshape(p, p).astype(np.uint32)

    def decay(self, bins=256):
        """Return the summed TCSPC histogram of length `bins`."""
        if self.dual_tdc == True:
            self._dual_tac_range()
        b = int(bins)
        out = np.zeros(b, dtype=np.uint64)
        for ch in self.iter_photons(('dt',)):
            dt = ch['dt'].astype(np.int64)
            dt = dt[(dt >= 0) & (dt < self.tac_range)]
            out += np.bincount((dt * b) // self.tac_range, minlength=b)[:b].astype(np.uint64)
        return out.astype(np.uint32)

    def flim_image(self, pixels=512, bins=256, binning=1):
        """Return the `(Y, X, H)` FLIM cube: intensity image with a TCSPC axis.

        Photons are streamed in chunks, so peak memory is about the size of
        the cube rather than the size of the file.
        """
        p = self._grid(pixels, binning)
        b = int(bins)
        if p == 0 or b == 0:
            return np.zeros((p, p, b), dtype=np.uint32)
        out = np.zeros(p * p * b, dtype=np.uint32)
        for xi, yi, di in self._binned_chunks(pixels, b, binning):
            u, c = np.unique((yi * p + xi) * b + di, return_counts=True)
            out[u] += c.astype(np.uint32)
        return out.reshape(p, p, b)

    def close(self):
        self._photons = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

def imread(filename, pixels=512, binning=1):
    """Read a `.photons` file and return the `(Y, X)` intensity image."""
    with PhotonsFile(filename) as f:
        return f.image(pixels=pixels, binning=binning)
