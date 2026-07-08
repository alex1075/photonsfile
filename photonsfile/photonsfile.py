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
    dataset_names,
    has_dual_tdc,
    _HAVE_NUMBA,
)

__version__ = '2026.7.8'

__all__ = [
    'PhotonsFile',
    'imread',
    'read_header',
    'read_attributes',
    'read_photons',
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

    def _binned(self, pixels, binning):
        s = self.photons()
        x = np.asarray(s['x']).astype(np.int64)
        y = np.asarray(s['y']).astype(np.int64)
        dt = np.asarray(s['dt']).astype(np.int64)
        n = min(x.shape[0], y.shape[0], dt.shape[0])
        x, y, dt = x[:n], y[:n], dt[:n]
        valid = ((x >= 0) & (x < self.position_range)
                 & (y >= 0) & (y < self.position_range)
                 & (dt >= 0) & (dt < self.tac_range))
        return x[valid], y[valid], dt[valid]

    def image(self, pixels=512, binning=1):
        """Return the `(Y, X)` intensity image, photon positions binned to a grid."""
        x, y, _ = self._binned(pixels, binning)
        p = int(pixels)
        xi = (x * p) // self.position_range
        yi = (y * p) // self.position_range
        if binning > 1:
            xi //= binning
            yi //= binning
            p = (p + binning - 1) // binning
        if p == 0:
            return np.zeros((0, 0), dtype=np.uint32)
        flat = yi * p + xi
        return np.bincount(flat, minlength=p * p).reshape(p, p).astype(np.uint32)

    def decay(self, bins=256):
        """Return the summed TCSPC histogram of length `bins`."""
        s = self.photons()
        dt = np.asarray(s['dt']).astype(np.int64)
        dt = dt[(dt >= 0) & (dt < self.tac_range)]
        di = (dt * bins) // self.tac_range
        return np.bincount(di, minlength=bins)[:bins].astype(np.uint32)

    def flim_image(self, pixels=512, bins=256, binning=1):
        """Return the `(Y, X, H)` FLIM cube: intensity image with a TCSPC axis."""
        x, y, dt = self._binned(pixels, binning)
        p = int(pixels)
        b = int(bins)
        xi = (x * p) // self.position_range
        yi = (y * p) // self.position_range
        di = (dt * b) // self.tac_range
        if binning > 1:
            xi //= binning
            yi //= binning
            p = (p + binning - 1) // binning
        if p == 0 or b == 0:
            return np.zeros((p, p, b), dtype=np.uint32)
        flat = (yi * p + xi) * b + di
        return np.bincount(flat, minlength=p * p * b).reshape(p, p, b).astype(np.uint32)

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
