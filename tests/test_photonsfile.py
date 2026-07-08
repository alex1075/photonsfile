import os
import glob
import numpy as np
import pytest
import photonsfile
from photonsfile import PhotonsFile, imread, read_header, have_numba

def _find_sample():
    pats = [
        os.path.expanduser('~/Downloads/**/*.photons'),
        os.path.expanduser('~/Downloads/*.photons'),
        os.path.join(os.path.dirname(__file__), 'data', '*.photons'),
    ]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    return None

_SAMPLE = _find_sample()

def test_api_surface():
    for name in ('PhotonsFile', 'imread', 'read_header', 'read_attributes',
                 'read_photons', 'dataset_names', 'has_dual_tdc', 'MAGIC'):
        assert hasattr(photonsfile, name)
    assert isinstance(have_numba(), bool)
    assert photonsfile.MAGIC == b'D7 Photons Data'

@pytest.mark.skipif(_SAMPLE is None, reason='no .photons sample present')
def test_read_real_sample():
    f = PhotonsFile(_SAMPLE)
    assert f.header['magic'] == 'D7 Photons Data'
    assert f.position_range >= 2
    assert isinstance(f.attributes, dict)

    ph = f.photons()
    assert 'x' in ph and 'y' in ph and 'dt' in ph
    n = min(len(ph['x']), len(ph['y']), len(ph['dt']))
    assert n > 0

    img = f.image(pixels=256)
    assert img.shape == (256, 256)
    assert int(img.sum()) > 0

    decay = f.decay(bins=256)
    assert decay.shape == (256,)
    assert int(decay.sum()) > 0

    cube = f.flim_image(pixels=128, bins=256)
    assert cube.shape == (128, 128, 256)
    assert int(cube.sum()) > 0
    assert int(cube.sum(axis=2).sum()) == int(f.image(pixels=128).sum())

    res = f.tcspc_resolution(256)
    assert res >= 0.0
    f.close()

@pytest.mark.skipif(_SAMPLE is None, reason='no .photons sample present')
def test_imread_matches_method():
    img = imread(_SAMPLE, pixels=128)
    assert img.shape == (128, 128)
    assert np.array_equal(img, PhotonsFile(_SAMPLE).image(pixels=128))
