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

def test_clean_pos_page_aligned_start():
    from photonsfile._d7 import _clean_pos, _strip_markers
    page = 16
    raw = bytes(range(64))
    for start in (0, 1, 2, 5, 16, 17, 18):
        clean = _strip_markers(raw[start:], start, page)
        for phys in range(start, 64):
            if phys % page < 2:
                continue
            assert clean[_clean_pos(phys, start, page)] == raw[phys]

@pytest.mark.skipif(_SAMPLE is None, reason='no .photons sample present')
def test_iter_photons_matches_read_photons():
    full = photonsfile.read_photons(_SAMPLE, ('x', 'y', 'dt'))
    n = min(len(full['x']), len(full['y']), len(full['dt']))
    parts = {'x': [], 'y': [], 'dt': []}
    sizes = []
    for ch in photonsfile.iter_photons(_SAMPLE, chunk_blocks=7):
        sizes.append(len(ch['x']))
        assert len(ch['x']) == len(ch['y']) == len(ch['dt'])
        for k in parts:
            parts[k].append(ch[k])
    assert len(sizes) > 1
    for k in parts:
        got = np.concatenate(parts[k])
        assert len(got) == n
        assert np.array_equal(got, full[k][:n])

@pytest.mark.skipif(_SAMPLE is None, reason='no .photons sample present')
def test_streamed_images_match_in_memory():
    f = PhotonsFile(_SAMPLE)
    ph = f.photons()
    x = np.asarray(ph['x']).astype(np.int64)
    y = np.asarray(ph['y']).astype(np.int64)
    dt = np.asarray(ph['dt']).astype(np.int64)
    n = min(len(x), len(y), len(dt))
    x, y, dt = x[:n], y[:n], dt[:n]
    ok = ((x >= 0) & (x < f.position_range) & (y >= 0) & (y < f.position_range)
          & (dt >= 0) & (dt < f.tac_range))
    x, y, dt = x[ok], y[ok], dt[ok]
    p, b = 128, 64
    xi = (x * p) // f.position_range
    yi = (y * p) // f.position_range
    di = (dt * b) // f.tac_range
    want = np.bincount((yi * p + xi) * b + di, minlength=p * p * b).reshape(p, p, b)
    f.close()
    g = PhotonsFile(_SAMPLE)
    cube = g.flim_image(pixels=p, bins=b)
    assert g._photons is None
    assert np.array_equal(cube, want)
    assert np.array_equal(g.image(pixels=p), want.sum(axis=2))
    dt_all = np.asarray(ph['dt']).astype(np.int64)
    dt_all = dt_all[(dt_all >= 0) & (dt_all < g.tac_range)]
    assert np.array_equal(g.decay(bins=b), np.bincount((dt_all * b) // g.tac_range, minlength=b)[:b])
    assert np.array_equal(g.flim_image(pixels=p, bins=b, binning=2).sum(axis=2),
                          g.image(pixels=p, binning=2))
