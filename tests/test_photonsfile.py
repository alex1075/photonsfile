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

def _uv(n):
    out = bytearray()
    while True:
        b = n & 0x7f
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)

def _ld(tag, payload):
    return bytes([tag]) + _uv(len(payload)) + payload

def _zz(n):
    return (n << 1) if n >= 0 else ((-n << 1) - 1)

def _append(f, entry, page=16384):
    # frame one FileEntry like the D7 LogAppender, returning its offset
    start = len(f)
    begin = True
    d = entry
    while len(d) > 0:
        left = page - len(f) % page
        if left < 2:
            f += b'\x00' * left
        avail = page - len(f) % page - 2
        n = min(len(d), avail)
        h = n | (0x8000 if begin else 0) | (0x4000 if n == len(d) else 0)
        f += bytes([h & 0xff, h >> 8]) + d[:n]
        d = d[n:]
        begin = False
    return start

def _framed_file(path, n_photons, filler=17000):
    rng = np.random.default_rng(1)
    names = ['/photons/x', '/photons/y', '/photons/dt']
    toc = b''
    for nm in names:
        toc += _ld(0x3a, _ld(0x0a, nm.encode()) + bytes([0x10]) + _uv(5))
    f = bytearray()
    _append(f, _ld(0x0a, _ld(0x0a, b'D7 Photons Data') + bytes([0x10]) + _uv(1) + toc))
    data = {nm: rng.integers(0, 4096, n_photons) for nm in names}
    index = b''
    for did, nm in enumerate(names):
        v = data[nm]
        for k in range(0, len(v), 3000):
            blk = v[k:k + 3000]
            packed = b''.join(_uv(_zz(int(x))) for x in np.diff(blk))
            msg = bytes([0x08]) + _uv(did) + bytes([0x18]) + _uv(_zz(int(blk[0]))) + _ld(0x22, packed)
            off = _append(f, _ld(0x12, msg))
            index += _ld(0x0a, bytes([0x08]) + _uv(did) + bytes([0x10]) + _uv(off))
    attrs = {'/photons/Filler': 'z' * filler, '/photons/TacChannel': '12.3',
             '/photons/PositionBits': '12', '/photons/TacBits': '12'}
    for k, v in attrs.items():
        index += _ld(0x12, _ld(0x0a, k.encode()) + _ld(0x12, v.encode()))
    index_off = _append(f, _ld(0x1a, index))
    _append(f, _ld(0x22, bytes([0x08]) + _uv(index_off) + _ld(0x12, b'End of D7 Photons Data File')))
    open(path, 'wb').write(bytes(f))
    return data, f[index_off]

def test_index_read_past_fragment_header(tmp_path):
    # issue #1: the index only parsed when its fragment header's low byte was >= 0x80
    path = str(tmp_path / 'framed.photons')
    lows = set()
    filler = 17000
    for n_photons in (20000, 20050, 20100, 20150, 20200, 20250):
        data, low = _framed_file(path, n_photons, filler)
        lows.add(low >= 0x80)
        attrs = photonsfile.read_attributes(path)
        assert attrs['/photons/TacChannel'] == '12.3'
        assert len(attrs['/photons/Filler']) == filler
        ph = photonsfile.read_photons(path, ('x', 'y', 'dt'))
        for k in ('x', 'y', 'dt'):
            assert np.array_equal(ph[k], data['/photons/' + k])
        got = np.concatenate([c['x'] for c in photonsfile.iter_photons(path, chunk_blocks=2)])
        assert np.array_equal(got, data['/photons/x'])
        assert PhotonsFile(path).tcspc_resolution(256) > 0
    assert lows == {True, False}
