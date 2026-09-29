import numpy as np

try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

MAGIC = b'D7 Photons Data'
_DEFAULT_PAGE = 16384
_MAX_FRAGMENT = 16 * 1024 * 1024

_DTYPES = {
    0: np.int8, 1: np.int16, 2: np.int32, 3: np.int64,
    4: np.uint8, 5: np.uint16, 6: np.uint32, 7: np.uint64,
    8: np.single, 9: np.double,
}

def _read_varint(buf, i):
    shift = 0
    val = 0
    while True:
        c = buf[i]
        i += 1
        val |= (c & 0x7f) << shift
        if not (c & 0x80):
            return val, i
        shift += 7

def _zigzag(v):
    return (v >> 1) ^ -(v & 1)

def _strip_markers(raw, phys_start, page):
    b = np.frombuffer(raw, dtype=np.uint8)
    n = b.shape[0]
    first = (-phys_start) % page
    keep = np.ones(n, dtype=bool)
    if phys_start % page == 1 and n > 0:
        keep[0] = False
    if first >= n:
        return b[keep]
    m0 = np.arange(first, n, page)
    keep[m0] = False
    m1 = m0[m0 + 1 < n] + 1
    keep[m1] = False
    return b[keep]

def _marker_bytes_before(pos, page):
    return 2 * (pos // page) + min(pos % page, 2)

def _clean_pos(phys, region_start, page):
    markers = _marker_bytes_before(phys, page) - _marker_bytes_before(region_start, page)
    return (phys - region_start) - markers

def _parse_descriptor(payload):
    name = None
    code = None
    k = 0
    while k < len(payload):
        tag = payload[k]
        k += 1
        field = tag >> 3
        wt = tag & 7
        if wt == 2:
            ln, k = _read_varint(payload, k)
            chunk = payload[k:k + ln]
            k += ln
            if field == 1:
                name = chunk.decode('ascii', 'replace')
        elif wt == 0:
            val, k = _read_varint(payload, k)
            if field == 2:
                code = val
        else:
            break
    return name, _DTYPES.get(code, None)

def read_header(path):
    with open(path, 'rb') as fh:
        head = fh.read(_DEFAULT_PAGE)
    if MAGIC not in head[:64]:
        raise ValueError(f'Not a D7 .photons file (magic {MAGIC!r} not found): {path}')
    i = 2
    tag, i = _read_varint(head, i)
    hlen, i = _read_varint(head, i)
    end = i + hlen
    header = {'magic': MAGIC.decode(), 'version': None, 'index_step': None,
              'page_size': _DEFAULT_PAGE, 'datasets': []}
    while i < end:
        tag, i = _read_varint(head, i)
        field = tag >> 3
        wt = tag & 7
        if wt == 2:
            ln, i = _read_varint(head, i)
            payload = head[i:i + ln]
            i += ln
            if field == 7:
                name, dtype = _parse_descriptor(payload)
                header['datasets'].append({'name': name, 'dtype': dtype})
        elif wt == 0:
            val, i = _read_varint(head, i)
            if field == 2:
                header['version'] = val
            elif field == 6:
                header['index_step'] = val
            elif field == 8 and val > 0:
                header['page_size'] = val
        else:
            break
    header['header_end'] = end
    return header

def dataset_names(path):
    return [d['name'] for d in read_header(path)['datasets']]

def _read_epilogue(fh, filesize, page):
    n = min(filesize, 8192)
    fh.seek(filesize - n)
    tail = _strip_markers(fh.read(n), filesize - n, page).tobytes()
    e = tail.rfind(b'End of D7 Photons Data File')
    if e < 0:
        raise ValueError('D7 epilogue signature not found')
    j = e - 3
    while j - 1 >= 0 and (tail[j - 1] & 0x80):
        j -= 1
    giof, _ = _read_varint(tail, j)
    return giof

def _read_entry(fh, offset, filesize, page):
    # reassemble one FileEntry from its fragments, as the D7 LogReader does:
    # each fragment has a 2-byte little-endian header (bits 0-13 length,
    # bit 14 end, bit 15 begin); a lone spare byte at a block end is padding
    fh.seek(offset)
    raw = fh.read(min(filesize - offset, _MAX_FRAGMENT + _MAX_FRAGMENT // (page - 2) * 2 + page))
    out = bytearray()
    i = 0
    in_block = offset % page
    found = False
    while len(out) < _MAX_FRAGMENT:
        if page - in_block == 1:
            i += 1
            in_block = 0
        if i + 2 > len(raw):
            raise ValueError('D7 entry at ' + str(offset) + ' is truncated')
        h = raw[i] | (raw[i + 1] << 8)
        i += 2
        ln = h & 0x3fff
        begin = (h & 0x8000) != 0
        end = (h & 0x4000) != 0
        if found == True and begin == True:
            raise ValueError('D7 entry at ' + str(offset) + ' has a second begin fragment')
        if in_block + 2 + ln > page:
            raise ValueError('D7 fragment at ' + str(offset + i - 2) + ' crosses a block boundary')
        in_block = (in_block + 2 + ln) % page
        if found == False and begin == False:
            i += ln
            continue
        found = True
        if i + ln > len(raw):
            raise ValueError('D7 entry at ' + str(offset) + ' is truncated')
        out += raw[i:i + ln]
        i += ln
        if end == True:
            return bytes(out)
    raise ValueError('D7 entry at ' + str(offset) + ' exceeds the 16 MB fragment limit')

def _index_body(fh, index_off, filesize, page):
    entry = _read_entry(fh, index_off, filesize, page)
    _tag, p = _read_varint(entry, 0)
    ln, p = _read_varint(entry, p)
    return entry[p:p + ln]

def _parse_index(fh, index_off, filesize, page, n_datasets):
    body = _index_body(fh, index_off, filesize, page)
    offsets = {}
    i = 0
    n = len(body)
    while i < n - 2:
        if body[i] == 0x0a:
            l = body[i + 1]
            pay = body[i + 2:i + 2 + l]
            if len(pay) >= 3 and pay[0] == 0x08 and pay[1] < n_datasets and pay[2] == 0x10:
                did = pay[1]
                off, _ = _read_varint(pay, 3)
                offsets.setdefault(did, []).append(off)
                i += 2 + l
                continue
        i += 1
    for did in offsets:
        offsets[did].sort()
    return offsets

def read_attributes(path):
    import os
    header = read_header(path)
    page = header['page_size']
    filesize = os.path.getsize(path)
    with open(path, 'rb') as fh:
        index_off = _read_epilogue(fh, filesize, page)
        body = _index_body(fh, index_off, filesize, page)
    attrs = {}
    i = 0
    n = len(body)
    while i < n:
        tag, i = _read_varint(body, i)
        f = tag >> 3
        wt = tag & 7
        if wt == 2:
            l, i = _read_varint(body, i)
            ent = body[i:i + l]
            i += l
            if f == 2:
                key = val = None
                k = 0
                while k < len(ent):
                    etag, k = _read_varint(ent, k)
                    if etag & 7 != 2:
                        break
                    el, k = _read_varint(ent, k)
                    s = ent[k:k + el].decode('utf8', 'replace')
                    k += el
                    if etag >> 3 == 1:
                        key = s
                    elif etag >> 3 == 2:
                        val = s
                if key:
                    attrs[key] = val
        elif wt == 0:
            _, i = _read_varint(body, i)
        else:
            break
    return attrs

if _HAVE_NUMBA:
    @_njit(cache=True)
    def _decode_varints_zigzag_nb(b):
        n = b.shape[0]
        out = np.empty(n, dtype=np.int64)
        m = 0
        val = np.uint64(0)
        shift = np.uint64(0)
        for i in range(n):
            c = b[i]
            val |= np.uint64(c & 0x7f) << shift
            if c & 0x80:
                shift += np.uint64(7)
            else:
                out[m] = np.int64(val >> np.uint64(1)) ^ -np.int64(val & np.uint64(1))
                m += 1
                val = np.uint64(0)
                shift = np.uint64(0)
        return out[:m]

def _decode_varints_zigzag_np(b):
    cont = (b & 0x80) != 0
    term = ~cont
    payload = (b & 0x7f).astype(np.uint64)
    n = b.shape[0]
    newgrp = np.empty(n, dtype=bool)
    newgrp[0] = True
    newgrp[1:] = term[:-1]
    starts = np.flatnonzero(newgrp)
    within = (np.arange(n) - np.repeat(starts, np.diff(np.append(starts, n)))).astype(np.uint64)
    contrib = payload << (np.uint64(7) * within)
    summed = np.add.reduceat(contrib, starts)
    return (summed >> np.uint64(1)).astype(np.int64) ^ -(summed & np.uint64(1)).astype(np.int64)

def _decode_varints_zigzag(b):
    if b.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if _HAVE_NUMBA:
        return _decode_varints_zigzag_nb(b)
    return _decode_varints_zigzag_np(b)

def _decode_data_block(buf, cp, n, n_datasets):
    q = cp - 1
    while True:
        q = buf.find(b'\x12', q + 1, cp + 16)
        if q < 0:
            return None
        msg_len, k = _read_varint(buf, q + 1)
        if (k + 2 < n and buf[k] == 0x08 and buf[k + 1] < n_datasets
                and buf[k + 2] in (0x18, 0x22, 0x2a)):
            break
    msg = buf[k:k + msg_len]
    seed = 0
    src = None
    m = 0
    while m < len(msg):
        tag, m = _read_varint(msg, m)
        f = tag >> 3
        wt = tag & 7
        if wt == 0:
            v, m = _read_varint(msg, m)
            if f == 3:
                seed = _zigzag(v)
        elif wt == 2:
            l, m = _read_varint(msg, m)
            if f in (4, 5):
                src = msg[m:m + l]
            m += l
        else:
            break
    if src is None:
        return None
    deltas = _decode_varints_zigzag(np.frombuffer(src, dtype=np.uint8))
    vals = np.empty(deltas.shape[0] + 1, dtype=np.int64)
    vals[0] = seed
    np.cumsum(deltas, out=vals[1:])
    vals[1:] += seed
    return vals

def _decode_region(fh, offsets, region_start, region_end, page, n_datasets):
    fh.seek(region_start)
    clean = _strip_markers(fh.read(region_end - region_start), region_start, page)
    buf = clean.tobytes()
    n = len(buf)
    parts = []
    for phys in offsets:
        vals = _decode_data_block(buf, _clean_pos(phys, region_start, page), n, n_datasets)
        if vals is not None:
            parts.append(vals)
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)

def _block_groups(fh, offs, all_offs, index_off, page, n_datasets, chunk_blocks):
    # blocks of one dataset are not always adjacent (the final partial block
    # sits at the end of the file), so read each adjacent run on its own
    import bisect
    for i in range(0, len(offs), chunk_blocks):
        grp = offs[i:i + chunk_blocks]
        parts = []
        run = [grp[0]]
        for o in grp[1:] + [None]:
            j = bisect.bisect_right(all_offs, run[-1])
            nxt = all_offs[j] if j < len(all_offs) else index_off
            if o is not None and o == nxt:
                run.append(o)
                continue
            parts.append(_decode_region(fh, run, run[0], min(nxt, index_off), page, n_datasets))
            run = [o]
        yield np.concatenate(parts) if len(parts) > 1 else parts[0]

def iter_photons(path, wanted=('x', 'y', 'dt'), chunk_blocks=32):
    """Yield dicts of equal-length per-photon arrays, a few blocks at a time.

    Only x, y and dt are per-photon (ms is a per-millisecond photon index), so
    only those can be streamed. Peak memory is set by chunk_blocks, not by the
    file size; the concatenated chunks equal read_photons() truncated to the
    shortest dataset.
    """
    for name in wanted:
        if name not in ('x', 'y', 'dt'):
            raise ValueError('iter_photons streams x, y and dt only, not ' + repr(name))
    import os
    header = read_header(path)
    page = header['page_size']
    datasets = header['datasets']
    n_datasets = len(datasets)
    id_by_name = {d['name']: i for i, d in enumerate(datasets)}
    dtype_by_id = {i: d['dtype'] for i, d in enumerate(datasets)}
    filesize = os.path.getsize(path)
    dual = '/start/time' in id_by_name and '/stop/time' in id_by_name
    with open(path, 'rb') as fh:
        index_off = _read_epilogue(fh, filesize, page)
        offsets = _parse_index(fh, index_off, filesize, page, n_datasets)
        all_offs = sorted(o for v in offsets.values() for o in v)
        sources = {}
        for name in wanted:
            if name == 'dt' and dual:
                fulls = ['/start/time', '/stop/time']
            else:
                fulls = ['/photons/' + name]
            if all(id_by_name.get(f) in offsets for f in fulls):
                sources[name] = fulls
        needed = []
        for fulls in sources.values():
            for f in fulls:
                if f not in needed:
                    needed.append(f)
        if len(needed) == 0:
            return
        gens = {}
        for f in needed:
            gens[f] = _block_groups(fh, offsets[id_by_name[f]], all_offs, index_off,
                                    page, n_datasets, chunk_blocks)
        bufs = {f: np.zeros(0, dtype=np.int64) for f in needed}
        done = {f: False for f in needed}
        def pull(f):
            try:
                bufs[f] = np.concatenate((bufs[f], next(gens[f])))
            except StopIteration:
                done[f] = True
        while True:
            for f in needed:
                if done[f] == False:
                    pull(f)
            top = max(len(b) for b in bufs.values())
            for f in needed:
                while len(bufs[f]) < top and done[f] == False:
                    pull(f)
            n = min(len(b) for b in bufs.values())
            if n == 0:
                return
            out = {}
            for name, fulls in sources.items():
                if len(fulls) == 2:
                    out[name] = bufs[fulls[1]][:n] - bufs[fulls[0]][:n]
                else:
                    did = id_by_name[fulls[0]]
                    out[name] = bufs[fulls[0]][:n].astype(dtype_by_id.get(did) or np.int64, copy=False)
            for f in needed:
                bufs[f] = bufs[f][n:]
            yield out

def has_dual_tdc(path):
    names = set(dataset_names(path))
    return '/start/time' in names and '/stop/time' in names

def read_photons(path, wanted=('x', 'y', 'dt', 'ms')):
    header = read_header(path)
    page = header['page_size']
    datasets = header['datasets']
    n_datasets = len(datasets)
    id_by_name = {d['name']: i for i, d in enumerate(datasets)}
    dtype_by_id = {i: d['dtype'] for i, d in enumerate(datasets)}
    import os
    filesize = os.path.getsize(path)
    with open(path, 'rb') as fh:
        index_off = _read_epilogue(fh, filesize, page)
        offsets = _parse_index(fh, index_off, filesize, page, n_datasets)
        def decode(full):
            did = id_by_name.get(full)
            if did is None or did not in offsets or not offsets[did]:
                return None
            v = offsets[did]
            start, end = min(v), min(index_off, max(v) + 65536)
            vals = _decode_region(fh, v, start, end, page, n_datasets)
            return vals.astype(dtype_by_id.get(did) or np.int64, copy=False)
        # newer LINCam files carry two TDCs: dt = photon /stop/time - laser /start/time
        dual = '/start/time' in id_by_name and '/stop/time' in id_by_name
        out = {}
        for name in wanted:
            if name == 'dt' and dual:
                start_t = decode('/start/time')
                stop_t = decode('/stop/time')
                if start_t is not None and stop_t is not None:
                    m = min(start_t.shape[0], stop_t.shape[0])
                    out['dt'] = (stop_t[:m].astype(np.int64) - start_t[:m].astype(np.int64))
                    continue
            vals = decode('/photons/' + name)
            if vals is not None:
                out[name] = vals
    return out
