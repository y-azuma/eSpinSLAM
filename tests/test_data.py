import csv

import numpy as np
import pytest

from espinslam.data import load_event_sequence, load_events_as_sequence, read_events, read_manifest


def test_npz_manifest_loader_preserves_explicit_test_membership(tmp_path):
    sample = tmp_path / "sample_000000"
    sample.mkdir()
    np.savez_compressed(
        sample / "events.npz",
        x=np.asarray((1, 2, 3)),
        y=np.asarray((4, 5, 6)),
        t=np.asarray((0.0, 0.1, 0.2)),
        p=np.asarray((-1, 1, -1)),
        omega_rad_s=np.asarray(2.0),
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sequence_id", "event_path", "split"))
        writer.writeheader()
        writer.writerow({
            "sequence_id": "debris_000000",
            "event_path": "sample_000000/events.npz",
            "split": "test",
        })
    row = read_manifest(manifest, tmp_path)[0]
    sequence = load_event_sequence(row, tmp_path, (346, 260))
    assert sequence.sequence_id == "debris_000000"
    assert sequence.omega_rad_s == 2.0
    assert sequence.p.tolist() == [0, 1, 0]


def _events():
    t_us = np.asarray([10, 500, 4_000_000, 17_000_000, 33_600_000], dtype=np.int64)
    x = np.asarray([0, 5, 640, 1279, 33])
    y = np.asarray([0, 7, 360, 719, 600])
    p = np.asarray([1, 0, 1, 0, 1])
    return t_us, x, y, p


def _evt3_words(t_us, x, y, p):
    # A camera emits EVT_TIME_HIGH every 4096 us, so 24-bit wraps are visible.
    words, last_block = [], -1
    for ts, xi, yi, pi in zip(t_us, x, y, p):
        block, low = int(ts) >> 12, int(ts) & 0xFFF
        for value in range(last_block + 1, block + 1):
            words.append(0x8000 | (value & 0xFFF))
        last_block = max(last_block, block)
        words += [0x6000 | low, 0x0000 | int(yi), 0x2000 | (int(pi) << 11) | int(xi)]
    # A vectorised group: 3 ON events at x = 100, 102, 111 on row 50 (same time).
    words += [0x0000 | 50, 0x3000 | (1 << 11) | 100, 0x4000 | 0b100000000101]
    return np.asarray(words, dtype="<u2")


def test_prophesee_evt3_raw_decoder(tmp_path):
    t_us, x, y, p = _events()
    path = tmp_path / "scene.raw"
    with path.open("wb") as stream:
        stream.write(b"% evt 3.0\n% format EVT3;height=720;width=1280\n% end\n")
        stream.write(_evt3_words(t_us, x, y, p).tobytes())
    arrays = read_events(path)
    assert arrays.width == 1280 and arrays.height == 720
    assert np.allclose(arrays.t[:5], t_us * 1e-6)  # includes a 24-bit time wrap
    assert arrays.x[:5].tolist() == x.tolist() and arrays.y[:5].tolist() == y.tolist()
    assert arrays.p[:5].tolist() == p.tolist()
    assert arrays.x[5:].tolist() == [100, 102, 111] and set(arrays.y[5:]) == {50}


def test_prophesee_evt2_raw_and_dat_decoders(tmp_path):
    t_us, x, y, p = _events()
    words = []
    for ts, xi, yi, pi in zip(t_us, x, y, p):
        words.append((0x8 << 28) | (int(ts) >> 6))
        words.append((int(pi) << 28) | ((int(ts) & 0x3F) << 22) | (int(xi) << 11) | int(yi))
    raw = tmp_path / "scene_evt2.raw"
    with raw.open("wb") as stream:
        stream.write(b"% format EVT2;height=720;width=1280\n% end\n")
        stream.write(np.asarray(words, dtype="<u4").tobytes())
    arrays = read_events(raw)
    assert np.allclose(arrays.t, t_us * 1e-6) and arrays.x.tolist() == x.tolist()
    dat = tmp_path / "scene.dat"
    with dat.open("wb") as stream:
        stream.write(b"% Width 1280\n% Height 720\n")
        stream.write(bytes((12, 8)))
        data = (x | (y << 14) | (p << 28)).astype("<u4")
        stream.write(np.column_stack((t_us.astype("<u4"), data)).astype("<u4").tobytes())
    arrays = read_events(dat)
    assert (arrays.width, arrays.height) == (1280, 720)
    assert np.allclose(arrays.t, t_us * 1e-6) and arrays.y.tolist() == y.tolist()
    assert arrays.p.tolist() == p.tolist()


@pytest.mark.parametrize("suffix", [".npz", ".csv", ".h5"])
def test_generic_readers_return_seconds_and_binary_polarity(tmp_path, suffix):
    t_us, x, y, p = _events()
    path = tmp_path / f"events{suffix}"
    if suffix == ".npz":
        np.savez(path, t=t_us, x=x, y=y, p=np.where(p > 0, 1, -1))
    elif suffix == ".csv":
        np.savetxt(path, np.column_stack((t_us, x, y, p)), fmt="%d", delimiter=",",
                   header="t,x,y,p", comments="")
    else:
        import h5py
        with h5py.File(path, "w") as handle:
            group = handle.create_group("events")
            for name, value in zip("txyp", (t_us, x, y, p)):
                group[name] = value
    sequence = load_events_as_sequence(path, "scene", width=1280, height=720)
    assert np.allclose(sequence.t, t_us * 1e-6)
    assert sequence.p.tolist() == p.tolist()
    assert sequence.x.dtype == np.int32
