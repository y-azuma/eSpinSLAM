"""Run the pre-change code (4b2a1a9) on TOPSPIN scenes, frequency + reconstruction."""
import json, math, sys, time, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
OLD = sys.argv[1]; sys.path.insert(0, OLD)
from espinslam.config import load_config
from espinslam.data import EventSequence
from espinslam.frontend import EventFrontend
from espinslam.geometry import reconstruct
from espinslam.pipeline import ESpinSLAM
root = Path("/home/user/topspin"); out = Path(sys.argv[2]); scenes = sys.argv[3:]
base = load_config(Path(OLD) / "examples/configs/debris_davis346.jsonc")
base.data.sensor_size = [1280, 720]
VARIANTS = {"as_is": (0.10, 1.00, 91),            # old defaults (0.01 Hz grid)
            "widened": (1.0 / 60.0, 0.5, 97)}     # TOPSPIN periods 2-60 s (0.005 Hz grid)
done = {json.loads(l)["scene"] for l in out.open()} if out.exists() else set()
for name in scenes:
    if name in done: continue
    d = np.load(root / name / "events.npz"); calib = [float(v) for v in (root / name / "calib.txt").read_text().split()]
    base.camera.fx, base.camera.fy, base.camera.cx, base.camera.cy = calib[:4]
    seq = EventSequence(name, d["x"].astype(np.int32), d["y"].astype(np.int32), d["t"].astype(np.float64) * 1e-6,
                        d["p"].astype(np.uint8), math.nan, 1280, 720, {}, root / name)
    t0 = time.perf_counter(); tracks = EventFrontend(1280, 720, base.frontend).process(seq); t_fe = time.perf_counter() - t0
    rec_out = {"scene": name, "events": seq.event_count, "frontend_s": t_fe, "tracks": len(tracks)}
    for key, (lo, hi, n) in VARIANTS.items():
        base.frequency.min_hz, base.frequency.max_hz, base.frequency.candidate_count = lo, hi, n
        model = ESpinSLAM(base); fr = model.estimate_frequency(seq, tracks=tracks)
        rec = reconstruct(tracks, fr.frequency_hz, base.camera.matrix(), base.backend) if fr.valid else None
        rec_out[key] = {"frequency_hz": fr.frequency_hz if fr.valid else None,
                        "landmarks": int(rec.points.shape[0]) if rec else 0,
                        "reprojection_px": float(rec.mean_reprojection_error_px) if rec and rec.valid else None}
    rec_out["total_s"] = time.perf_counter() - t0
    with out.open("a") as fh: fh.write(json.dumps(rec_out) + "\n")
    print(json.dumps(rec_out), flush=True)
