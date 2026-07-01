"""Generate a tiny ASVspoof2019-LA-layout dataset to smoke-test the training pipeline."""
import os, sys
import numpy as np
import soundfile as sf

root = sys.argv[1] if len(sys.argv) > 1 else "data/smoke"
SR = 16000
rng = np.random.default_rng(0)

def synth(label, t):
    if label == "bonafide":                      # harmonic, speech-ish
        f0 = rng.uniform(100, 200)
        y = sum(np.sin(2*np.pi*f0*k*t)/k for k in (1, 2, 3)) + 0.05*rng.standard_normal(t.size)
    else:                                        # buzzy / noisy -> spoof-ish
        y = 0.3*rng.standard_normal(t.size) + np.sin(2*np.pi*rng.uniform(200, 400)*t)
    return 0.9 * y / (np.max(np.abs(y)) + 1e-9)

def make(subset, proto_name, n_bona, n_spoof):
    ad = os.path.join(root, f"ASVspoof2019_LA_{subset}", "flac"); os.makedirs(ad, exist_ok=True)
    pd = os.path.join(root, "ASVspoof2019_LA_cm_protocols"); os.makedirs(pd, exist_ok=True)
    lines, idx = [], 0
    for label, n in (("bonafide", n_bona), ("spoof", n_spoof)):
        for _ in range(n):
            utt = f"LA_{subset[0].upper()}_{idx:07d}"; idx += 1
            dur = rng.uniform(1.5, 4.0)
            t = np.linspace(0, dur, int(dur*SR), endpoint=False)
            sf.write(os.path.join(ad, f"{utt}.flac"), synth(label, t).astype(np.float32), SR, subtype="PCM_16")
            sysid = "-" if label == "bonafide" else f"A{int(rng.integers(1,20)):02d}"
            lines.append(f"LA_00{idx%9} {utt} - {sysid} {label}")
    with open(os.path.join(pd, proto_name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  {subset}: {n_bona} bona + {n_spoof} spoof -> {proto_name}")

make("train", "ASVspoof2019.LA.cm.train.trn.txt", 30, 30)
make("dev",   "ASVspoof2019.LA.cm.dev.trl.txt",   20, 20)
print("smoke data ready at", root)
