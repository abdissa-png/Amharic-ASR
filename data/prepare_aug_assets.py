"""Write a tiny synthetic noise/RIR bank so augmentation has files on disk.

Real MUSAN / OpenSLR RIR files can replace these later; the trainer falls
back to on-the-fly synthesis if the folders are empty anyway.

    python -m data.prepare_aug_assets
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import soundfile as sf

from data.augment import SAMPLE_RATE, colored_noise, synthetic_rir
from utils import ROOT, setup_logging

log = setup_logging()


def write_bank(
    *,
    noise_dir: Path,
    rir_dir: Path,
    n_noise: int = 24,
    n_rir: int = 16,
    seconds: float = 4.0,
    seed: int = 0,
) -> dict[str, int]:
    rng = random.Random(seed)
    noise_dir.mkdir(parents=True, exist_ok=True)
    rir_dir.mkdir(parents=True, exist_ok=True)
    n = int(seconds * SAMPLE_RATE)
    for i in range(n_noise):
        beta = (0.0, 1.0, 1.5)[i % 3]
        wav = colored_noise(n, rng, beta=beta).numpy()
        sf.write(str(noise_dir / f"synth_{i:03d}_b{beta:.1f}.flac"), wav, SAMPLE_RATE, format="FLAC")
    for i in range(n_rir):
        wav = synthetic_rir(rng, SAMPLE_RATE).numpy()
        sf.write(str(rir_dir / f"synth_rir_{i:03d}.flac"), wav, SAMPLE_RATE, format="FLAC")
    log.info("wrote %d noise clips → %s", n_noise, noise_dir)
    log.info("wrote %d rirs → %s", n_rir, rir_dir)
    return {"noise": n_noise, "rir": n_rir}


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--noise-dir", type=Path, default=ROOT / "data" / "noise")
    p.add_argument("--rir-dir", type=Path, default=ROOT / "data" / "rir")
    p.add_argument("--n-noise", type=int, default=24)
    p.add_argument("--n-rir", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(list(argv) if argv is not None else None)
    write_bank(
        noise_dir=args.noise_dir,
        rir_dir=args.rir_dir,
        n_noise=args.n_noise,
        n_rir=args.n_rir,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
