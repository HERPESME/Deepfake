#!/usr/bin/env python3
"""Copy regenerated Grad-CAM grids to BTP and papers directories."""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC_CELEBD = ROOT / "experiments" / "phase6_gradcam" / "celebd_comparisons"
SRC_FF = ROOT / "experiments" / "phase6_gradcam" / "ff_comparisons"

DESTINATIONS = [
    ROOT / "LNMIIT BTP Report Template" / "Figures",
    ROOT / "papers" / "bridging_the_gap" / "figures",
]

MAPPING = {
    "gradcam_celebd.png":      SRC_CELEBD / "fake_5.png",
    "gradcam_ff.png":          SRC_FF / "real_0.png",
    "gradcam_celebd_real.png": SRC_CELEBD / "real_0.png",
    "gradcam_ff_fake.png":     SRC_FF / "fake_5.png",
}

for dest_name, src_path in MAPPING.items():
    if not src_path.exists():
        print(f"  ✗ Source missing: {src_path}")
        continue
    for dest_dir in DESTINATIONS:
        dest_path = dest_dir / dest_name
        shutil.copy2(str(src_path), str(dest_path))
        size_kb = dest_path.stat().st_size / 1024
        print(f"  ✓ {dest_name} -> {dest_dir.name}/ ({size_kb:.0f} KB)")

print("\nDone! All 4 images copied to both directories.")
