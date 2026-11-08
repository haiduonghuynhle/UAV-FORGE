"""UAV-VisLoc CSV pose columns: Omega=pitch, Kappa=roll, Phi1/Phi2=yaw (README_dataset.txt)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

import pytest

from data.dataset_adapters import UAVVisLocDataset


@pytest.fixture
def visloc_root(tmp_path):
    loc = tmp_path / "01"
    (loc / "drone").mkdir(parents=True)
    (loc / "drone" / "01_0001.JPG").write_bytes(b"")
    (loc / "01.csv").write_text(
        "num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2\n"
        "1,01_0001.JPG,2018-09-17T12:44:21,29.76096029,115.9747973,405.76,"
        "2.251724,-0.27502,163.877389,178.527919\n"
    )
    return tmp_path


def test_camera_intrinsics_centered_on_real_image(visloc_root):
    import cv2
    import numpy as np

    img_path = visloc_root / "01" / "drone" / "01_0001.JPG"
    cv2.imwrite(str(img_path), np.zeros((30, 40, 3), dtype=np.uint8))

    ds = UAVVisLocDataset(data_root=Path(visloc_root), locations=["01"])
    ds.load()
    cam = ds._get_frame(0)["camera"]
    assert (cam["width"], cam["height"]) == (40, 30)
    assert (cam["cx"], cam["cy"]) == (20.0, 15.0)
    assert cam["fx"] == pytest.approx(405.76 / 0.15)
    assert cam["fy"] == cam["fx"]


def test_pose_columns_mapped_per_dataset_readme(visloc_root):
    ds = UAVVisLocDataset(data_root=Path(visloc_root), locations=["01"])
    ds.load()
    assert len(ds._entries) == 1
    pose = ds._entries[0]["gt_pose"]
    assert pose["yaw"] == pytest.approx(163.877389)
    assert pose["pitch"] == pytest.approx(2.251724)
    assert pose["roll"] == pytest.approx(-0.27502)
    assert pose["pitch_convention"] == "aircraft_nadir_zero"
