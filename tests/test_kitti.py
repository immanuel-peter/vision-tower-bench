import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from scripts.prep_kitti import DEPTH_DIVISOR, paired_samples, read_depth, sample_key


def test_kitti_sample_keys_pair_image_and_ground_truth_names():
    image = "2011_09_26_drive_0002_sync_image_0000000005_image_02.png"
    depth = "2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png"

    assert sample_key(Path(image), "image") == sample_key(Path(depth), "groundtruth_depth")


def test_kitti_depth_is_metric_and_keeps_sparse_zeroes(tmp_path):
    encoded = np.array([[0, 256], [512, 1280]], dtype=np.uint16)
    path = tmp_path / "depth.png"
    Image.fromarray(encoded).save(path)

    depth = read_depth(path)

    assert depth.tolist() == [[0.0, 1.0], [2.0, 5.0]]
    assert DEPTH_DIVISOR == 256.0


def test_kitti_prep_rejects_unpaired_files(tmp_path):
    (tmp_path / "image").mkdir()
    (tmp_path / "groundtruth_depth").mkdir()
    Image.new("RGB", (2, 2)).save(
        tmp_path / "image" / "drive_sync_image_0001_image_02.png"
    )

    with pytest.raises(ValueError, match="unpaired"):
        paired_samples(tmp_path)
