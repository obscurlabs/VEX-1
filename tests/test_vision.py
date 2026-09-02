"""Phase 0: detection, embedding, validation and quality behaviour."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import CONFIG
from src.models import FaceStatus, ImageStatus
from src.vision import quality
from src.vision.embedder import EMBEDDING_DIM, ArcFaceEmbedder, l2_normalize


# --- image validation -------------------------------------------------

def test_missing_file_reports_not_found():
    status, img, err = quality.load_image("does/not/exist.jpg")
    assert status is ImageStatus.NOT_FOUND
    assert img is None and "no such file" in err


def test_undecodable_bytes_report_invalid_image():
    status, img, err = quality.decode_bytes(b"this is not an image")
    assert status is ImageStatus.INVALID_IMAGE
    assert img is None and err


def test_empty_body_reports_invalid_image():
    status, _, err = quality.decode_bytes(b"")
    assert status is ImageStatus.INVALID_IMAGE
    assert "empty" in err


def test_low_resolution_image_is_rejected(portrait):
    tiny = cv2.resize(portrait, (32, 32))
    status, img, err = quality.validate_array(tiny)
    assert status is ImageStatus.TOO_SMALL
    assert img is None
    assert str(CONFIG.vision.min_image_dim) in err


def test_valid_image_passes_validation(portrait):
    status, img, err = quality.validate_array(portrait)
    assert status is ImageStatus.OK
    assert img is not None and err is None


# --- quality metrics --------------------------------------------------

def test_blur_lowers_laplacian_variance(portrait):
    sharp = quality.assess(portrait)
    blurred = quality.assess(cv2.GaussianBlur(portrait, (31, 31), 0))
    assert blurred.blur_variance < sharp.blur_variance
    assert blurred.is_blurry


def test_crop_is_clamped_to_image_bounds(portrait):
    h, w = portrait.shape[:2]
    crop = quality.crop_face(portrait, (-50, -50, w + 500, h + 500))
    assert crop.shape[0] <= h and crop.shape[1] <= w


# --- detection --------------------------------------------------------

def test_detects_single_face_in_portrait(embedder, portrait):
    result = embedder.process_image(portrait)
    assert result.image_status is ImageStatus.OK
    assert result.face_status is FaceStatus.OK
    assert result.faces_detected >= 1
    face = result.faces[0]
    assert 0.0 < face.det_score <= 1.0
    assert face.det_score >= CONFIG.vision.min_det_score
    x1, y1, x2, y2 = face.bbox
    assert x2 > x1 and y2 > y1


def test_no_face_image_reports_no_face(embedder, no_face_image):
    result = embedder.process_image(no_face_image)
    assert result.image_status is ImageStatus.OK
    assert result.face_status is FaceStatus.NO_FACE
    assert result.faces_detected == 0
    assert result.embeddings == []


def test_detects_every_face_in_group_photo(embedder, group_image):
    result = embedder.process_image(group_image)
    assert result.face_status is FaceStatus.OK
    assert result.faces_detected >= 4, "group photo should yield several faces"
    assert len(result.embeddings) == result.faces_detected


def test_faces_are_ordered_largest_first(embedder, group_image):
    faces = embedder.detector.detect(group_image)
    areas = [f.area for f in faces]
    assert areas == sorted(areas, reverse=True)
    assert [f.index for f in faces] == list(range(len(faces)))


def test_tiny_face_is_rejected_as_low_quality(embedder, portrait):
    """A face shrunk below the pixel floor must not produce an embedding."""
    small = cv2.resize(portrait, None, fx=0.18, fy=0.18)
    status, img, _ = quality.validate_array(small)
    if status is not ImageStatus.OK:
        pytest.skip("image fell below the resolution floor first")
    result = embedder.process_image(img)
    assert result.face_status in (FaceStatus.NO_FACE, FaceStatus.LOW_QUALITY)
    assert result.embeddings == []


# --- embedding --------------------------------------------------------

def test_embedding_is_512d_and_l2_normalized(embedder, portrait):
    result = embedder.process_image(portrait)
    emb = result.embeddings[0]
    assert emb.dim == EMBEDDING_DIM
    assert emb.vector.shape == (EMBEDDING_DIM,)
    assert emb.vector.dtype == np.float32
    assert emb.norm == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic(embedder, portrait):
    a = embedder.process_image(portrait).embeddings[0]
    b = embedder.process_image(portrait).embeddings[0]
    assert np.allclose(a.vector, b.vector, atol=1e-6)


def test_l2_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(EMBEDDING_DIM, dtype=np.float32))


def test_l2_normalize_produces_unit_norm():
    v = np.arange(1, EMBEDDING_DIM + 1, dtype=np.float32)
    assert float(np.linalg.norm(l2_normalize(v))) == pytest.approx(1.0, abs=1e-6)
