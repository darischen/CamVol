import numpy as np
import pytest

from handvol.face_profile import FaceProfile, MATCH_THRESHOLD


def _unit(vec):
    return vec / np.linalg.norm(vec)


def test_empty_profile_does_not_match():
    p = FaceProfile.create_empty(embedding_dim=8)
    is_match, sim = p.matches(_unit(np.ones(8, dtype=np.float32)))
    assert is_match is False
    assert sim == 0.0


def test_identical_embedding_matches_at_similarity_1():
    p = FaceProfile.create_empty(embedding_dim=8)
    v = _unit(np.arange(1, 9, dtype=np.float32))
    p.add_capture(v)
    is_match, sim = p.matches(v)
    assert is_match is True
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_embedding_does_not_match():
    p = FaceProfile.create_empty(embedding_dim=8)
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[1] = 1.0
    p.add_capture(a)
    is_match, sim = p.matches(b)
    assert is_match is False
    assert sim == pytest.approx(0.0, abs=1e-6)


def test_max_similarity_used_across_multiple_captures():
    p = FaceProfile.create_empty(embedding_dim=8)
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[3] = 1.0
    p.add_capture(a)
    p.add_capture(b)
    query = b.copy()
    _, sim = p.matches(query)
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_threshold_boundary(tmp_path):
    p = FaceProfile.create_empty(embedding_dim=8)
    v = _unit(np.arange(1, 9, dtype=np.float32))
    p.add_capture(v)
    perturbed = _unit(v + 0.01 * np.ones(8, dtype=np.float32))
    is_match, sim = p.matches(perturbed)
    assert is_match is (sim >= MATCH_THRESHOLD)


def test_save_and_load_round_trip(tmp_path):
    p = FaceProfile.create_empty(embedding_dim=8)
    v1 = _unit(np.arange(1, 9, dtype=np.float32))
    v2 = _unit(np.arange(8, 0, -1).astype(np.float32))
    p.add_capture(v1)
    p.add_capture(v2)

    path = tmp_path / "profile.npz"
    p.save(path)
    assert path.exists()

    loaded = FaceProfile.load(path)
    assert loaded is not None
    assert loaded.embeddings.shape == (2, 8)
    np.testing.assert_allclose(loaded.embeddings[0], v1, rtol=1e-6)
    np.testing.assert_allclose(loaded.embeddings[1], v2, rtol=1e-6)


def test_load_missing_file_returns_none(tmp_path):
    assert FaceProfile.load(tmp_path / "nope.npz") is None


def test_load_corrupted_file_returns_none(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a real npz file")
    assert FaceProfile.load(bad) is None


def test_add_capture_rejects_wrong_dim():
    p = FaceProfile.create_empty(embedding_dim=8)
    with pytest.raises(ValueError):
        p.add_capture(np.zeros(7, dtype=np.float32))
