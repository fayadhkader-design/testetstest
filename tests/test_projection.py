import numpy as np
import pytest

from nnscope.projection import StableProjector, _orthonormalize, _procrustes_rotation


def random_basis(rng, d, k=2):
    return _orthonormalize(rng.standard_normal((d, k)))


def test_rejects_momentum_outside_unit_interval():
    with pytest.raises(ValueError):
        StableProjector(momentum=1.5)
    with pytest.raises(ValueError):
        StableProjector(momentum=-0.1)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        StableProjector().fit_transform(np.zeros(10))


def test_empty_batch_yields_empty_coords():
    result = StableProjector().fit_transform(np.zeros((0, 8)))
    assert result.coords.shape == (0, 2)


def test_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    result = StableProjector().fit_transform(rng.standard_normal((64, 16)))

    assert result.coords.shape == (64, 2)
    assert result.coords.dtype == np.float32


def test_low_dimensional_input_passes_through_centered():
    x = np.array([[1.0, 5.0], [3.0, 7.0]])
    result = StableProjector().fit_transform(x)

    assert result.explained_variance == 1.0
    np.testing.assert_allclose(result.coords, [[-1.0, -1.0], [1.0, 1.0]], atol=1e-6)


def test_one_dimensional_input_is_padded():
    result = StableProjector().fit_transform(np.array([[1.0], [3.0]]))

    assert result.coords.shape == (2, 2)
    np.testing.assert_allclose(result.coords[:, 1], [0.0, 0.0])


def test_rank_two_data_explains_all_variance():
    rng = np.random.default_rng(1)
    plane = rng.standard_normal((100, 2)) @ rng.standard_normal((2, 12))
    result = StableProjector().fit_transform(plane)

    assert result.explained_variance == pytest.approx(1.0, abs=1e-9)


def test_isotropic_noise_explains_roughly_its_share():
    rng = np.random.default_rng(2)
    result = StableProjector().fit_transform(rng.standard_normal((4000, 10)))

    # Two of ten equally-sized directions, so ~20% with sampling slack.
    assert 0.15 < result.explained_variance < 0.28


def test_procrustes_recovers_a_known_rotation():
    rng = np.random.default_rng(3)
    target = random_basis(rng, 9)
    theta = 0.7
    known = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    source = target @ known

    recovered = _procrustes_rotation(source, target)

    np.testing.assert_allclose(source @ recovered, target, atol=1e-10)


def test_procrustes_undoes_a_sign_flip():
    rng = np.random.default_rng(4)
    target = random_basis(rng, 7)
    flipped = target * np.array([1.0, -1.0])

    aligned = flipped @ _procrustes_rotation(flipped, target)

    np.testing.assert_allclose(aligned, target, atol=1e-10)


def test_zero_momentum_freezes_the_basis():
    rng = np.random.default_rng(5)
    projector = StableProjector(momentum=0.0)

    projector.fit_transform(rng.standard_normal((80, 12)))
    frozen = projector._basis.copy()
    projector.fit_transform(rng.standard_normal((80, 12)))

    np.testing.assert_allclose(projector._basis, frozen, atol=1e-12)


def test_reset_forgets_the_basis():
    rng = np.random.default_rng(6)
    projector = StableProjector()
    projector.fit_transform(rng.standard_normal((32, 8)))
    assert projector.fitted

    projector.reset()

    assert not projector.fitted


def test_projection_stays_continuous_as_embeddings_drift():
    """The whole point of the module: no strobing between frames.

    A drifting cloud is projected over many frames; consecutive frames must
    stay strongly positively correlated. A sign flip or basis swap would show
    up as a correlation near -1 or 0.
    """
    rng = np.random.default_rng(7)
    cloud = rng.standard_normal((200, 24))
    projector = StableProjector(momentum=0.2)

    previous = projector.fit_transform(cloud).coords
    correlations = []
    for _ in range(60):
        cloud = cloud + rng.standard_normal(cloud.shape) * 0.02
        current = projector.fit_transform(cloud).coords
        for axis in range(2):
            correlations.append(
                np.corrcoef(previous[:, axis], current[:, axis])[0, 1]
            )
        previous = current

    assert min(correlations) > 0.9, f"projection flipped: min corr {min(correlations)}"


def test_rotation_settles_once_geometry_is_stable():
    rng = np.random.default_rng(8)
    cloud = rng.standard_normal((150, 20))
    projector = StableProjector(momentum=0.2)

    projector.fit_transform(cloud)
    rotations = [projector.fit_transform(cloud).rotation for _ in range(10)]

    assert rotations[-1] < 1e-6, "static data should stop rotating the basis"
