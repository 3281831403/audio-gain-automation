"""Red tests for the audio gain automation exercise.

Every test calls at least one public function from
``automation.envelope``.  The stubs raise ``NotImplementedError`` until the
module is implemented, so all 15 tests fail at runtime without collection,
import, or syntax errors.
"""

import math

import pytest

from automation.envelope import (
    AutomationError,
    apply_gain,
    automate_batch,
    db_to_linear,
    insert_point,
    interpolate,
    linear_to_db,
    ramp,
    trim_outliers,
    validate,
)


def test_defect_01_exponential_curve_is_linear_in_amplitude():
    """An exponential fade must not collapse to a straight line in dB."""
    points = [(0, 0.0), (100, -60.0)]
    midpoint = interpolate(points, 50, "exponential")
    midpoint_amplitude = (1.0 + 10.0 ** (-60.0 / 20.0)) / 2.0
    expected = 20.0 * math.log10(midpoint_amplitude)

    assert midpoint == pytest.approx(expected, abs=1e-6)
    assert abs(midpoint - (-30.0)) > 20.0


def test_defect_02_outside_points_hold_endpoint_gain():
    """Positions before/after the envelope must use the endpoint gain."""
    points = [(4, -3.0), (8, -9.0)]

    assert interpolate(points, 0, "linear") == pytest.approx(-3.0, abs=1e-6)
    assert interpolate(points, 12, "linear") == pytest.approx(-9.0, abs=1e-6)


def test_defect_03_db_to_linear_uses_the_amplitude_formula():
    """dBFS conversion must use 10**(db/20), not 20**(db/10)."""
    assert db_to_linear(-6.0) == pytest.approx(
        10.0 ** (-6.0 / 20.0), abs=1e-6
    )
    assert db_to_linear(6.0) == pytest.approx(
        10.0 ** (6.0 / 20.0), abs=1e-6
    )
    assert db_to_linear(-6.0) > 0.4


def test_defect_04_unsorted_points_are_normalized():
    """An out-of-order keyframe list must not make time run backwards."""
    points = [(20, -12.0), (0, 0.0), (10, -6.0)]

    assert interpolate(points, 15, "linear") == pytest.approx(-9.0, abs=1e-6)
    assert interpolate(points, 5, "linear") == pytest.approx(-3.0, abs=1e-6)
    assert validate(points) == ["sample indices are not sorted"]


def test_defect_05_duplicate_keyframes_use_the_last_gain():
    """A repeated sample index must overwrite, not keep the first value."""
    points = insert_point([(0, 0.0), (10, -6.0)], 10, -24.0)

    assert points == [(0, 0.0), (10, -24.0)]
    duplicate_input = [(0, 0.0), (10, -6.0), (10, -24.0)]
    assert interpolate(duplicate_input, 10, "linear") == pytest.approx(
        -24.0, abs=1e-6
    )
    assert interpolate(duplicate_input, 5, "linear") == pytest.approx(
        -12.0, abs=1e-6
    )


def test_defect_06_ramp_duration_is_measured_in_samples():
    """A ramp must cover the sample interval, not a frame count."""
    points = [(0, 0.0), (10, 0.0)]
    result = ramp(points, 100, 500, 0.0, -30.0, "linear")
    by_index = dict(result)

    assert len([i for i in by_index if 100 <= i <= 500]) == 401
    assert by_index[100] == pytest.approx(0.0, abs=1e-6)
    assert by_index[300] == pytest.approx(-15.0, abs=1e-6)
    assert by_index[500] == pytest.approx(-30.0, abs=1e-6)
    assert by_index[0] == pytest.approx(0.0, abs=1e-6)
    assert by_index[10] == pytest.approx(0.0, abs=1e-6)


def test_defect_07_gain_jumps_are_bridged_with_intermediate_points():
    """A large gain step must be split into bounded intermediate steps."""
    result = trim_outliers([(0, 0.0), (100, 12.0)], 3.0)

    assert len(result) > 2
    assert result[0][0] == 0
    assert result[0][1] == pytest.approx(0.0, abs=1e-6)
    assert result[-1][0] == 100
    assert result[-1][1] == pytest.approx(12.0, abs=1e-6)
    for (i0, g0), (i1, g1) in zip(result, result[1:]):
        assert i1 > i0
        assert abs(g1 - g0) <= 3.0 + 1e-6


def test_defect_08_over_unity_results_are_not_silently_clipped():
    """An overload must remain visible outside the [-1, 1] range."""
    gain = 10.0 ** (6.0 / 20.0)
    result = apply_gain([0.8, -0.8, 0.5], [(0, 6.0)], "linear")

    assert result == pytest.approx(
        [0.8 * gain, -0.8 * gain, 0.5 * gain], abs=1e-6
    )
    assert result[0] > 1.0
    assert result[1] < -1.0


def test_defect_09_batch_tracks_are_processed_independently():
    """One track's envelope must not leak into another track's result."""
    points = [(0, -6.0), (2, 0.0)]
    tracks = {
        "loud": [1.0, 1.0, 1.0],
        "quiet": [0.25, 0.25, 0.25],
    }
    loud_before = list(tracks["loud"])
    quiet_before = list(tracks["quiet"])

    result = automate_batch(tracks, points, "linear")
    gains = [
        10.0 ** (-6.0 / 20.0),
        10.0 ** (-3.0 / 20.0),
        1.0,
    ]

    assert result["loud"] == pytest.approx(gains, abs=1e-6)
    assert result["quiet"] == pytest.approx(
        [0.25 * gain for gain in gains], abs=1e-6
    )
    assert result["loud"] is not result["quiet"]
    assert points == [(0, -6.0), (2, 0.0)]
    assert tracks["loud"] == loud_before
    assert tracks["quiet"] == quiet_before


def test_defect_10_linear_to_db_of_zero_is_negative_infinity():
    """Zero and negative linear gains map to -inf instead of dividing by 0."""
    zero_db = linear_to_db(0.0)
    negative_db = linear_to_db(-0.5)

    assert math.isinf(zero_db) and zero_db < 0.0
    assert math.isinf(negative_db) and negative_db < 0.0
    assert linear_to_db(0.5) == pytest.approx(
        20.0 * math.log10(0.5), abs=1e-6
    )


def test_empty_points_return_original_samples():
    """With no keyframes, apply_gain is a value-preserving bypass."""
    samples = [0.0, 0.25, -0.5]
    result = apply_gain(samples, [], "linear")

    assert result == pytest.approx(samples, abs=1e-6)


def test_zero_db_is_unity_linear_gain():
    """0 dBFS is a linear gain of exactly 1."""
    assert db_to_linear(0.0) == pytest.approx(1.0, abs=1e-12)


def test_single_point_envelope_is_constant():
    """A one-keyframe envelope applies its gain across the whole timeline."""
    gain = 10.0 ** (-12.0 / 20.0)
    result = apply_gain([1.0, 0.5, -0.25], [(7, -12.0)], "linear")

    assert result == pytest.approx(
        [gain, 0.5 * gain, -0.25 * gain], abs=1e-6
    )


def test_non_positive_max_step_raises_automation_error():
    """A non-positive jump threshold cannot define a valid envelope."""
    for bad_step in (0.0, -0.5):
        with pytest.raises(AutomationError):
            trim_outliers([(0, 0.0), (1, 1.0)], bad_step)


def test_batch_success_preserves_track_order():
    """A successful batch must keep the input mapping's insertion order."""
    tracks = {
        "intro": [1.0],
        "verse": [0.5],
        "outro": [0.25],
    }
    result = automate_batch(tracks, [(0, 0.0)], "linear")

    assert list(result) == ["intro", "verse", "outro"]
    assert result["intro"] == pytest.approx([1.0], abs=1e-6)
    assert result["verse"] == pytest.approx([0.5], abs=1e-6)
    assert result["outro"] == pytest.approx([0.25], abs=1e-6)
