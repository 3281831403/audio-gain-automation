"""Sample-accurate automation envelopes for an offline audio gain tool.

The public API works with keyframes represented as
``[(sample_index, gain_db), ...]``.  Time is measured in samples, not
seconds or frames, so automation stays aligned when the sample rate
changes.  Gains are in dBFS and are converted with ``10 ** (db / 20)``.  The
supported curve names are ``"linear"``, ``"hold"``, and
``"exponential"``.

Functions that build or evaluate an envelope normalize keyframe input
before use: points are sorted by sample index and duplicate indices keep
the last value.  Inputs are never mutated.  Unless a tighter bound is
stated, absolute comparisons use a tolerance of ``1e-6``.
"""

import math
from bisect import bisect_left

GAIN_DB_MIN = -120.0
GAIN_DB_MAX = 120.0

_CURVES = frozenset(("linear", "hold", "exponential"))
_TOL = 1e-6


class AutomationError(ValueError):
    """Raised for malformed keyframes, samples, or automation parameters."""


def _is_int(value) -> bool:
    """Return ``True`` for a real integer sample index (not a bool)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    """Return ``True`` for a real numeric value (not a bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_index(index, what: str = "sample index") -> int:
    if not _is_int(index) or index < 0:
        raise AutomationError(f"{what} must be a non-negative integer")
    return int(index)


def _check_gain(gain, what: str = "gain_db") -> float:
    if (
        not _is_number(gain)
        or not math.isfinite(gain)
        or gain < GAIN_DB_MIN
        or gain > GAIN_DB_MAX
    ):
        raise AutomationError(
            f"{what} must be finite and within "
            f"[{GAIN_DB_MIN:g}, {GAIN_DB_MAX:g}] dB"
        )
    return float(gain)


def _check_curve(curve) -> str:
    if curve not in _CURVES:
        raise AutomationError(f"unknown curve: {curve!r}")
    return curve


def _normalize(points) -> list[tuple[int, float]]:
    """Sort keyframes by sample index, keeping the last duplicate value.

    A new list of new tuples is returned; the input is never mutated.
    Malformed indices or gains raise :class:`AutomationError`.
    """
    if isinstance(points, (str, bytes)):
        raise AutomationError("points must be a sequence of keyframes")
    try:
        raw = list(points)
    except TypeError as exc:
        raise AutomationError(
            "points must be a sequence of keyframes"
        ) from exc

    parsed: list[tuple[int, float]] = []
    for point in raw:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise AutomationError(f"malformed keyframe: {point!r}")
        index, gain = point
        parsed.append((_check_index(index), _check_gain(gain)))

    # ``sorted`` is stable, so for equal indices the value occurring later
    # in the original list comes last and wins the duplicate overwrite.
    parsed.sort(key=lambda item: item[0])
    result: list[tuple[int, float]] = []
    for index, gain in parsed:
        if result and result[-1][0] == index:
            result[-1] = (index, gain)
        else:
            result.append((index, gain))
    return result


def insert_point(
    points: list[tuple[int, float]],
    index: int,
    gain_db: float,
) -> list[tuple[int, float]]:
    """Insert or replace a keyframe at ``index`` and return a sorted list.

    ``points`` uses ``(sample_index, gain_db)`` pairs.  Existing points are
    normalized by sorting, and duplicate indices keep their last value.  If
    ``index`` already exists, its gain is replaced by ``gain_db``.  The
    returned list is a new ascending, duplicate-free list; the input list
    and its tuples are not mutated.

    ``gain_db`` is in dBFS, and ``sample_index`` values are non-negative
    integers.  Absolute comparisons use a tolerance of ``1e-6``.

    Raises:
        AutomationError: if ``index`` is not a non-negative integer, or
            ``gain_db`` is non-finite or outside ``[-120, 120]``.
    """
    index = _check_index(index)
    gain_db = _check_gain(gain_db)

    result = _normalize(points)
    position = bisect_left([item[0] for item in result], index)
    if position < len(result) and result[position][0] == index:
        result[position] = (index, gain_db)
    else:
        result.insert(position, (index, gain_db))
    return result


def _crossfade(
    gain0_db: float,
    gain1_db: float,
    t: float,
    curve: str,
) -> float:
    """Interpolate between two gains with ``t`` in ``[0, 1]``."""
    if curve == "linear":
        return gain0_db + (gain1_db - gain0_db) * t
    if curve == "hold":
        return gain0_db
    # exponential: linear in amplitude, i.e. perceptually smooth level.
    amp0 = db_to_linear(gain0_db)
    amp1 = db_to_linear(gain1_db)
    return linear_to_db(amp0 + (amp1 - amp0) * t)


def interpolate(
    points: list[tuple[int, float]],
    index: int,
    curve: str,
) -> float:
    """Return the envelope gain in dB at ``index``.

    Points are normalized first (sorted, with duplicate indices keeping the
    last value).  ``curve`` selects one of three rules between the two
    surrounding keyframes:

    ``"linear"``
        Linear in dB: ``g0 + (g1 - g0) * t``.
    ``"hold"``
        Step: use ``g0`` until the next keyframe is reached, then ``g1``.
    ``"exponential"``
        Linear in amplitude (the musically smooth fade): convert both
        endpoint gains with ``10 ** (db / 20)``, interpolate those
        amplitudes linearly, then convert the result back with
        ``20 * log10``.

    Before the first keyframe the first gain is returned; after the last
    keyframe the last gain is returned.  A single keyframe therefore
    applies to the whole timeline.  At a keyframe the stored gain is
    returned within ``1e-6``.

    Raises:
        AutomationError: if ``points`` is empty, ``index`` is not a
            non-negative integer, ``curve`` is unknown, or a keyframe is
            malformed.
    """
    index = _check_index(index)
    _check_curve(curve)
    normalized = _normalize(points)
    if not normalized:
        raise AutomationError("cannot interpolate an empty keyframe list")
    return _search(normalized, [item[0] for item in normalized], index, curve)


def _search(
    normalized: list[tuple[int, float]],
    indices: list[int],
    index: int,
    curve: str,
) -> float:
    """Evaluate a normalized envelope with precomputed ``indices``."""
    position = bisect_left(indices, index)

    # On a keyframe (indices are integers, so this is exact).
    if position < len(normalized) and indices[position] == index:
        return normalized[position][1]
    # Outside the envelope: hold the nearest endpoint gain.
    if position == 0:
        return normalized[0][1]
    if position == len(normalized):
        return normalized[-1][1]

    index0, gain0 = normalized[position - 1]
    index1, gain1 = normalized[position]
    t = (index - index0) / (index1 - index0)
    return _crossfade(gain0, gain1, t, curve)


def db_to_linear(db: float) -> float:
    """Convert dBFS to linear gain with ``10 ** (db / 20)``.

    ``db == 0`` returns exactly ``1.0``.  This is the inverse of
    :func:`linear_to_db` for positive linear gains.  No clipping is
    performed, and absolute comparisons use a tolerance of ``1e-6``.
    """
    return 10.0 ** (db / 20.0)


def linear_to_db(x: float) -> float:
    """Convert a linear gain to dBFS with ``20 * log10(x)``.

    ``x <= 0`` returns negative infinity instead of raising a
    divide-by-zero or math-domain error.  ``x == 1`` returns ``0.0``.
    Positive values use the standard dB formula, and absolute comparisons
    use a tolerance of ``1e-6``.
    """
    if x <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(x)


def _coerce_samples(samples) -> list[float]:
    if isinstance(samples, (str, bytes)):
        raise AutomationError("samples must be a sequence of finite numbers")
    try:
        raw = list(samples)
    except TypeError as exc:
        raise AutomationError(
            "samples must be a sequence of finite numbers"
        ) from exc

    result: list[float] = []
    for sample in raw:
        if not _is_number(sample) or not math.isfinite(sample):
            raise AutomationError(f"non-finite sample: {sample!r}")
        result.append(float(sample))
    return result


def apply_gain(
    samples: list[float],
    points: list[tuple[int, float]],
    curve: str,
) -> list[float]:
    """Apply an automation envelope to ``samples`` and return a new list.

    For sample position ``n`` the gain is ``interpolate(points, n, curve)``
    converted by :func:`db_to_linear`, and the output is
    ``samples[n] * gain``.  ``curve`` is one of ``"linear"``, ``"hold"``,
    or ``"exponential"``.  The result has the same length as ``samples``.

    Values whose magnitude exceeds ``1.0`` are preserved: they remain
    outside ``[-1, 1]`` as visible overload markers, and no silent clipping
    or clamping is performed.  A caller that needs a ceiling must check or
    limit the output explicitly.

    Empty ``points`` is a bypass and returns a new list with the original
    sample values.  Empty ``samples`` returns ``[]``.  Absolute comparisons
    use a tolerance of ``1e-6``.

    Raises:
        AutomationError: for a non-finite sample, invalid keyframes, or an
            unknown ``curve``.
    """
    _check_curve(curve)
    normalized = _normalize(points)
    values = _coerce_samples(samples)

    if not normalized:
        return values
    if not values:
        return []

    output: list[float] = []
    indices = [item[0] for item in normalized]
    for position, sample in enumerate(values):
        gain_db = _search(normalized, indices, position, curve)
        output.append(sample * db_to_linear(gain_db))
    return output


def ramp(
    points: list[tuple[int, float]],
    start: int,
    end: int,
    from_db: float,
    to_db: float,
    curve: str,
) -> list[tuple[int, float]]:
    """Build a sample-index ramp and return a new keyframe list.

    Every integer sample from ``start`` through ``end`` inclusive is
    replaced by a ramp keyframe.  With ``span = end - start`` and
    ``t = (n - start) / span``, the gain follows the curve rules used by
    :func:`interpolate`:

    * ``"linear"``: ``from_db + (to_db - from_db) * t`` in dB;
    * ``"exponential"``: amplitude-linear between the two endpoint gains;
    * ``"hold"``: ``from_db`` for ``start <= n < end`` and ``to_db`` at
      ``end``.

    The ramp therefore has ``end - start + 1`` keyframes and its duration
    is fixed in samples, independent of sample rate or frame count.
    Keyframes outside the closed interval are preserved; any inside are
    replaced.  The returned list is sorted and duplicate-free.

    ``start`` and ``end`` are non-negative integers, and ``end`` must be
    greater than ``start``.  ``from_db`` and ``to_db`` are in dBFS.
    Absolute comparisons use a tolerance of ``1e-6``.

    Raises:
        AutomationError: for invalid bounds, gains, keyframes, or an
            unknown ``curve``.
    """
    start = _check_index(start, "start")
    end = _check_index(end, "end")
    if end <= start:
        raise AutomationError("end must be greater than start")
    from_db = _check_gain(from_db, "from_db")
    to_db = _check_gain(to_db, "to_db")
    _check_curve(curve)

    kept = [
        (index, gain)
        for index, gain in _normalize(points)
        if index < start or index > end
    ]

    span = end - start
    ramp_points: list[tuple[int, float]] = []
    for index in range(start, end + 1):
        if curve == "hold":
            gain = from_db if index < end else to_db
        elif index == start:
            gain = from_db
        elif index == end:
            gain = to_db
        else:
            t = (index - start) / span
            gain = _crossfade(from_db, to_db, t, curve)
        ramp_points.append((index, gain))

    return _normalize(kept + ramp_points)


def trim_outliers(
    points: list[tuple[int, float]],
    max_step_db: float,
) -> list[tuple[int, float]]:
    """Insert intermediate keyframes so adjacent gain steps are bounded.

    Points are normalized first.  For each adjacent pair whose absolute
    gain difference exceeds ``max_step_db``, the interval is bisected
    recursively: a keyframe is inserted at the integer midpoint
    ``(i0 + i1) // 2`` with gain ``(g0 + g1) / 2``, and both halves are
    checked again.  Splitting stops when every adjacent gain difference is
    at most ``max_step_db`` or when no integer sample position remains
    between a pair.  Original keyframes are preserved, the result is
    sorted and duplicate-free, and the input is not mutated.

    ``max_step_db`` is a positive dB threshold.  A jump is considered a
    problem only when it is strictly greater than the threshold.  Absolute
    comparisons use a tolerance of ``1e-6``.

    Raises:
        AutomationError: if ``max_step_db`` is not positive and finite, if
            a keyframe is malformed, or if an oversized jump occurs between
            adjacent sample positions and cannot be bridged with an integer
            keyframe.
    """
    if (
        not _is_number(max_step_db)
        or not math.isfinite(max_step_db)
        or max_step_db <= 0.0
    ):
        raise AutomationError("max_step_db must be positive and finite")

    normalized = _normalize(points)
    if len(normalized) < 2:
        return normalized

    result: list[tuple[int, float]] = [normalized[0]]

    def split(
        index0: int,
        gain0: float,
        index1: int,
        gain1: float,
    ) -> None:
        """Append the bounded interior of (index0, index1), then index1."""
        difference = abs(gain1 - gain0)
        if difference <= max_step_db + _TOL:
            result.append((index1, gain1))
            return
        if index1 - index0 < 2:
            raise AutomationError(
                "oversized gain jump between adjacent sample positions"
            )
        midpoint = (index0 + index1) // 2
        mid_gain = (gain0 + gain1) / 2.0
        split(index0, gain0, midpoint, mid_gain)
        split(midpoint, mid_gain, index1, gain1)

    for (index0, gain0), (index1, gain1) in zip(
        normalized, normalized[1:]
    ):
        split(index0, gain0, index1, gain1)

    return result


def validate(points: list[tuple[int, float]]) -> list[str]:
    """Return a list of problems found in ``points``; ``[]`` means valid.

    One message is collected per problem category, in this order:

    1. a sample index is negative or not an integer
       (``"sample index must be a non-negative integer"``);
    2. sample indices are not strictly increasing
       (``"sample indices are not sorted"``);
    3. a sample index occurs more than once
       (``"duplicate sample index"``);
    4. a gain is non-finite or outside ``[-120, 120]`` dB
       (``"gain_db outside [-120, 120]"``).

    Duplicate indices are reported separately from ordering, so
    ``[(2, 0.0), (2, 1.0)]`` reports a duplicate but not a sort problem.
    The input is not modified.  These structural checks do not apply the
    ``1e-6`` interpolation tolerance; the gain bounds are inclusive.
    """
    if isinstance(points, (str, bytes)):
        raise AutomationError("points must be a sequence of keyframes")
    try:
        raw = list(points)
    except TypeError as exc:
        raise AutomationError(
            "points must be a sequence of keyframes"
        ) from exc

    bad_index = False
    not_sorted = False
    duplicate = False
    bad_gain = False
    previous_index = None
    seen: set[int] = set()

    for point in raw:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise AutomationError(f"malformed keyframe: {point!r}")
        index, gain = point

        if not _is_int(index) or index < 0:
            bad_index = True
        else:
            if previous_index is not None and index < previous_index:
                not_sorted = True
            if index in seen:
                duplicate = True
            seen.add(index)
            previous_index = index

        if (
            not _is_number(gain)
            or not math.isfinite(gain)
            or gain < GAIN_DB_MIN
            or gain > GAIN_DB_MAX
        ):
            bad_gain = True

    problems: list[str] = []
    if bad_index:
        problems.append("sample index must be a non-negative integer")
    if not_sorted:
        problems.append("sample indices are not sorted")
    if duplicate:
        problems.append("duplicate sample index")
    if bad_gain:
        problems.append("gain_db outside [-120, 120]")
    return problems


def automate_batch(
    tracks: dict[str, list[float]],
    points: list[tuple[int, float]],
    curve: str,
) -> dict[str, list[float]]:
    """Apply one envelope to a mapping of tracks, atomically.

    ``tracks`` maps track names to sample sequences.  The returned dict has
    the same keys in the same insertion order; each value is the result of
    :func:`apply_gain` for that track.  Every track receives an independent
    output list and an independent envelope evaluation, so no envelope
    state or output list is shared between tracks.

    The operation is all-or-nothing.  ``points`` is checked with
    :func:`validate`, and every sample is checked for finiteness before any
    output is produced.  If a track, the shared points, or the curve is
    invalid, ``AutomationError`` is raised and neither ``tracks`` nor
    ``points`` is modified.  On success the inputs are still not mutated.
    Empty ``points`` copies every track, and an empty track maps to ``[]``.
    ``curve`` is one of ``"linear"``, ``"hold"``, or ``"exponential"``, and
    absolute comparisons use a tolerance of ``1e-6``.

    Raises:
        AutomationError: if ``points`` fails validation, ``curve`` is
            unknown, a track is not a sample sequence, or any sample is not
            finite.
    """
    problems = validate(points)
    if problems:
        raise AutomationError("; ".join(problems))
    _check_curve(curve)

    if not hasattr(tracks, "items"):
        raise AutomationError("tracks must be a mapping of sample sequences")

    # Validate and copy every track before producing any output so the
    # operation stays all-or-nothing.
    prepared: dict[str, list[float]] = {}
    for name, samples in tracks.items():
        prepared[name] = _coerce_samples(samples)

    # A fresh apply_gain call per track means separate envelope evaluation
    # and a distinct output list for every track.
    return {
        name: apply_gain(samples, points, curve)
        for name, samples in prepared.items()
    }
