"""Stopping rules for adaptive BER simulation."""


def get_stop_reason(
    errors_zf: int,
    errors_mmse: int,
    total_bits: int,
    target_errors: int,
    max_bits: int,
) -> str | None:
    """Determine whether a BER simulation should stop.

    Args:
        errors_zf: Accumulated ZF bit errors.
        errors_mmse: Accumulated MMSE bit errors.
        total_bits: Number of tested bits per equalizer.
        target_errors: Required errors for each equalizer.
        max_bits: Maximum number of tested bits.

    Returns:
        "target_reached" if both algorithms reach the target.
        "max_bits_reached" if the bit limit is reached.
        None if simulation should continue.

    Raises:
        ValueError: If counters or limits are invalid.
    """
    if (
        min(errors_zf, errors_mmse, total_bits) < 0
        or target_errors <= 0
        or max_bits <= 0
        or max(errors_zf, errors_mmse) > total_bits
    ):
        raise ValueError("Invalid BER stopping parameters")

    if errors_zf >= target_errors and errors_mmse >= target_errors:
        return "target_reached"

    if total_bits >= max_bits:
        return "max_bits_reached"

    return None
