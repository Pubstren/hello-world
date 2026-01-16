"""Simple BPSK + AWGN channel simulation."""

from __future__ import annotations

import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.int8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8)
    if bits.size % 8 != 0:
        raise ValueError("bit length must be multiple of 8")
    return np.packbits(bits).tobytes()


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    return np.where(bits == 0, -1.0, 1.0)


def bpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    return (symbols > 0).astype(np.int8)


def awgn_channel(symbols: np.ndarray, eb_n0_db: float) -> np.ndarray:
    eb_n0 = 10 ** (eb_n0_db / 10)
    sigma = np.sqrt(1 / (2 * eb_n0))
    noise = np.random.normal(0, sigma, size=symbols.shape)
    return symbols + noise


def transmit_bpsk_awgn(data: bytes, eb_n0_db: float) -> bytes:
    bits = bytes_to_bits(data)
    symbols = bpsk_modulate(bits)
    noisy = awgn_channel(symbols, eb_n0_db)
    rx_bits = bpsk_demodulate(noisy)
    return bits_to_bytes(rx_bits)
