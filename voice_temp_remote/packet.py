"""Packet formatting and CRC utilities."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import PacketConfig


@dataclass
class PacketHeader:
    version: int
    msg_type: int
    seq: int
    cmd_id: int
    n_frames: int
    dim: int
    scale_q: int


HEADER_FORMAT = ">2sBBH B H B H"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
CRC_FORMAT = ">H"


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def pack_packet(
    header: PacketHeader,
    payload: np.ndarray,
    packet_cfg: PacketConfig,
) -> bytes:
    if payload.dtype != np.int8:
        raise ValueError("payload must be int8")
    scale_q = int(round(header.scale_q))
    header_bytes = struct.pack(
        HEADER_FORMAT,
        packet_cfg.magic,
        header.version,
        header.msg_type,
        header.seq,
        header.cmd_id,
        header.n_frames,
        header.dim,
        scale_q,
    )
    payload_bytes = payload.tobytes()
    crc = crc16(header_bytes + payload_bytes)
    return header_bytes + payload_bytes + struct.pack(CRC_FORMAT, crc)


def unpack_packet(
    data: bytes, packet_cfg: PacketConfig
) -> Tuple[PacketHeader, np.ndarray]:
    if len(data) < HEADER_SIZE + 2:
        raise ValueError("packet too short")
    header_values = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    magic = header_values[0]
    if magic != packet_cfg.magic:
        raise ValueError("invalid magic")
    header = PacketHeader(
        version=header_values[1],
        msg_type=header_values[2],
        seq=header_values[3],
        cmd_id=header_values[4],
        n_frames=header_values[5],
        dim=header_values[6],
        scale_q=header_values[7],
    )
    payload_size = header.n_frames * header.dim
    payload_end = HEADER_SIZE + payload_size
    if len(data) < payload_end + 2:
        raise ValueError("packet length mismatch")
    payload_bytes = data[HEADER_SIZE:payload_end]
    crc_expected = struct.unpack(CRC_FORMAT, data[payload_end : payload_end + 2])[0]
    crc_actual = crc16(data[:payload_end])
    if crc_actual != crc_expected:
        raise ValueError("crc mismatch")
    payload = np.frombuffer(payload_bytes, dtype=np.int8).reshape(header.n_frames, header.dim)
    return header, payload
