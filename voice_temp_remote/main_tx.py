"""Transmit-side entry point: capture audio, extract MFCC, and send packets."""

from __future__ import annotations

import argparse

from .config import AudioConfig, PacketConfig, QuantizationConfig
from .mfcc_feat import compute_mfcc, quantize_features
from .packet import PacketHeader, pack_packet
from .tx_capture import capture_with_vad


def build_packet(seq: int, features, scale, packet_cfg, quant_cfg):
    header = PacketHeader(
        version=packet_cfg.version,
        msg_type=0,
        seq=seq,
        cmd_id=0,
        n_frames=features.shape[0],
        dim=features.shape[1],
        scale_q=int(scale * quant_cfg.scale_q_multiplier),
    )
    return pack_packet(header, features, packet_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice temperature remote TX")
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()

    audio_cfg = AudioConfig()
    quant_cfg = QuantizationConfig()
    packet_cfg = PacketConfig()

    capture = capture_with_vad(args.duration, audio_cfg)
    feat = compute_mfcc(capture.audio, audio_cfg)
    quantized, scale = quantize_features(feat.features, quant_cfg)
    packet = build_packet(1, quantized, scale, packet_cfg, quant_cfg)
    print(f"Packet bytes: {len(packet)}")


if __name__ == "__main__":
    main()
