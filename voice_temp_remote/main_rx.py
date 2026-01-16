"""Receive-side entry point: decode, recognize, and control plant."""

from __future__ import annotations

import argparse
import pathlib
import pickle

from .channel_sim import transmit_bpsk_awgn
from .config import ControlConfig, PacketConfig, PIDConfig, PlantConfig, QuantizationConfig
from .controller import PIDController
from .mfcc_feat import dequantize_features
from .packet import unpack_packet
from .plant_temp import TemperaturePlant
from .rx_recognize import recognize_command


def load_templates(path: pathlib.Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice temperature remote RX")
    parser.add_argument("packet", type=pathlib.Path)
    parser.add_argument("--templates", type=pathlib.Path, required=True)
    parser.add_argument("--ebn0", type=float, default=10.0)
    args = parser.parse_args()

    packet_cfg = PacketConfig()
    quant_cfg = QuantizationConfig()
    control_cfg = ControlConfig()
    pid_cfg = PIDConfig()
    plant_cfg = PlantConfig()

    raw = args.packet.read_bytes()
    noisy = transmit_bpsk_awgn(raw, args.ebn0)
    header, payload = unpack_packet(noisy, packet_cfg)
    scale = header.scale_q / quant_cfg.scale_q_multiplier
    features = dequantize_features(payload, scale)

    templates = load_templates(args.templates)
    result = recognize_command(features, templates, band=40)

    controller = PIDController(control_cfg, pid_cfg)
    plant = TemperaturePlant(plant_cfg)
    controller.apply_command("on")
    controller.apply_command(result.command)

    disturbance = plant.apply_random_disturbance()
    measurement = plant.step(0.0, control_cfg.dt, disturbance=disturbance)
    control_u = controller.compute(measurement)
    measurement = plant.step(control_u, control_cfg.dt)

    print(
        f"cmd={result.command} dist={result.distance:.2f} "
        f"T={measurement:.2f} set={controller.state.setpoint:.2f} u={control_u:.2f}"
    )


if __name__ == "__main__":
    main()
