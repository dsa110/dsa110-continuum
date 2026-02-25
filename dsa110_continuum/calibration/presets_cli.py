#!/usr/bin/env python
"""
CLI utility for working with calibration presets.

Usage:
    # List available presets
    python -m dsa110_contimg.core.calibration.presets_cli list

    # Show preset details
    python -m dsa110_contimg.core.calibration.presets_cli show standard

    # Show customized preset
    python -m dsa110_contimg.core.calibration.presets_cli show standard --refant=105 --gain_solint=60s
"""

import argparse
import json
import sys

from dsa110_contimg.core.calibration.presets import (
    PRESETS,
    get_preset,
    list_presets,
)


def cmd_list(args) -> int:
    """List available presets."""
    print("Available calibration presets:")
    for name in list_presets():
        preset = PRESETS[name]
        # Show brief summary
        stages = []
        if preset.solve_delay:
            stages.append("K")
        if preset.solve_bandpass:
            stages.append("BP")
        if preset.solve_gains:
            stages.append(f"G({preset.gain_calmode})")
        stages_str = "+".join(stages) if stages else "none"
        print(
            f"  {name:12s} - {stages_str:15s} (refant={preset.refant}, solint={preset.gain_solint})"
        )
    return 0


def cmd_show(args) -> int:
    """Show preset details."""
    try:
        preset = get_preset(args.name)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Apply overrides if provided
    overrides = {}
    if args.overrides:
        for override in args.overrides:
            if "=" not in override:
                print(
                    f"Error: Invalid override format '{override}'. Expected key=value",
                    file=sys.stderr,
                )
                return 1
            key, value = override.split("=", 1)
            # Try to parse value as int/float/bool
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            overrides[key] = value

    if overrides:
        print(f"Preset: {args.name} (with overrides)")
        preset = preset.with_overrides(**overrides)
    else:
        print(f"Preset: {args.name}")

    print("=" * 80)

    if args.format == "dict":
        # Show as Python dict
        params = preset.to_dict()
        for key, value in sorted(params.items()):
            print(f"{key:25s} = {value!r}")
    elif args.format == "json":
        # Show as JSON (useful for piping to other tools)
        params = preset.to_dict()
        print(json.dumps(params, indent=2))
    else:
        # Show as human-readable summary
        print("\nGeneral:")
        print(f"  field: {preset.field}")
        print(f"  refant: {preset.refant}")
        print(f"  model_source: {preset.model_source}")
        print(f"  calibrator_name: {preset.calibrator_name}")

        print("\nStages:")
        print(f"  solve_delay: {preset.solve_delay}")
        print(f"  solve_bandpass: {preset.solve_bandpass}")
        print(f"  solve_gains: {preset.solve_gains}")

        if preset.solve_delay:
            print("\nDelay (K) parameters:")
            print(f"  k_minsnr: {preset.k_minsnr}")
            print(f"  k_t_slow: {preset.k_t_slow}")
            print(f"  k_t_fast: {preset.k_t_fast}")

        if preset.solve_bandpass:
            print("\nBandpass (BP) parameters:")
            print(f"  bp_minsnr: {preset.bp_minsnr}")
            print(f"  bp_combine_field: {preset.bp_combine_field}")
            print(f"  bp_combine_spw: {preset.bp_combine_spw}")
            if preset.prebp_phase:
                print(f"  prebp_phase: {preset.prebp_phase} (minsnr={preset.prebp_minsnr})")

        if preset.solve_gains:
            print("\nGain (G) parameters:")
            print(f"  gain_solint: {preset.gain_solint}")
            print(f"  gain_calmode: {preset.gain_calmode}")
            print(f"  gain_minsnr: {preset.gain_minsnr}")

        print("\nFlagging:")
        print(f"  do_flagging: {preset.do_flagging}")
        print(f"  flag_autocorr: {preset.flag_autocorr}")
        print(f"  use_adaptive_flagging: {preset.use_adaptive_flagging}")

        if preset.fast:
            print("\n Fast mode enabled (phase-only, skip K/BP)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Calibration preset utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    subparsers.add_parser("list", help="List available presets")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show preset details")
    show_parser.add_argument("name", help="Preset name")
    show_parser.add_argument(
        "--format",
        choices=["summary", "dict", "json"],
        default="summary",
        help="Output format (default: summary)",
    )
    show_parser.add_argument(
        "overrides",
        nargs="*",
        help="Parameter overrides (e.g., refant=105 gain_solint=60s)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "list":
        return cmd_list(args)
    elif args.command == "show":
        return cmd_show(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
