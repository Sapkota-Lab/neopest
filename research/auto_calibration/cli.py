"""Command-line interface for the auto-calibration research prototype."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research.auto_calibration.evaluation import evaluate_result_sets
from research.auto_calibration.manual_reference import (
    ManualReferenceError,
    build_manual_reference,
)
from research.auto_calibration.pipeline import (
    list_supported_images,
    run_auto_calibration_proposal,
    run_batch_auto_calibration_proposals,
)
from research.auto_calibration.visualization import write_proposal_overlay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research.auto_calibration.cli",
        description=(
            "NeoPest assisted auto-calibration research prototype (AC-0). "
            "Outputs are proposals, not applied calibrations."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the pipeline on one image.")
    run_parser.add_argument("--image", required=True, type=Path, help="Source image path.")
    run_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for result JSON and optional overlay.",
    )
    run_parser.add_argument(
        "--write-overlay",
        action="store_true",
        help="Write a diagnostic overlay image alongside the result JSON.",
    )

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run the pipeline on all supported images in a directory.",
    )
    batch_parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory containing source images.",
    )
    batch_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for per-image result JSON files.",
    )
    batch_parser.add_argument(
        "--write-overlay",
        action="store_true",
        help="Write diagnostic overlay images.",
    )

    reference_parser = subparsers.add_parser(
        "make-reference",
        help="Create or validate a manual-reference JSON file.",
    )
    reference_parser.add_argument("--image", required=True, type=Path)
    reference_parser.add_argument("--x1", required=True, type=float)
    reference_parser.add_argument("--y1", required=True, type=float)
    reference_parser.add_argument("--x2", required=True, type=float)
    reference_parser.add_argument("--y2", required=True, type=float)
    reference_parser.add_argument("--output", required=True, type=Path)
    reference_parser.add_argument(
        "--annotator",
        default="anonymous",
        help="Annotator identifier or anonymous label.",
    )
    reference_parser.add_argument("--notes", default="", help="Optional notes.")
    reference_parser.add_argument(
        "--annotated-at",
        default=None,
        help=(
            "ISO-8601 annotation timestamp (defaults to current UTC). "
            "Supply a fixed value for byte-identical fixture JSON."
        ),
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Compare result JSON files against manual-reference JSON files.",
    )
    evaluate_parser.add_argument("--results-dir", required=True, type=Path)
    evaluate_parser.add_argument("--references-dir", required=True, type=Path)
    evaluate_parser.add_argument("--output", required=True, type=Path)

    return parser


def _result_output_path(output_dir: Path, image_path: Path) -> Path:
    return output_dir / f"{image_path.stem}.json"


def _overlay_output_path(output_dir: Path, image_path: Path) -> Path:
    return output_dir / f"{image_path.stem}_overlay.png"


def cmd_run(args: argparse.Namespace) -> int:
    image_path: Path = args.image
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_auto_calibration_proposal(image_path)
    result_path = _result_output_path(output_dir, image_path)
    result.write_json(result_path)

    if args.write_overlay:
        write_proposal_overlay(
            image_path,
            result,
            _overlay_output_path(output_dir, image_path),
        )

    print(result_path)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_batch_auto_calibration_proposals(args.images_dir)
    image_paths = list_supported_images(args.images_dir)
    for result, image_path in zip(results, image_paths, strict=True):
        result_path = _result_output_path(output_dir, image_path)
        result.write_json(result_path)
        if args.write_overlay:
            write_proposal_overlay(
                image_path,
                result,
                _overlay_output_path(output_dir, image_path),
            )
        print(result_path)
    return 0


def cmd_make_reference(args: argparse.Namespace) -> int:
    try:
        reference = build_manual_reference(
            source_image=str(args.image),
            endpoint_a_px=(args.x1, args.y1),
            endpoint_b_px=(args.x2, args.y2),
            annotator=args.annotator,
            annotated_at=args.annotated_at,
            notes=args.notes,
        )
    except ManualReferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    reference.write_json(args.output)
    print(args.output)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_result_sets(args.results_dir, args.references_dir)
    report.write_json(args.output)
    print(args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)
    if args.command == "batch":
        return cmd_batch(args)
    if args.command == "make-reference":
        return cmd_make_reference(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
