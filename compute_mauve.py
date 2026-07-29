'''
Run it with:
python compute_mauve.py outputs/cap_remdm_mdlm_32_mauve_inputs.json

It will produce:
outputs/generated_sequences_mauve_result.json
'''
import argparse
import json
from pathlib import Path

import mauve


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate MAUVE from saved samples and references."
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the saved *_mauve_inputs.json file.",
    )

    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device ID. Use -1 for CPU.",
    )

    parser.add_argument(
        "--max-text-length",
        type=int,
        default=1024,
        help="Maximum text length used by MAUVE.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for the MAUVE result.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"MAUVE input file does not exist: {args.input_path}"
        )

    with args.input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    human_references = data.get("human_references")
    samples = data.get("samples")

    if not isinstance(human_references, list):
        raise ValueError(
            "'human_references' is missing or is not a list."
        )

    if not isinstance(samples, list):
        raise ValueError(
            "'samples' is missing or is not a list."
        )

    if len(samples) != len(human_references):
        raise ValueError(
            f"Number of samples and references must match: "
            f"{len(samples)} samples versus "
            f"{len(human_references)} references."
        )

    if not samples:
        raise ValueError("The input file contains no samples.")

    print(f"Calculating MAUVE using {len(samples)} text pairs...")

    results = mauve.compute_mauve(
        p_text=human_references,
        q_text=samples,
        device_id=args.device_id,
        max_text_length=args.max_text_length,
        verbose=False,
    )

    mauve_score = float(results.mauve)

    if args.output_path is None:
        input_name = args.input_path.stem

        if input_name.endswith("_mauve_inputs"):
            input_name = input_name.removesuffix("_mauve_inputs")

        output_path = args.input_path.with_name(
            f"{input_name}_mauve_result.json"
        )
    else:
        output_path = args.output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "MAUVE": mauve_score,
        "number_of_samples": len(samples),
        "max_text_length": args.max_text_length,
        "device_id": args.device_id,
        "input_path": str(args.input_path),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"MAUVE score: {mauve_score:.6f}")
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()