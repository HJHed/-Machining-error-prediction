import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run prediction for one machining process with one saved model."
    )
    parser.add_argument("--model-path", required=True, help="Full path to the saved .keras/.h5 model.")
    parser.add_argument("--artifact-path", help="Full path to the matching preprocessing .pkl artifact.")
    parser.add_argument("--output-dir", help="Directory for prediction outputs.")
    parser.add_argument(
        "--process-number",
        type=int,
        default=10,
        help="One-based machining process number. Default: 10.",
    )
    parser.add_argument(
        "--source-name",
        help="Exact source JSON filename, for example 模型10.json. Overrides --process-number.",
    )
    parser.add_argument(
        "--no-open-plots",
        action="store_true",
        help="Do not open saved prediction plots after the run.",
    )
    args = parser.parse_args()

    if args.process_number < 1:
        raise ValueError("--process-number must be one-based and >= 1.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(script_dir, "main.py")
    cmd = [
        sys.executable,
        main_py,
        "--run-mode",
        "predict_only",
        "--model-path",
        args.model_path,
    ]
    if args.no_open_plots:
        cmd.append("--no-open-plots")
    else:
        cmd.append("--open-plots")
    if args.artifact_path:
        cmd.extend(["--artifact-path", args.artifact_path])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.source_name:
        cmd.extend(["--predict-source-name", args.source_name])
    else:
        cmd.extend(["--predict-source-index", str(args.process_number - 1)])

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
