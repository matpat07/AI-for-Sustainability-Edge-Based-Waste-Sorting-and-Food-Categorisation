#!/usr/bin/env bash
# ==============================================================================
# run_edge.sh - Throttled & Energy-Efficient Inference Runner for Raspberry Pi
# ==============================================================================
# Ensures inferences stay within thermal budget, leaving OS & I/O cores free.
#
# Usage:
#   ./run_edge.sh eval                  # Run evaluation with balanced throttling
#   ./run_edge.sh eval --eco            # 1 core (max 25% CPU), 100ms pacing
#   ./run_edge.sh eval --balanced       # 2 cores (max 50% CPU), 50ms pacing (Default)
#   ./run_edge.sh eval --max            # All cores (max throughput)
#   ./run_edge.sh infer <image_path>    # Single image inference
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON="${DIR}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

MODE="${1:-eval}"
shift || true

case "$MODE" in
    eval)
        PROFILE="balanced"
        EXTRA_ARGS=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --eco)
                    PROFILE="eco"
                    shift
                    ;;
                --balanced)
                    PROFILE="balanced"
                    shift
                    ;;
                --max)
                    PROFILE="max"
                    shift
                    ;;
                *)
                    EXTRA_ARGS+=("$1")
                    shift
                    ;;
            esac
        done

        if [ "$PROFILE" = "eco" ]; then
            echo "[*] Running in ECO mode (1 Core, max 25% CPU, 100ms pacing)..."
            exec taskset -c 0 nice -n 15 "$PYTHON" eval.py --cores 0 --delay 0.10 "${EXTRA_ARGS[@]}"
        elif [ "$PROFILE" = "max" ]; then
            echo "[*] Running in MAX mode (All Cores, unconstrained)..."
            exec "$PYTHON" eval.py --cores all --delay 0.0 "${EXTRA_ARGS[@]}"
        else
            echo "[*] Running in BALANCED mode (2 Cores, max 50% CPU, 50ms pacing)..."
            exec taskset -c 0,1 nice -n 10 "$PYTHON" eval.py --cores 0,1 --delay 0.05 "${EXTRA_ARGS[@]}"
        fi
        ;;

    infer)
        if [ $# -lt 1 ]; then
            echo "Usage: ./run_edge.sh infer <image_path> [options]"
            exit 1
        fi
        exec taskset -c 0,1 nice -n 10 "$PYTHON" infer.py "$@"
        ;;

    *)
        echo "Usage: ./run_edge.sh [eval|infer] [options]"
        exit 1
        ;;
esac
