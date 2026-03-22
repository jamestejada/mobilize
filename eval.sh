#!/usr/bin/env bash
set -euo pipefail

# Load .env if present
if [ -f .env ]; then
    eval "$(uv run python -c 'from dotenv import dotenv_values; [print(f"export {k}={v!r}") for k,v in dotenv_values().items()]')"
fi

usage() {
    echo "Usage: $0 {eval [agent]|review [file]}"
    echo ""
    echo "  eval [agent]  Run the eval test suite, save results to logs/eval/eval_TIMESTAMP.json"
    echo "                Optionally filter to a single agent: praetor, explorator, tabularius,"
    echo "                nuntius, cogitator, probator"
    echo "  review [file] Print model comparison tables from the latest (or given) result file"
    exit 1
}

check_ollama() {
    local url="${OLLAMA_ROOT_URL:-http://localhost:11434}"
    echo -n "Checking Ollama at $url ... "
    if ! curl -sf "$url/api/tags" > /dev/null 2>&1; then
        echo "UNREACHABLE"
        echo "ERROR: Cannot connect to Ollama. Is it running?"
        echo "  Start with: ollama serve"
        echo "  Or set OLLAMA_ROOT_URL to the correct address."
        exit 1
    fi
    echo "OK"
}

case "${1:-}" in
    eval)
        check_ollama
        mkdir -p logs/eval
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        AGENT="${2:-}"
        if [ -n "$AGENT" ]; then
            TEST_PATH="tests/eval/test_${AGENT}.py"
            if [ ! -f "$TEST_PATH" ]; then
                echo "ERROR: Unknown agent '${AGENT}'. Valid agents: praetor, explorator, tabularius, nuntius, cogitator, probator"
                exit 1
            fi
            FILE="logs/eval/eval_${AGENT}_${TIMESTAMP}.json"
            echo "Running eval suite for agent '${AGENT}' → $FILE"
        else
            TEST_PATH=""
            FILE="logs/eval/eval_${TIMESTAMP}.json"
            echo "Running full eval suite → $FILE"
        fi
        echo "Partial results will be written to logs/eval/eval_partial.json during the run."
        uv run pytest -m eval \
            --json-report \
            --json-report-file="$FILE" \
            -v \
            $TEST_PATH \
            "${@:3}"
        echo ""
        echo "Results saved to $FILE"
        echo "Run './eval.sh review' to see the report."
        ;;

    review)
        FILE="${2:-}"
        if [ -z "$FILE" ]; then
            FILE=$(ls -t logs/eval/eval_*.json 2>/dev/null | head -1 || true)
            if [ -z "$FILE" ]; then
                echo "No logs/eval/eval_*.json found. Run './eval.sh eval' first."
                exit 1
            fi
        fi
        uv run python -m tests.eval.report_plugin "$FILE"
        ;;

    *)
        usage
        ;;
esac
