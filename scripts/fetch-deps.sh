#!/usr/bin/env bash
# Downloads what Rookery uses but does not keep in git, checking each file's SHA-256:
#   - cutechess-cli 1.5.1 (GPL-3.0), which plays engine matches;
#   - Stockfish 19 (GPL-3.0), the opponent the rating is measured against;
#   - with --data: 12 million evaluated positions streamed from the Lichess evaluation database
#     (CC0), for training. The full file is tens of gigabytes; only its start is read.
#   scripts/fetch-deps.sh [--data]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tp="$root/third_party"
dl="$tp/downloads"
mkdir -p "$dl"
data=0
for a in "$@"; do
  case "$a" in
    --data) data=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

fetch() { # url sha256 file
  local url="$1" sha="$2" file="$dl/$3"
  if [ ! -f "$file" ] || ! echo "$sha  $file" | sha256sum -c - >/dev/null 2>&1; then
    echo "downloading $url"
    curl -fsSL --retry 3 -o "$file.part" "$url" || { echo "download failed: $url" >&2; exit 1; }
    echo "$sha  $file.part" | sha256sum -c - >/dev/null || { echo "checksum mismatch: $url" >&2; rm -f "$file.part"; exit 1; }
    mv "$file.part" "$file"
  fi
}

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    fetch https://github.com/cutechess/cutechess/releases/download/v1.5.1/cutechess-1.5.1-win64.zip \
      048942ca3473db860cb914fe94108da37051f693f47e368688ec6ec450f924bc cutechess.zip
    fetch https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip \
      3c8bf1f9ea66a09350a40df4f632288285ac206d99f33ab5842c408fc30b48a7 stockfish.zip
    mkdir -p "$tp/cutechess" "$tp/stockfish"
    unzip -qo "$dl/cutechess.zip" -d "$tp/cutechess"
    unzip -qo "$dl/stockfish.zip" -d "$tp/stockfish" ;;
  *)
    echo "On Linux, install cutechess-cli and stockfish with your package manager, and put them under $tp/cutechess and $tp/stockfish." ;;
esac

if [ $data = 1 ]; then
  mkdir -p "$root/data"
  echo "streaming 12 million positions from database.lichess.org (a few minutes)"
  curl -fsSL https://database.lichess.org/lichess_db_eval.jsonl.zst | zstd -dc 2>/dev/null |
    (cd "$root/trainer" && python -m rookery_train.fetch_evals --limit 12000000) > "$root/data/evals.csv" || true
  wc -l "$root/data/evals.csv"
fi
echo "done"
