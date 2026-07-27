#!/usr/bin/env bash
#
# Downloads pe_properties.csv from the Kaggle dataset:
# https://www.kaggle.com/datasets/rmjacobsen/property-listings-for-5-south-american-countries
#
# Requires Kaggle API credentials. Get a token at:
# https://www.kaggle.com/settings -> API tokens
# Any of these work:
#   - export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx
#   - `kaggle auth login` (writes ~/.kaggle/access_token)
#   - legacy kaggle.json at ~/.kaggle/kaggle.json (chmod 600)
#   - legacy KAGGLE_USERNAME + KAGGLE_KEY env vars

set -euo pipefail

DATASET="rmjacobsen/property-listings-for-5-south-american-countries"
FILE="pe_properties.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${REPO_ROOT}/data/raw"

if [[ -f "${DEST_DIR}/${FILE}" ]]; then
  echo "Already present: ${DEST_DIR}/${FILE} (delete it to force re-download)"
  exit 0
fi

if ! command -v kaggle &>/dev/null; then
  echo "Kaggle CLI not found. Run: source .venv/bin/activate && pip install -r ml/requirements.txt" >&2
  exit 1
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" \
   && ! -f "${HOME}/.kaggle/access_token" \
   && ! -f "${HOME}/.kaggle/kaggle.json" \
   && -z "${KAGGLE_USERNAME:-}" ]]; then
  echo "No Kaggle credentials found." >&2
  echo "Get a token at https://www.kaggle.com/settings -> API tokens, then either:" >&2
  echo "  export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx" >&2
  echo "  or run: kaggle auth login" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

echo "Downloading ${FILE} from ${DATASET}..."
kaggle datasets download \
  "${DATASET}" \
  -f "${FILE}" \
  -p "${DEST_DIR}" \
  -o

# The CLI's --unzip flag is unreliable across versions (silently no-ops on
# some), so extraction is handled here instead of trusting it.
ZIP_PATH="${DEST_DIR}/${FILE}.zip"
if [[ -f "${ZIP_PATH}" ]]; then
  echo "Unzipping ${FILE}.zip..."
  unzip -o -q "${ZIP_PATH}" -d "${DEST_DIR}"
  rm -f "${ZIP_PATH}"
fi

if [[ ! -f "${DEST_DIR}/${FILE}" ]]; then
  echo "Download reported success but ${DEST_DIR}/${FILE} is missing. Check the output above." >&2
  exit 1
fi

echo "Done: ${DEST_DIR}/${FILE}"
