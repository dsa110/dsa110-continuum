#!/usr/bin/env bash
set -euo pipefail

# Lightweight cleanup gate:
# 1) show touched files
# 2) ruff check on touched Python files
# 3) targeted pytest selection
# 4) one CLI smoke check (--help) for touched script

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "ERROR: not inside a git repository"
  exit 2
fi
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniforge/envs/casa6/bin/python}"

print_header() {
  echo
  echo "== $1 =="
}

collect_changed_files() {
  (
    git diff --name-only
    git diff --name-only --cached
    git ls-files --others --exclude-standard
  ) | awk 'NF' | sort -u
}

declare -a FILES=()
if [[ "$#" -gt 0 ]]; then
  for p in "$@"; do
    FILES+=("$p")
  done
else
  while IFS= read -r p; do
    FILES+=("$p")
  done < <(collect_changed_files)
fi

if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "No touched files detected. Nothing to check."
  exit 0
fi

print_header "Touched Files"
for f in "${FILES[@]}"; do
  echo " - ${f}"
done

declare -a PY_FILES=()
for f in "${FILES[@]}"; do
  if [[ "${f}" == *.py ]] && [[ -f "${f}" ]]; then
    PY_FILES+=("${f}")
  fi
done

if [[ "${#PY_FILES[@]}" -gt 0 ]]; then
  print_header "Ruff Check"
  ruff check "${PY_FILES[@]}"
else
  print_header "Ruff Check"
  echo "No touched Python files; skipping ruff."
fi

declare -a TEST_TARGETS=()
for f in "${FILES[@]}"; do
  if [[ "${f}" == tests/test_*.py ]] && [[ -f "${f}" ]]; then
    TEST_TARGETS+=("${f}")
  fi
done

if [[ "${#TEST_TARGETS[@]}" -eq 0 ]]; then
  # Heuristic: script/module basename -> tests/test_<basename>.py
  for f in "${PY_FILES[@]}"; do
    base="$(basename "${f}" .py)"
    candidate="tests/test_${base}.py"
    if [[ -f "${candidate}" ]]; then
      TEST_TARGETS+=("${candidate}")
    fi
  done
fi

if [[ "${#TEST_TARGETS[@]}" -gt 0 ]]; then
  # De-duplicate
  mapfile -t TEST_TARGETS < <(printf "%s\n" "${TEST_TARGETS[@]}" | awk 'NF' | sort -u)
  print_header "Targeted Pytest"
  echo "${PYTHON_BIN} -m pytest ${TEST_TARGETS[*]} -q"
  "${PYTHON_BIN}" -m pytest "${TEST_TARGETS[@]}" -q
else
  print_header "Targeted Pytest"
  echo "No targeted tests discovered from touched files; skipping pytest."
fi

declare -a SCRIPT_SMOKES=()
for f in "${FILES[@]}"; do
  if [[ "${f}" == scripts/*.py ]] && [[ -f "${f}" ]]; then
    SCRIPT_SMOKES+=("${f}")
  fi
done

if [[ "${#SCRIPT_SMOKES[@]}" -gt 0 ]]; then
  print_header "CLI Smoke"
  smoke="${SCRIPT_SMOKES[0]}"
  echo "${PYTHON_BIN} ${smoke} --help"
  "${PYTHON_BIN}" "${smoke}" --help >/dev/null
  echo "OK: ${smoke} --help"
else
  print_header "CLI Smoke"
  echo "No touched scripts/*.py files; skipping CLI smoke."
fi

print_header "Gate Result"
echo "PASS"
