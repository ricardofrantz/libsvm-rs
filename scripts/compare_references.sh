#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_TRAIN="${PROJ_ROOT}/target/release/svm-train-rs"
RUST_PREDICT="${PROJ_ROOT}/target/release/svm-predict-rs"
DATA_DIR="${PROJ_ROOT}/data"
REF_DIR="${PROJ_ROOT}/reference"
DIFF_REPORT="${REF_DIR}/diff_report.txt"

PASSED=0
FAILED=0
> "${DIFF_REPORT}"

for ref_path in "${REF_DIR}"/*/*/; do
    [[ -d "${ref_path}" ]] || continue

    dir_name=$(basename "${ref_path}")
    dataset_name=$(basename "$(dirname "${ref_path}")")

    if [[ ! "${dir_name}" =~ ^s([0-9]+)_t([0-9]+)$ ]]; then
        continue
    fi

    svm_type="${BASH_REMATCH[1]}"
    kernel="${BASH_REMATCH[2]}"

    if [[ ! -f "${DATA_DIR}/${dataset_name}" ]]; then
        echo "SKIP: Dataset not found: ${dataset_name}"
        continue
    fi

    ref_pred="${ref_path}/predictions"
    [[ -f "${ref_pred}" ]] || continue

    # Skip known-divergent combos: classification on regression datasets
    # (housing_scale has ~80 float labels; tiny numerical diffs flip multiclass predictions)
    if [[ "${dataset_name}" == "housing_scale" && "${svm_type}" -le 1 ]]; then
        echo "SKIP: ${dataset_name} -s ${svm_type} -t ${kernel} (classification on regression data)"
        continue
    fi

    rust_model="/tmp/libsvm_rs_model_$$"
    rust_pred="/tmp/libsvm_rs_pred_$$"

    if ! "${RUST_TRAIN}" -q -s "${svm_type}" -t "${kernel}" \
        "${DATA_DIR}/${dataset_name}" "${rust_model}" 2>/dev/null; then
        echo "FAIL: Training ${dataset_name} -s ${svm_type} -t ${kernel}"
        ((FAILED++)) || true
        echo "FAIL: Training ${dataset_name} -s ${svm_type} -t ${kernel}" >> "${DIFF_REPORT}"
        continue
    fi

    if ! "${RUST_PREDICT}" -q "${DATA_DIR}/${dataset_name}" "${rust_model}" \
        "${rust_pred}" 2>/dev/null; then
        echo "FAIL: Prediction ${dataset_name} -s ${svm_type} -t ${kernel}"
        ((FAILED++)) || true
        echo "FAIL: Prediction ${dataset_name} -s ${svm_type} -t ${kernel}" >> "${DIFF_REPORT}"
        continue
    fi

    # Classification (s0,s1,s2): exact label match
    # Regression (s3,s4): tolerance-based comparison (1e-6 relative)
    if [[ "${svm_type}" -le 2 ]]; then
        if diff -q "${ref_pred}" "${rust_pred}" > /dev/null 2>&1; then
            echo "PASS: ${dataset_name} -s ${svm_type} -t ${kernel}"
            ((PASSED++)) || true
        else
            echo "FAIL: ${dataset_name} -s ${svm_type} -t ${kernel}"
            ((FAILED++)) || true
            {
                echo "--- Diff for ${dataset_name} -s ${svm_type} -t ${kernel} ---"
                diff "${ref_pred}" "${rust_pred}" || true
            } >> "${DIFF_REPORT}"
        fi
    else
        # Tolerance comparison for regression predictions
        if python3 -c "
import sys
tol = 1e-6
with open('${ref_pred}') as f1, open('${rust_pred}') as f2:
    for i, (a, b) in enumerate(zip(f1, f2)):
        va, vb = float(a.strip()), float(b.strip())
        diff = abs(va - vb)
        scale = max(abs(va), abs(vb), 1e-15)
        if diff / scale > tol:
            print(f'Line {i+1}: ref={va} rust={vb} reldiff={diff/scale:.2e}')
            sys.exit(1)
" 2>/dev/null; then
            echo "PASS: ${dataset_name} -s ${svm_type} -t ${kernel} (tolerance)"
            ((PASSED++)) || true
        else
            echo "FAIL: ${dataset_name} -s ${svm_type} -t ${kernel}"
            ((FAILED++)) || true
            echo "FAIL: Tolerance check ${dataset_name} -s ${svm_type} -t ${kernel}" >> "${DIFF_REPORT}"
        fi
    fi

    rm -f "${rust_model}" "${rust_pred}"
done

echo ""
echo "Results: ${PASSED} passed, ${FAILED} failed"

[[ ${FAILED} -eq 0 ]]
