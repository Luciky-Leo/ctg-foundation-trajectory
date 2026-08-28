#!/usr/bin/env bash
set -euo pipefail

url="https://zenodo.org/api/records/21800730/files/JNU-CTG.zip/content"
out_root="/mnt/f/Database/99_incoming_review/jnu_ctg_zenodo_21800730_20260827"
parts_dir="${out_root}/parts_wsl"
total=156915297
chunk=4194304
parallel=8

mkdir -p "${parts_dir}"
n=$(( (total + chunk - 1) / chunk ))
export url parts_dir total chunk

download_part() {
  local i="$1"
  local start=$((i * chunk))
  local end=$((start + chunk - 1))
  if (( end >= total )); then
    end=$((total - 1))
  fi
  local expected=$((end - start + 1))
  local out
  out=$(printf "%s/part_%03d.bin" "${parts_dir}" "${i}")

  if [[ -f "${out}" ]] && [[ $(stat -c %s "${out}") -eq ${expected} ]]; then
    exit 0
  fi

  for attempt in 1 2 3 4 5; do
    rm -f "${out}.tmp"
    if curl -sS -L --fail --retry 3 --retry-all-errors \
      --connect-timeout 30 --max-time 900 --range "${start}-${end}" \
      -o "${out}.tmp" "${url}"; then
      local actual
      actual=$(stat -c %s "${out}.tmp")
      if [[ ${actual} -eq ${expected} ]]; then
        mv "${out}.tmp" "${out}"
        exit 0
      fi
    fi
    sleep 2
  done

  echo "Failed part ${i}" >&2
  exit 2
}
export -f download_part

seq 0 $((n - 1)) | xargs -P "${parallel}" -I{} bash -c 'download_part "$@"' _ {}

archive="${out_root}/JNU-CTG.zip"
tmp_archive="${archive}.assembling"
rm -f "${tmp_archive}"
for i in $(seq 0 $((n - 1))); do
  part=$(printf "%s/part_%03d.bin" "${parts_dir}" "${i}")
  cat "${part}" >> "${tmp_archive}"
done

actual_total=$(stat -c %s "${tmp_archive}")
if [[ ${actual_total} -ne ${total} ]]; then
  echo "Assembled size mismatch: ${actual_total} != ${total}" >&2
  exit 3
fi
mv "${tmp_archive}" "${archive}"
md5sum "${archive}"

