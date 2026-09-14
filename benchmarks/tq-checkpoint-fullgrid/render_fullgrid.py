#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render two recorded allocation summaries; never launch or modify experiments."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

LEVELS = (8192, 16384, 32768, 65536, 131072)
BACKENDS = ("simple", "mooncake")
FIELDS = (
    "backend",
    "rows",
    "context",
    "logical_gib",
    "save_s",
    "load_s",
    "correctness",
    "run_dir",
)
EXPECTED = {(r, c, b) for r in LEVELS for c in LEVELS for b in BACKENDS}
SOURCE_SHA = "428520fccb081381b73cdc1efa556a3fa7a3515b"
TQ_SHA = "c51614308b68c8d7a87c9b3ef62d59e14c69bde2"


def read_cells(data, allowed_rows):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    if not set(FIELDS).issubset(reader.fieldnames or []):
        raise ValueError("Summary CSV is missing required columns")
    cells = {}
    for row in reader:
        if None in row or any(row.get(field) is None for field in FIELDS):
            raise ValueError("Malformed CSV row")
        key = int(row["rows"]), int(row["context"]), row["backend"]
        if key not in EXPECTED or key[0] not in allowed_rows or key in cells:
            raise ValueError(f"Unexpected, wrong-allocation, or duplicate cell: {key}")
        values = [float(row[field]) for field in ("save_s", "load_s", "logical_gib")]
        expected_gib = key[0] * (key[1] * 20 + 12) / 1024**3
        if (
            row["correctness"] != "pass"
            or not row["run_dir"].strip()
            or not all(math.isfinite(value) and value > 0 for value in values)
            or not math.isclose(values[2], expected_gib, rel_tol=1e-12)
        ):
            raise ValueError(f"Invalid or non-passing measurement: {key}")
        cells[key] = row
    return cells


def prepare(parts):
    """Validate the independent low/high allocation records, including partials."""
    cells, inputs = {}, []
    if len(parts) != 2:
        raise ValueError("Exactly two allocation inputs are required")
    for (label, allowed), (path, status) in zip(
        (("low", LEVELS[:3]), ("high", LEVELS[3:])), parts, strict=True
    ):
        job_id, state, exit_code = status
        if not job_id.isdecimal() or not state.strip() or not exit_code.strip():
            raise ValueError("Each input needs a numeric job ID, state, and exit code")
        source = path.resolve()
        if any(
            item["job_id"] == job_id or item["source"] == str(source) for item in inputs
        ):
            raise ValueError("Allocation job IDs and CSV paths must be distinct")
        raw = source.read_bytes()
        observed = read_cells(raw, allowed)
        if cells.keys() & observed.keys():
            raise ValueError("Inputs overlap")
        cells.update(observed)
        inputs.append(
            {
                "label": label,
                "job_id": job_id,
                "state": state.strip().upper(),
                "exit_code": exit_code.strip(),
                "source": str(source),
                "raw_csv": f"raw-{label}.csv",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "passing_cases": len(observed),
                "expected_cases": len(allowed) * len(LEVELS) * len(BACKENDS),
                "bytes": raw,
            }
        )
    complete = cells.keys() == EXPECTED and all(
        item["state"] == "COMPLETED" and item["exit_code"] == "0:0" for item in inputs
    )
    result = {
        "complete": complete,
        "passing_backend_cases": len(cells),
        "expected_backend_cases": len(EXPECTED),
        "repetitions_per_backend_cell": 1,
        "intended_source": {"nemo_rl": SOURCE_SHA, "transferqueue": TQ_SHA},
        "inputs": [
            {key: value for key, value in item.items() if key != "bytes"}
            for item in inputs
        ],
        "missing_cells": [list(key) for key in sorted(EXPECTED - cells.keys())],
        "measurements_as_recorded": [cells[key] for key in sorted(cells)],
    }
    return result, cells, inputs


def table(cells, metric, backend=None):
    lines = [
        "| Rows / context | " + " | ".join(map(str, LEVELS)) + " |",
        "|---:|" + "---:|" * len(LEVELS),
    ]
    for rows in LEVELS:
        values = []
        for context in LEVELS:
            if backend is not None:
                row = cells.get((rows, context, backend))
                value = float(row[metric]) if row else None
            else:
                simple = cells.get((rows, context, "simple"))
                mooncake = cells.get((rows, context, "mooncake"))
                value = (
                    float(mooncake[metric]) / float(simple[metric])
                    if simple and mooncake
                    else None
                )
            values.append(f"{value:.6f}" if value is not None else "—")
        lines.append(f"| {rows} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def render(result, cells):
    status = (
        "Complete: 50/50 passing backend cases; both recorded Slurm jobs COMPLETED 0:0."
        if result["complete"]
        else f"NOT COMPLETE: {len(cells)}/50 passing backend cases, or job success not confirmed."
    )
    sections = [
        "# Simple versus Mooncake: full production grid",
        status,
        "Four OCI-HSG nodes; two storage workers per node (eight total). "
        "One measurement per backend and shape (n=1), not three repetitions. "
        "Each shape's Simple/Mooncake pair runs within the same allocation. "
        "The low-row and high-row bands use separate allocations that may overlap in time.",
        "Rows and context lengths below are exact counts. Latencies are seconds; lower is faster. "
        "Tables round to six decimal places; the two archived raw CSV files preserve every input byte.",
    ]
    for metric, title in (("save_s", "Save"), ("load_s", "Load")):
        for backend in BACKENDS:
            sections.extend(
                (
                    f"## {backend.title()} {title.lower()} (seconds)",
                    table(cells, metric, backend),
                )
            )
    for metric, title in (("save_s", "Save"), ("load_s", "Load")):
        sections.extend((f"## {title}: Mooncake / Simple", table(cells, metric)))
    sections.extend(
        (
            "A ratio above 1 means Mooncake is slower; below 1 means Mooncake is faster. "
            "A dash is missing evidence, never a zero or estimate. One sample supplies no variability estimate. "
            "The shared Lustre filesystem may serve both allocations concurrently. "
            "Pairing on the same hosts and alternating backend order do not guarantee identical "
            "filesystem-load or cache effects for the two backends. "
            "Fresh-process load may benefit from host/filesystem caches.",
            "These are TQ checkpoint API wall times, not total Slurm duration or end-to-end training "
            "checkpoint time. Payload preparation and post-load correctness verification are outside the timer. "
            "Correctness checks all logical keys and aggregate restored bytes, plus exact values/tags for "
            "64 sampled rows; it is not exhaustive payload validation. Mooncake retains fsync; "
            "Simple keeps its as-shipped durability behavior. Both use the same logical train-ready payload.",
            "## Recorded provenance",
            f"Intended NeMo-RL source: `{SOURCE_SHA}`; TransferQueue: `{TQ_SHA}`. "
            "Source/container/runtime hashes, actor placement, resolved commands, and application "
            "completion markers must also pass the campaign evidence audit. This offline renderer "
            "does not query Slurm or independently attest those runtime facts.",
        )
    )
    for item in result["inputs"]:
        sections.append(
            f"- {item['label']}: job `{item['job_id']}`, state `{item['state']}`, exit "
            f"`{item['exit_code']}`; {item['passing_cases']}/{item['expected_cases']} passing cases. "
            f"Raw CSV: `{item['raw_csv']}`, SHA-256 `{item['sha256']}`."
        )
    return "\n\n".join(sections) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("low", "high"):
        parser.add_argument(f"--{label}-summary", type=Path, required=True)
        parser.add_argument(
            f"--{label}-status",
            nargs=3,
            metavar=("JOB_ID", "STATE", "EXIT_CODE"),
            required=True,
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result, cells, inputs = prepare(
        [
            (args.low_summary, args.low_status),
            (args.high_summary, args.high_status),
        ]
    )
    markdown = render(result, cells)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for item in inputs:
        with (args.output_dir / item["raw_csv"]).open("xb") as stream:
            stream.write(item["bytes"])
    with (args.output_dir / "report.md").open("x") as stream:
        stream.write(markdown)
    with (args.output_dir / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        f"Rendered {len(cells)}/50 passing backend cases; complete={result['complete']}"
    )


if __name__ == "__main__":
    main()
