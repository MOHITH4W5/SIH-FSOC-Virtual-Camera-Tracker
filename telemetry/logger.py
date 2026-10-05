"""
telemetry/logger.py - High-Resolution Telemetry Time-Series Recorder.
Exports frame-by-frame simulation, estimation, control, and error records in JSON and CSV formats.
"""

import os
import csv
import json
from typing import Dict, Any, List


class TelemetryLogger:
    """Logs per-frame telemetry records to disk for reproducible benchmarking."""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.records: List[Dict[str, Any]] = []
        self._csv_writer = None
        self._csv_file = None

    def log_frame(self, record: Dict[str, Any]):
        """Appends a frame telemetry dictionary."""
        self.records.append(record)

    def export_csv(self, filename: str = "telemetry.csv") -> str:
        filepath = os.path.join(self.output_dir, filename)
        if not self.records:
            return filepath

        keys = list(self.records[0].keys())
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rec in self.records:
                # Flatten any nested structures
                row = {}
                for k, v in rec.items():
                    if isinstance(v, (list, tuple)):
                        row[k] = ";".join(map(str, v))
                    elif isinstance(v, dict):
                        row[k] = json.dumps(v)
                    else:
                        row[k] = v
                writer.writerow(row)
        return filepath

    def export_json(self, filename: str = "telemetry.json") -> str:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)
        return filepath

    def clear(self):
        self.records.clear()
