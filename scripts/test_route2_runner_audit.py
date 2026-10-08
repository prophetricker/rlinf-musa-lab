#!/usr/bin/env python3
"""CPU regression checks for lifecycle evidence rejection and historical audits."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from audit_route2_runner_result import sampling_checks

ROOT = Path(__file__).resolve().parents[1]


class LifecycleAuditTests(unittest.TestCase):
    def setUp(self):
        def start(tasks, trials):
            return {"event": "training_bootstrap", "identities": [{
                "task_ids": tasks, "trial_ids": trials, "task_descriptions": ["task"] * 6}]}

        def end(row):
            identities = copy.deepcopy(row["identities"])
            for item in identities:
                item.pop("task_descriptions")
            return {"event": "training_horizon_complete", "identities": identities,
                    "elapsed_simulator_steps": [[240] * 6]}

        first = start([4, 8, 2, 6, 7, 1], [21, 34, 9, 34, 25, 21])
        second = start([7, 4, 5, 3, 3, 4], [28, 5, 40, 31, 32, 27])
        self.good = {"training_sampling_reports": [first, end(first), second, end(second)],
                     "training_reset": {"specific_reset_id": None},
                     "training_sampling_checks": {"valid": True}}

    def checks(self, data):
        return sampling_checks(data, 6, 240, 2, 2)

    def test_valid_sampling(self):
        self.assertTrue(all(self.checks(self.good).values()))

    def test_empty_or_reordered_reports_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["training_sampling_reports"] = []
        self.assertFalse(all(self.checks(bad).values()))
        bad = copy.deepcopy(self.good)
        bad["training_sampling_reports"].reverse()
        self.assertFalse(self.checks(bad)["sampling_events_alternate_and_cover_iterations"])

    def test_changed_trial_during_horizon_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["training_sampling_reports"][3]["identities"][0]["trial_ids"][0] += 1
        self.assertFalse(self.checks(bad)["sampling_task_trial_identity_stable"])

    def test_short_lane_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["training_sampling_reports"][3]["elapsed_simulator_steps"][0][-1] = 239
        self.assertFalse(self.checks(bad)["sampling_all_lanes_complete_horizon"])

    def test_repeated_reset_batch_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["training_sampling_reports"][2:] = copy.deepcopy(bad["training_sampling_reports"][:2])
        self.assertFalse(self.checks(bad)["sampling_reset_ids_advance"])

    def test_single_task_and_missing_lane_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["training_sampling_reports"][2]["identities"][0]["task_ids"] = [3] * 6
        self.assertFalse(self.checks(bad)["sampling_multiple_tasks"])
        bad["training_sampling_reports"][2]["identities"][0]["trial_ids"].pop()
        self.assertFalse(self.checks(bad)["sampling_lane_counts"])

    def test_zero_batch_rejected_before_loading_models(self):
        args = [sys.executable, str(ROOT / "routes/route2/model_probes/official_training_probe.py"),
                "--rlinf-source", "missing", "--gr00t-source", "missing", "--model-path", "missing",
                "--output", "/tmp/unused-lifecycle-test", "--global-batch-size", "0"]
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("global-batch-size must equal collected samples", result.stderr)

    def test_frozen_two_rank_results(self):
        evidence = ROOT / "routes/route2/evidence/multi-gpu"
        with tempfile.TemporaryDirectory() as temp:
            for case in ("save", "resume"):
                args = [sys.executable, str(ROOT / "scripts/audit_route2_runner_result.py"),
                        str(evidence / f"official-runner-{case}-v1.json"), "--probe",
                        str(evidence / "versions/official-training-lifecycle-v1-probe.py"), "--source-lock",
                        str(ROOT / "routes/route2/model_probes/two-rank-source-lock.json"),
                        "--output", str(Path(temp) / f"{case}.json")]
                if case == "resume":
                    args.extend(["--resume-reference", str(evidence / "official-runner-save-v1.json")])
                subprocess.run(args, check=True, capture_output=True)
                self.assertEqual(json.loads((Path(temp) / f"{case}.json").read_text())["status"], "pass")


if __name__ == "__main__":
    unittest.main()
