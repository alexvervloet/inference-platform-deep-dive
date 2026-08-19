from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hands_on.plan_fleet import (
    CANDIDATE_RELEASE,
    INVENTORY,
    REQUIRED_EVIDENCE,
    WORKLOADS,
    main,
    run_plan,
)
from inference_platform.admission import AdmissionAction, AdmissionDecision, AdmissionController


class FleetPlanTests(unittest.TestCase):
    def test_default_plan_is_release_ready_and_deterministic(self) -> None:
        first = run_plan()
        second = run_plan()
        self.assertEqual(first, second)
        self.assertTrue(first["release_ready"])
        self.assertEqual(first["violations"], [])

    def test_main_writes_machine_readable_release_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            self.assertEqual(main(["--output", str(output)]), 0)
            persisted = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(persisted["release_ready"])
        self.assertEqual(set(persisted["observed_evidence"]), REQUIRED_EVIDENCE)

    def test_dropping_a_workload_does_not_drop_its_requirement(self) -> None:
        reduced = tuple(item for item in WORKLOADS if item.name != "generation")
        report = run_plan(workloads=reduced)
        self.assertFalse(report["release_ready"])
        self.assertNotIn("workload:generation", report["observed_evidence"])
        self.assertIn("workload:generation", report["required_evidence"])

    def test_ineligible_inventory_removes_placement_evidence(self) -> None:
        too_small = tuple(replace(gpu, free_memory_gib=30) for gpu in INVENTORY)
        report = run_plan(inventory=too_small)
        self.assertFalse(report["release_ready"])
        self.assertNotIn("gpu-placement", report["observed_evidence"])
        self.assertIn("memory", report["deciding_reasons"]["gpu-placement"])

    def test_canary_regression_removes_rollout_evidence(self) -> None:
        regressed = replace(CANDIDATE_RELEASE, p95_tpot_seconds=0.3)
        report = run_plan(candidate_release=regressed)
        self.assertFalse(report["release_ready"])
        self.assertNotIn("canary-gate", report["observed_evidence"])
        self.assertIn("p95 TPOT", report["deciding_reasons"]["canary-gate"])

    def test_bypassing_oversize_shedding_fails_the_evidence_gate(self) -> None:
        original = AdmissionController.decide

        def bypass(
            controller: AdmissionController, request: object, **kwargs: float
        ) -> AdmissionDecision:
            if getattr(request, "identifier") == "unbounded-request":
                return AdmissionDecision(AdmissionAction.ADMIT, 9000, "test bypass")
            return original(controller, request, **kwargs)

        with mock.patch.object(AdmissionController, "decide", bypass):
            report = run_plan()
        self.assertFalse(report["release_ready"])
        self.assertNotIn("oversize-shedding", report["observed_evidence"])

    def test_benign_work_is_decided_by_real_admission_and_placement_paths(self) -> None:
        report = run_plan()
        self.assertEqual(report["decisions"]["benign_admission"]["action"], "admit")
        self.assertTrue(report["decisions"]["placement"]["placed"])
        self.assertIn("reservation", report["deciding_reasons"]["benign-admission"])

    def test_every_primary_decision_names_its_reason(self) -> None:
        reasons = run_plan()["deciding_reasons"]
        self.assertEqual(
            set(reasons),
            {
                "memory-fit",
                "capacity-plan",
                "parallel-layout",
                "gpu-placement",
                "benign-admission",
                "oversize-shedding",
                "queue-scaling",
                "canary-gate",
            },
        )
        self.assertTrue(all(reason.strip() for reason in reasons.values()))


if __name__ == "__main__":
    unittest.main()
