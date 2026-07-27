"""
Tests for GitHub issue #14: Track rule effectiveness from MCP usage.

Coverage:
  - LightRAGManager.record_feedback increments times_served, times_applied,
    or times_dismissed counters in ChromaDB metadata
  - Auto-promotion: rules served >=5 times with >70% acceptance rate are
    promoted from 'needs_review' to 'active'
  - GET /api/rules/{rule_id}/effectiveness returns per-rule effectiveness stats
  - GET /api/stats/effectiveness returns aggregate stats across all rules
  - POST /api/rules/{rule_id}/feedback calls record_feedback and returns
    the updated counters
"""

import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_db_with_mocked_chromadb():
    """Return a LightRAGManager with its ChromaDB PersistentClient fully mocked."""
    with patch("db.lightrag_manager.chromadb.PersistentClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_main_coll = MagicMock()
        mock_history_coll = MagicMock()
        mock_client.get_or_create_collection.side_effect = [
            mock_main_coll,
            mock_history_coll,
        ]

        from db.lightrag_manager import LightRAGManager
        db = LightRAGManager()

    db.collection = mock_main_coll
    db.history_collection = mock_history_coll
    return db, mock_main_coll


def _stub_rule_metadata(mock_collection, rule_id: str, metadata: dict):
    """Wire mock_collection.get to return a rule with the given metadata."""
    mock_collection.get.return_value = {
        "ids": [rule_id],
        "documents": [f"Rule: {rule_id} - Context: test. Enforce: test"],
        "metadatas": [metadata],
    }


# ---------------------------------------------------------------------------
# LightRAGManager.record_feedback — counter increments
# ---------------------------------------------------------------------------

class TestRecordFeedbackCounterIncrements(unittest.TestCase):
    """record_feedback must increment the correct counters in ChromaDB."""

    def test_applied_action_increments_times_applied_and_times_served(self):
        """Action 'applied' must increment both times_applied and times_served."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        _stub_rule_metadata(mock_coll, "rule-1", {
            "status": "active",
            "times_served": 2,
            "times_applied": 1,
            "times_dismissed": 1,
        })

        db.record_feedback("rule-1", "applied")

        call_kwargs = mock_coll.update.call_args
        updated = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(updated["times_served"], 3)
        self.assertEqual(updated["times_applied"], 2)
        self.assertEqual(updated["times_dismissed"], 1)

    def test_dismissed_action_increments_times_dismissed_and_times_served(self):
        """Action 'dismissed' must increment both times_dismissed and times_served."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        _stub_rule_metadata(mock_coll, "rule-2", {
            "status": "active",
            "times_served": 3,
            "times_applied": 2,
            "times_dismissed": 1,
        })

        db.record_feedback("rule-2", "dismissed")

        call_kwargs = mock_coll.update.call_args
        updated = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(updated["times_served"], 4)
        self.assertEqual(updated["times_applied"], 2)
        self.assertEqual(updated["times_dismissed"], 2)

    def test_record_feedback_raises_for_unknown_action(self):
        """An action other than 'applied' or 'dismissed' must raise ValueError."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        _stub_rule_metadata(mock_coll, "rule-3", {
            "status": "active",
            "times_served": 0,
            "times_applied": 0,
            "times_dismissed": 0,
        })

        with self.assertRaises(ValueError):
            db.record_feedback("rule-3", "ignored")

    def test_record_feedback_raises_when_rule_not_found(self):
        """record_feedback must raise ValueError if the rule does not exist."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        mock_coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        with self.assertRaises(ValueError):
            db.record_feedback("nonexistent-rule", "applied")

    def test_counters_default_to_zero_when_absent_from_metadata(self):
        """If a rule has no counter fields yet, they start from 0 before incrementing."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        _stub_rule_metadata(mock_coll, "rule-4", {
            "status": "active",
            # no times_served / times_applied / times_dismissed
        })

        db.record_feedback("rule-4", "applied")

        call_kwargs = mock_coll.update.call_args
        updated = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(updated["times_served"], 1)
        self.assertEqual(updated["times_applied"], 1)
        self.assertEqual(updated["times_dismissed"], 0)

    def test_record_feedback_calls_update_with_correct_rule_id(self):
        """update() must be called with the same rule_id passed to record_feedback."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        _stub_rule_metadata(mock_coll, "rule-5", {
            "status": "active",
            "times_served": 0,
            "times_applied": 0,
            "times_dismissed": 0,
        })

        db.record_feedback("rule-5", "dismissed")

        call_kwargs = mock_coll.update.call_args
        self.assertEqual(call_kwargs.kwargs["ids"], ["rule-5"])


# ---------------------------------------------------------------------------
# LightRAGManager.record_feedback — auto-promotion logic
# ---------------------------------------------------------------------------

class TestAutoPromotion(unittest.TestCase):
    """
    A rule in 'needs_review' that has been served >=5 times with >70% acceptance
    must be promoted to 'active'.
    """

    def test_rule_promoted_when_threshold_met(self):
        """
        After 5th applied feedback that pushes acceptance above 70%,
        status must flip from 'needs_review' to 'active'.
        """
        db, mock_coll = _make_db_with_mocked_chromadb()
        # 4 applied out of 4 served — adding 1 more applied gives 5/5 = 100%
        _stub_rule_metadata(mock_coll, "promo-1", {
            "status": "needs_review",
            "times_served": 4,
            "times_applied": 4,
            "times_dismissed": 0,
        })

        db.record_feedback("promo-1", "applied")

        call_kwargs = mock_coll.update.call_args
        updated = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(updated["status"], "active")

    def test_rule_not_promoted_when_below_min_serves(self):
        """
        A rule served only 4 times (below minimum 5) must not be promoted
        even if acceptance rate is 100%.
        """
        db, mock_coll = _make_db_with_mocked_chromadb()
        # 3 applied out of 3 served; after this feedback: 4/4 = 100% but only 4 serves
        _stub_rule_metadata(mock_coll, "promo-2", {
            "status": "needs_review",
            "times_served": 3,
            "times_applied": 3,
            "times_dismissed": 0,
        })

        db.record_feedback("promo-2", "applied")

        call_kwargs = mock_coll.update.call_args
        updated = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(updated["status"], "needs_review")

    def test_rule_not_promoted_when_acceptance_rate_too_low(self):
        """
        A rule served 5 times with <=70% acceptance must stay in 'needs_review'.
        Exactly 70% (3.5/5 rounded to 3/5 = 60%) is below threshold.
        """
        db, mock_coll = _make_db_with_mocked_chromadb()
        # 3 applied out of 5 served → 3/5 = 60% acceptance → not promoted
        _stub_rule_metadata(mock_coll, "promo-3", {
            "status": "needs_review",
            "times_served": 4,
            "times_applied": 3,
            "times_dismissed": 1,
        })

        db.record_feedback("promo-3", "dismissed")

        call_kwargs = mock_coll.update.call_args
        updated = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(updated["status"], "needs_review")

    def test_already_active_rule_stays_active_after_feedback(self):
        """Rules already 'active' must remain 'active' regardless of counters."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        _stub_rule_metadata(mock_coll, "active-1", {
            "status": "active",
            "times_served": 10,
            "times_applied": 9,
            "times_dismissed": 1,
        })

        db.record_feedback("active-1", "applied")

        call_kwargs = mock_coll.update.call_args
        updated = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(updated["status"], "active")

    def test_promotion_boundary_exactly_70_percent(self):
        """
        Acceptance rate must be strictly greater than 70% to trigger promotion.
        Exactly 70.0% must NOT promote.
        """
        db, mock_coll = _make_db_with_mocked_chromadb()
        # After this feedback: 7 applied, 10 served = exactly 70.0%
        _stub_rule_metadata(mock_coll, "promo-4", {
            "status": "needs_review",
            "times_served": 9,
            "times_applied": 6,
            "times_dismissed": 3,
        })

        db.record_feedback("promo-4", "applied")

        call_kwargs = mock_coll.update.call_args
        updated = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(updated["status"], "needs_review")

    def test_promotion_just_above_70_percent(self):
        """
        Acceptance rate just above 70% with >=5 serves must promote.
        E.g. 4 applied out of 5 served = 80% → promotes.
        """
        db, mock_coll = _make_db_with_mocked_chromadb()
        # After: 4 applied, 1 dismissed, 5 served = 80%
        _stub_rule_metadata(mock_coll, "promo-5", {
            "status": "needs_review",
            "times_served": 4,
            "times_applied": 3,
            "times_dismissed": 1,
        })

        db.record_feedback("promo-5", "applied")

        call_kwargs = mock_coll.update.call_args
        updated = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(updated["status"], "active")


# ---------------------------------------------------------------------------
# LightRAGManager — get_rule_effectiveness
# ---------------------------------------------------------------------------

class TestGetRuleEffectiveness(unittest.TestCase):
    """get_rule_effectiveness returns a stats dict for a single rule."""

    def test_returns_correct_counters_and_acceptance_rate(self):
        """acceptance_rate must equal times_applied / times_served."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        _stub_rule_metadata(mock_coll, "eff-1", {
            "status": "active",
            "times_served": 10,
            "times_applied": 7,
            "times_dismissed": 3,
        })

        stats = db.get_rule_effectiveness("eff-1")

        self.assertEqual(stats["rule_id"], "eff-1")
        self.assertEqual(stats["times_served"], 10)
        self.assertEqual(stats["times_applied"], 7)
        self.assertEqual(stats["times_dismissed"], 3)
        self.assertAlmostEqual(stats["acceptance_rate"], 0.7)

    def test_acceptance_rate_is_zero_when_never_served(self):
        """acceptance_rate must be 0.0 when times_served is 0 (no division by zero)."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        _stub_rule_metadata(mock_coll, "eff-2", {
            "status": "active",
            "times_served": 0,
            "times_applied": 0,
            "times_dismissed": 0,
        })

        stats = db.get_rule_effectiveness("eff-2")

        self.assertAlmostEqual(stats["acceptance_rate"], 0.0)

    def test_raises_when_rule_not_found(self):
        """get_rule_effectiveness must raise ValueError for unknown rule_id."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        mock_coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        with self.assertRaises(ValueError):
            db.get_rule_effectiveness("no-such-rule")

    def test_counters_default_to_zero_when_absent(self):
        """Rules without counter fields must report 0 for all counters."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        _stub_rule_metadata(mock_coll, "eff-3", {"status": "needs_review"})

        stats = db.get_rule_effectiveness("eff-3")

        self.assertEqual(stats["times_served"], 0)
        self.assertEqual(stats["times_applied"], 0)
        self.assertEqual(stats["times_dismissed"], 0)
        self.assertAlmostEqual(stats["acceptance_rate"], 0.0)


# ---------------------------------------------------------------------------
# LightRAGManager — get_all_effectiveness_stats
# ---------------------------------------------------------------------------

class TestGetAllEffectivenessStats(unittest.TestCase):
    """get_all_effectiveness_stats returns aggregate stats across all rules."""

    def _make_db_with_all_rules(self, rules: list[dict]):
        """Stub collection.get to return a list of rules for aggregate queries."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        mock_coll.get.return_value = {
            "ids": [r["rule_id"] for r in rules],
            "documents": [f"Rule: {r['rule_id']}" for r in rules],
            "metadatas": [
                {k: v for k, v in r.items() if k != "rule_id"} for r in rules
            ],
        }
        return db, mock_coll

    def test_returns_per_rule_stats_for_all_rules(self):
        """Result must include one entry per rule in the collection."""
        db, _ = self._make_db_with_all_rules([
            {"rule_id": "r1", "status": "active", "times_served": 5,
             "times_applied": 4, "times_dismissed": 1},
            {"rule_id": "r2", "status": "needs_review", "times_served": 2,
             "times_applied": 1, "times_dismissed": 1},
        ])

        result = db.get_all_effectiveness_stats()

        self.assertEqual(len(result["rules"]), 2)
        ids = [r["rule_id"] for r in result["rules"]]
        self.assertIn("r1", ids)
        self.assertIn("r2", ids)

    def test_aggregate_totals_computed_correctly(self):
        """total_served, total_applied, total_dismissed must sum across all rules."""
        db, _ = self._make_db_with_all_rules([
            {"rule_id": "r1", "status": "active", "times_served": 5,
             "times_applied": 4, "times_dismissed": 1},
            {"rule_id": "r2", "status": "active", "times_served": 10,
             "times_applied": 6, "times_dismissed": 4},
        ])

        result = db.get_all_effectiveness_stats()

        self.assertEqual(result["totals"]["total_served"], 15)
        self.assertEqual(result["totals"]["total_applied"], 10)
        self.assertEqual(result["totals"]["total_dismissed"], 5)

    def test_overall_acceptance_rate_calculated_from_totals(self):
        """overall_acceptance_rate must equal total_applied / total_served."""
        db, _ = self._make_db_with_all_rules([
            {"rule_id": "r1", "status": "active", "times_served": 4,
             "times_applied": 3, "times_dismissed": 1},
            {"rule_id": "r2", "status": "active", "times_served": 6,
             "times_applied": 3, "times_dismissed": 3},
        ])

        result = db.get_all_effectiveness_stats()

        # 6 applied out of 10 served = 0.6
        self.assertAlmostEqual(result["totals"]["overall_acceptance_rate"], 0.6)

    def test_overall_acceptance_rate_is_zero_when_nothing_served(self):
        """overall_acceptance_rate must be 0.0 when total_served is 0."""
        db, _ = self._make_db_with_all_rules([
            {"rule_id": "r1", "status": "active", "times_served": 0,
             "times_applied": 0, "times_dismissed": 0},
        ])

        result = db.get_all_effectiveness_stats()

        self.assertAlmostEqual(result["totals"]["overall_acceptance_rate"], 0.0)

    def test_returns_empty_rules_list_when_collection_empty(self):
        """An empty collection must return an empty rules list and zero totals."""
        db, mock_coll = _make_db_with_mocked_chromadb()
        mock_coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        result = db.get_all_effectiveness_stats()

        self.assertEqual(result["rules"], [])
        self.assertEqual(result["totals"]["total_served"], 0)
        self.assertAlmostEqual(result["totals"]["overall_acceptance_rate"], 0.0)


# ---------------------------------------------------------------------------
# API — POST /api/rules/{rule_id}/feedback
# ---------------------------------------------------------------------------

def _make_api_module():
    """Import api module with all heavy deps mocked out.

    Reloading api creates a fresh unbound LazyDbProxy as `api.db`. Patching
    LightRAGManager is not enough: nothing calls db.bind() outside the
    background embedding loader, so every endpoint touching `db` would wait on
    the proxy's readiness event until it timed out. Bind the mock explicitly so
    requests resolve against it immediately.
    """
    mock_db = MagicMock()
    mock_conf = MagicMock()
    mock_orch = MagicMock()

    with patch("db.lightrag_manager.LightRAGManager", return_value=mock_db), \
         patch("pipeline.config_manager.ConfigManager", return_value=mock_conf), \
         patch("pipeline.job_orchestrator.JobOrchestrator", return_value=mock_orch):
        import api as api_module
        importlib.reload(api_module)

    api_module.db.bind(mock_db)

    return api_module, mock_db


class TestFeedbackEndpoint(unittest.TestCase):
    """POST /api/rules/{rule_id}/feedback delegates to db.record_feedback."""

    def test_feedback_endpoint_exists_and_calls_record_feedback_with_applied(self):
        """Endpoint must call db.record_feedback(rule_id, 'applied')."""
        from fastapi.testclient import TestClient

        api_module, mock_db = _make_api_module()
        client = TestClient(api_module.app)

        response = client.post(
            "/api/rules/rule-abc/feedback",
            json={"action": "applied"},
        )

        self.assertEqual(response.status_code, 200)
        mock_db.record_feedback.assert_called_once_with("rule-abc", "applied")

    def test_feedback_endpoint_calls_record_feedback_with_dismissed(self):
        """Endpoint must call db.record_feedback(rule_id, 'dismissed')."""
        from fastapi.testclient import TestClient

        api_module, mock_db = _make_api_module()
        client = TestClient(api_module.app)

        response = client.post(
            "/api/rules/rule-xyz/feedback",
            json={"action": "dismissed"},
        )

        self.assertEqual(response.status_code, 200)
        mock_db.record_feedback.assert_called_once_with("rule-xyz", "dismissed")

    def test_feedback_endpoint_returns_400_for_invalid_action(self):
        """
        Endpoint must return 422 (pydantic validation) or 400 for invalid action strings.
        The action field must be constrained to 'applied' | 'dismissed'.
        """
        from fastapi.testclient import TestClient

        api_module, mock_db = _make_api_module()
        client = TestClient(api_module.app)

        response = client.post(
            "/api/rules/rule-xyz/feedback",
            json={"action": "unknown"},
        )

        self.assertIn(response.status_code, [400, 422])
        mock_db.record_feedback.assert_not_called()

    def test_feedback_endpoint_returns_404_when_rule_not_found(self):
        """Endpoint must return 404 when db.record_feedback raises ValueError."""
        from fastapi.testclient import TestClient

        api_module, mock_db = _make_api_module()
        mock_db.record_feedback.side_effect = ValueError("Rule not found")
        client = TestClient(api_module.app)

        response = client.post(
            "/api/rules/no-such-rule/feedback",
            json={"action": "applied"},
        )

        self.assertEqual(response.status_code, 404)

    def test_feedback_endpoint_response_body_contains_rule_id_and_action(self):
        """Response body must include rule_id and action for client confirmation."""
        from fastapi.testclient import TestClient

        api_module, mock_db = _make_api_module()
        client = TestClient(api_module.app)

        response = client.post(
            "/api/rules/rule-resp/feedback",
            json={"action": "applied"},
        )

        body = response.json()
        self.assertEqual(body["rule_id"], "rule-resp")
        self.assertEqual(body["action"], "applied")


# ---------------------------------------------------------------------------
# API — GET /api/rules/{rule_id}/effectiveness
# ---------------------------------------------------------------------------

class TestEffectivenessEndpoint(unittest.TestCase):
    """GET /api/rules/{rule_id}/effectiveness returns per-rule stats."""

    def test_effectiveness_endpoint_returns_stats_dict(self):
        """Endpoint must return the dict from db.get_rule_effectiveness."""
        from fastapi.testclient import TestClient

        api_module, mock_db = _make_api_module()
        mock_db.get_rule_effectiveness.return_value = {
            "rule_id": "rule-eff",
            "times_served": 10,
            "times_applied": 7,
            "times_dismissed": 3,
            "acceptance_rate": 0.7,
        }
        client = TestClient(api_module.app)

        response = client.get("/api/rules/rule-eff/effectiveness")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["rule_id"], "rule-eff")
        self.assertEqual(body["times_served"], 10)
        self.assertAlmostEqual(body["acceptance_rate"], 0.7)

    def test_effectiveness_endpoint_returns_404_when_rule_not_found(self):
        """Endpoint must return 404 when db.get_rule_effectiveness raises ValueError."""
        from fastapi.testclient import TestClient

        api_module, mock_db = _make_api_module()
        mock_db.get_rule_effectiveness.side_effect = ValueError("Rule not found")
        client = TestClient(api_module.app)

        response = client.get("/api/rules/missing-rule/effectiveness")

        self.assertEqual(response.status_code, 404)


# ---------------------------------------------------------------------------
# API — GET /api/stats/effectiveness
# ---------------------------------------------------------------------------

class TestBulkStatsEndpoint(unittest.TestCase):
    """GET /api/stats/effectiveness returns aggregate effectiveness data."""

    def test_bulk_stats_endpoint_returns_rules_and_totals(self):
        """Endpoint must return dict with 'rules' list and 'totals' dict."""
        from fastapi.testclient import TestClient

        api_module, mock_db = _make_api_module()
        mock_db.get_all_effectiveness_stats.return_value = {
            "rules": [
                {"rule_id": "r1", "times_served": 5, "times_applied": 4,
                 "times_dismissed": 1, "acceptance_rate": 0.8},
            ],
            "totals": {
                "total_served": 5,
                "total_applied": 4,
                "total_dismissed": 1,
                "overall_acceptance_rate": 0.8,
            },
        }
        client = TestClient(api_module.app)

        response = client.get("/api/stats/effectiveness")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("rules", body)
        self.assertIn("totals", body)
        self.assertEqual(len(body["rules"]), 1)
        self.assertAlmostEqual(body["totals"]["overall_acceptance_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()
