"""Регрессионные проверки слоя данных; отдельная тестовая модель в памяти.
Пользовательская condition_pipeline.joblib не изменяется и не переобучается.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fleet import make_fleet, bind_record, accept_event, discover_source, FIELD_KEYS
from neft_core import NUMERIC, CATEGORICAL, FEATURES, load_pipeline


def raw(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


class FleetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT/"data/kaggle1.csv"
        cls.df = pd.read_csv(cls.source)
        cls.meta = json.loads((ROOT/"models/condition_metadata.json").read_text(encoding="utf-8"))
        prep = ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        cls.model = Pipeline([("preprocessing", prep),
                              ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42))])
        cls.model.fit(cls.df[FEATURES], cls.df["Condition"])
        cls.base = make_fleet(cls.source.read_bytes(), cls.model, cls.meta, "test")

    def test_01_source_unchanged(self):
        self.assertEqual(self.base["source_hash"], hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_02_all_rows_kept(self):
        self.assertEqual(len(self.base["records"]), len(self.df))
        self.assertEqual(sum(self.base["counts"].values()), len(self.df))

    def test_03_probabilities_from_model(self):
        expected = self.model.predict_proba(self.df[FEATURES])
        for i, record in enumerate(self.base["records"]):
            for j, label in enumerate(self.model.classes_):
                self.assertAlmostEqual(record["probabilities"][label], expected[i, j])

    def test_04_target_not_used(self):
        a = make_fleet(raw(self.df.head(3).drop(columns=["Condition"])), self.model, self.meta)
        b = make_fleet(raw(self.df.head(3).assign(Condition="wrong")), self.model, self.meta)
        self.assertEqual(a["records"], b["records"])

    def test_05_unknown_material_not_normal(self):
        df = self.df.head(3).copy()
        df.loc[0, "Material"] = "not-seen"
        result = make_fleet(raw(df), self.model, self.meta)
        self.assertEqual(result["records"][0]["signal"], "Unknown")
        self.assertIsNone(result["records"][0]["label"])

    def test_06_bad_number_not_replaced(self):
        df = self.df.head(3).copy().astype({"Thickness_mm": object})
        df.loc[0, "Thickness_mm"] = "bad"
        record = make_fleet(raw(df), self.model, self.meta)["records"][0]
        self.assertFalse(record["valid"])
        self.assertIsNone(record["inputs"]["Thickness_mm"])
        json.dumps(record, allow_nan=False)

    def test_07_quality_flags_preserved(self):
        bad = next(r for r in self.base["records"] if r["inputs"]["Material_Loss_Percent"] > 100)
        self.assertEqual(bad["signal"], "Unknown")
        self.assertTrue(bad["notes"])
        self.assertGreater(bad["inputs"]["Material_Loss_Percent"], 100)

    def test_08_missing_column(self):
        with self.assertRaises(ValueError):
            make_fleet(raw(self.df.head(3).drop(columns=["Grade"])), self.model, self.meta)

    def test_09_duplicate_ids(self):
        with self.assertRaises(ValueError):
            make_fleet(raw(self.df.head(3).assign(well_id="duplicate")), self.model, self.meta)

    def test_10_partial_coordinates(self):
        with self.assertRaises(ValueError):
            make_fleet(raw(self.df.head(3).assign(latitude=50)), self.model, self.meta)

    def test_11_valid_coordinates_not_replaced(self):
        result = make_fleet(raw(self.df.head(3).assign(latitude=[50,51,52], longitude=[10,11,12])), self.model, self.meta)
        self.assertTrue(result["has_geo"])
        self.assertEqual(result["records"][0]["latitude"], 50)

    def test_12_invalid_coordinates(self):
        with self.assertRaises(ValueError):
            make_fleet(raw(self.df.head(3).assign(latitude=100, longitude=50)), self.model, self.meta)

    def test_13_bind_same_token_preserves_edits(self):
        record = self.base["records"][0]
        state = {}
        bind_record(state, record, ("a",))
        state[FIELD_KEYS["Temperature_C"]] = 101
        bind_record(state, record, ("a",))
        self.assertEqual(state[FIELD_KEYS["Temperature_C"]], 101)

    def test_14_selection_updates_all_fields(self):
        state = {}
        record = self.base["records"][1]
        bind_record(state, record, ("b",))
        for key, value in record["inputs"].items():
            self.assertEqual(state[FIELD_KEYS[key]], value)

    def test_15_manual_result_is_not_registry(self):
        record = self.base["records"][0]
        original = record["probabilities"]["Critical"]
        state = {}
        bind_record(state, record, ("a",))
        state["condition_result"]["probabilities"]["Critical"] = .333
        self.assertEqual(record["probabilities"]["Critical"], original)

    def test_16_event_deduplication(self):
        r = self.base["records"][0]
        records, state = {r["id"]: r}, {}
        event = {"id": r["id"], "nonce": "1", "action": "select"}
        self.assertTrue(accept_event(event, records, state))
        self.assertFalse(accept_event(event, records, state))

    def test_17_foreign_id_is_rejected(self):
        self.assertFalse(accept_event({"id":"missing","action":"ack","nonce":"2"}, {}, {}))

    def test_18_ack_does_not_change_signal(self):
        r = next(r for r in self.base["records"] if r["signal"]=="Critical")
        state, signal = {}, r["signal"]
        self.assertTrue(accept_event({"id":r["id"],"action":"ack","nonce":"3"}, {r["id"]:r}, state))
        self.assertEqual(r["signal"], signal)
        self.assertEqual(state["acknowledged_signals"][r["id"]], r["signature"])

    def test_19_changed_input_invalidates_ack_signature(self):
        df = self.df.head(1).copy()
        a = make_fleet(raw(df), self.model, self.meta)["records"][0]
        df["Temperature_C"] += 1
        b = make_fleet(raw(df), self.model, self.meta)["records"][0]
        self.assertNotEqual(a["signature"], b["signature"])

    def test_20_source_lookup(self):
        self.assertEqual(discover_source(ROOT), ROOT/"data/kaggle1.csv")

    def test_21_model_loader_same_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"test_model.joblib"
            joblib.dump(self.model, p)
            restored = load_pipeline(p)
            self.assertEqual(set(restored.classes_), {"Normal", "Moderate", "Critical"})

    def test_22_metadata_and_user_model_not_rewritten(self):
        self.assertEqual(self.meta["target"], "Condition")
        self.assertEqual(self.meta["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
