"""Exercise the actual solver and exporter cells from the self-contained notebook."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import networkx as nx
import numpy as np


EN_FOLDER = Path(__file__).resolve().parents[1]
NOTEBOOK = EN_FOLDER / "Student_Graph_Signal_Assignment_MSSV_MODULO.ipynb"


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.functions = {"__name__": "assignment_under_test"}
        for tag in ("recovery-functions", "submission-function"):
            cells = [cell for cell in notebook["cells"]
                     if tag in cell.get("metadata", {}).get("tags", [])]
            assert len(cells) == 1, f"Expected exactly one {tag} cell"
            exec(compile("".join(cells[0]["source"]), f"{NOTEBOOK}:{tag}", "exec"), cls.functions)

    def test_weighted_laplacian_preserves_node_order_and_edge_penalty(self):
        graph = nx.Graph()
        graph.add_nodes_from([2, 0, 1])
        graph.add_weighted_edges_from([(0, 1, 2), (1, 2, 3)])
        adjacency, laplacian = self.functions["prepare_graph"](graph, [1, 3, 7])
        np.testing.assert_allclose(adjacency, [[0, 2, 0], [2, 0, 3], [0, 3, 0]])
        np.testing.assert_allclose(laplacian.sum(axis=1), 0)
        values = np.array([1, 3, 7])
        self.assertAlmostEqual(values @ laplacian @ values, 2 * 2**2 + 3 * 4**2)

    def test_two_sensor_quadratic_solution(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=2)
        _, laplacian = self.functions["prepare_graph"](graph, [0, 10])
        fit = self.functions["fit_graph_signal"](laplacian, [0, 10], 0.5)
        np.testing.assert_allclose(fit["x"], [10 / 3, 20 / 3])
        self.assertTrue(fit["converged"])

    def test_quadratic_preserves_mean_and_reduces_graph_roughness(self):
        graph = nx.path_graph(6)
        values = np.array([1, 2, 1, 7, 2, 1], dtype=float)
        _, laplacian = self.functions["prepare_graph"](graph, values)
        fit = self.functions["fit_graph_signal"](laplacian, values, 0.3)
        self.assertAlmostEqual(fit["x"].mean(), values.mean())
        self.assertLess(fit["x"] @ laplacian @ fit["x"], values @ laplacian @ values)

    def test_huber_limits_the_influence_of_a_synthetic_outlier(self):
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4))
        truth = np.full(16, 5.0)
        values = truth.copy()
        values[5] += 30
        _, laplacian = self.functions["prepare_graph"](graph, values)
        ordinary = self.functions["fit_graph_signal"](laplacian, values, 1)
        robust = self.functions["fit_graph_signal"](laplacian, values, 1, delta=0.5)
        self.assertTrue(robust["converged"])
        self.assertLess(np.linalg.norm(robust["x"] - truth),
                        np.linalg.norm(ordinary["x"] - truth) / 5)
        self.assertLess(robust["weights"][5], 0.1)
        self.assertLessEqual(robust["relative_gradient"], 1e-8)
        history = np.array(robust["objective_history"])
        self.assertTrue(np.all(np.diff(history) <= 1e-9 * np.maximum(1, np.abs(history[:-1]))))

    def test_withheld_value_cannot_leak_into_fitting_or_scale(self):
        graph = nx.cycle_graph(6)
        values = np.array([1, 2, 1.4, 7, 2.1, 1.7])
        adjacency, laplacian = self.functions["prepare_graph"](graph, values)
        mask = np.ones(6, dtype=bool)
        mask[3] = False
        scale = self.functions["noise_scale"](adjacency, values, mask)
        original = self.functions["fit_graph_signal"](laplacian, values, 0.1, scale, mask)
        for hidden_value in (1e12, -1e12, np.nan):
            changed = values.copy()
            changed[3] = hidden_value
            self.assertEqual(self.functions["noise_scale"](adjacency, changed, mask), scale)
            fit = self.functions["fit_graph_signal"](laplacian, changed, 0.1, scale, mask)
            np.testing.assert_allclose(fit["x"], original["x"], rtol=0, atol=1e-12)
            self.assertEqual(fit["weights"][3], 0)

    def test_constant_signal_and_degenerate_noise_scale(self):
        graph = nx.path_graph(5)
        values = np.full(5, 18.0)
        adjacency, laplacian = self.functions["prepare_graph"](graph, values)
        scale = self.functions["noise_scale"](adjacency, values, np.ones(5, dtype=bool))
        self.assertGreater(scale, 0)
        for delta in (scale, np.inf):
            fit = self.functions["fit_graph_signal"](laplacian, values, 100, delta)
            self.assertTrue(fit["converged"])
            np.testing.assert_allclose(fit["x"], values)

    def test_two_node_validation_predicts_only_the_other_reading(self):
        graph = nx.path_graph(2)
        adjacency, laplacian = self.functions["prepare_graph"](graph, [2, 8])
        result = self.functions["evaluate_methods"](
            adjacency, laplacian, [2, 8], smoothing_grid=[0.1, 1],
            huber_factors=[1.5], max_expansions=0
        )
        for row in result["rows"]:
            np.testing.assert_allclose(row["predictions"], [8, 2])
            self.assertAlmostEqual(row["rmse"], 6)
            self.assertAlmostEqual(row["mae"], 6)

    def test_example_case_reproduces_documented_comparison(self):
        case = json.loads((EN_FOLDER / "public_cases.json").read_text())["cases"]["20260001"]
        graph = nx.Graph()
        graph.add_nodes_from(case["nodes"])
        graph.add_weighted_edges_from(case["edges"])
        adjacency, laplacian = self.functions["prepare_graph"](graph, case["y"])
        result = self.functions["evaluate_methods"](adjacency, laplacian, case["y"])
        self.assertEqual(result["selected"]["method"], "quadratic")
        self.assertEqual(result["selected"]["smoothing"], 0.1)
        self.assertAlmostEqual(result["selected"]["rmse"], 0.7309479012, places=8)
        self.assertEqual(result["x_hat"].shape, (32,))
        self.assertTrue(np.all(np.isfinite(result["x_hat"])))
        self.assertFalse(np.allclose(result["x_hat"], case["y"]))
        self.assertTrue(all(row["converged_folds"] == 32 for row in result["rows"]))
        expanded = self.functions["evaluate_methods"](
            adjacency, laplacian, case["y"], smoothing_grid=[3, 10], huber_factors=[1.5]
        )
        self.assertTrue(any(row["smoothing"] is not None and row["smoothing"] < 3
                            for row in expanded["rows"]))

    def test_invalid_inputs_are_rejected(self):
        invalid_graphs = [nx.DiGraph([(0, 1)]), nx.empty_graph(2), nx.path_graph([1, 2])]
        for weight in (-1, 0, np.inf, np.nan):
            graph = nx.Graph()
            graph.add_edge(0, 1, weight=weight)
            invalid_graphs.append(graph)
        for graph in invalid_graphs:
            with self.subTest(graph=repr(graph)), self.assertRaises(ValueError):
                self.functions["prepare_graph"](graph, [1, 2])
        graph = nx.path_graph(2)
        for values in ([1], [1, np.nan], [[1], [2]]):
            with self.assertRaises(ValueError):
                self.functions["prepare_graph"](graph, values)
        _, laplacian = self.functions["prepare_graph"](graph, [1, 2])
        for smoothing in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                self.functions["fit_graph_signal"](laplacian, [1, 2], smoothing)
        for delta in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                self.functions["fit_graph_signal"](laplacian, [1, 2], 0.1, delta)

    def test_csv_contract_and_invalid_predictions(self):
        self.functions.update(y=np.array([1.0, 2.0, 3.0]), MSSV_DAI_DIEN="0012",
                              NHOM_MSSV=["0012", "0034"], loaded_representative_mssv="0012")
        save = self.functions["save_submission"]
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(folder)
                filename = save([1.1, 2.2, 3.3])
                self.assertEqual(filename, "submission_0012.csv")
                with open(filename, encoding="utf-8-sig", newline="") as file:
                    reader = csv.DictReader(file)
                    self.assertEqual(reader.fieldnames,
                                     ["representative_mssv", "group_mssvs", "node_id", "x_hat"])
                    rows = list(reader)
                self.assertEqual([row["node_id"] for row in rows], ["0", "1", "2"])
                self.assertTrue(all(row["representative_mssv"] == "0012" for row in rows))
                self.assertTrue(all(row["group_mssvs"] == "0012|0034" for row in rows))
                np.testing.assert_allclose([float(row["x_hat"]) for row in rows], [1.1, 2.2, 3.3])
                for prediction in ([1, 2], [1, 2, np.nan], [1, 2, np.inf]):
                    with self.assertRaises(ValueError):
                        save(prediction)
                for group in ([], ["0012", ""], ["0012", "0012"], ["0034"], ["0012", "a|b"]):
                    with self.assertRaises(ValueError):
                        save([1, 2, 3], group_mssvs=group)
                with self.assertRaises(ValueError):
                    save([1, 2, 3], representative_mssv="0034")
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
