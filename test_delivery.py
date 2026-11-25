"""Test cases for the Mini Food Delivery Assistant.  Run: python -m unittest -v"""
import unittest
import delivery_assistant as d


class TestAStar(unittest.TestCase):
    def setUp(self):
        self.start, self.goal = d.parse_map(d.CITY_MAP)

    def test_shortest_route_cost(self):
        path, cost, _ = d.a_star(d.CITY_MAP, self.start, self.goal)
        self.assertEqual(cost, 10)                       # all main road
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)

    def test_route_never_enters_buildings(self):
        path, _, _ = d.a_star(d.CITY_MAP, self.start, self.goal)
        self.assertTrue(all(d.CITY_MAP[r][c] != "#" for r, c in path))

    def test_detour_avoids_cell_and_costs_more(self):
        path, cost, _ = d.a_star(d.CITY_MAP, self.start, self.goal, avoid={(1, 3)})
        self.assertNotIn((1, 3), path)
        self.assertEqual(cost, 14)

    def test_unreachable_goal(self):
        walled = ["S#", "#C"]
        path, cost, _ = d.a_star(walled, (0, 0), (1, 1))
        self.assertIsNone(path)
        self.assertEqual(cost, float("inf"))


class TestBayes(unittest.TestCase):
    def test_alert_raises_probability(self):
        post, *_ = d.bayes_update(0.20, alert=True)
        self.assertAlmostEqual(post, 0.60, places=3)     # 0.18 / 0.30

    def test_no_alert_lowers_probability(self):
        post, *_ = d.bayes_update(0.20, alert=False)
        self.assertAlmostEqual(post, 0.02 / 0.70, places=4)

    def test_posteriors_sum_to_one(self):
        post, lb, lnb, ev = d.bayes_update(0.3, alert=True)
        self.assertAlmostEqual(post + lnb * 0.7 / ev, 1.0)


class TestMDP(unittest.TestCase):
    def setUp(self):
        self.R = d.build_rewards(10, 14)
        self.V = d.initial_values(self.R)

    def test_bellman_hand_calculation(self):
        Q = d.bellman_q(self.R, self.V)
        self.assertAlmostEqual(Q["Blocked"]["Continue"], -36.52)
        self.assertAlmostEqual(Q["Clear"]["Continue"], -19.36)

    def test_certain_states(self):
        self.assertEqual(d.decide(self.R, 0.0, self.V)[0], "Continue")
        self.assertEqual(d.decide(self.R, 1.0, self.V)[0], "Reroute")

    def test_break_even_threshold(self):
        p = d.break_even(self.R, self.V)
        self.assertEqual(d.decide(self.R, p - 0.01, self.V)[0], "Continue")
        self.assertEqual(d.decide(self.R, p + 0.01, self.V)[0], "Reroute")


class TestPipeline(unittest.TestCase):
    def test_alert_scenario_reroutes(self):
        out = d.run(alert=True, quiet=True)
        self.assertEqual(out["action"], "Reroute")
        self.assertNotIn((1, 3), out["final_path"])

    def test_silent_sensor_continues(self):
        out = d.run(alert=False, quiet=True)
        self.assertEqual(out["action"], "Continue")
        self.assertEqual(out["final_path"], out["path"])

    def test_low_prior_alert_not_enough(self):
        # quiet-hour prior: even after an alert P(B) stays below break-even
        out = d.run(alert=True, prior=0.03, quiet=True)
        self.assertEqual(out["action"], "Continue")


class TestEdgeCases(unittest.TestCase):
    def test_sensor_never_on_restaurant_or_customer(self):
        res = d.solve(["S.C"])
        self.assertEqual(res["alert_cell"], (0, 1))

    def test_no_detour_forces_continue(self):
        res = d.solve(["S.C"], alert=True, prior=0.9)
        self.assertEqual(res["action"], "Continue")
        self.assertIn("note", res)

    def test_walled_customer_reports_error(self):
        self.assertIn("error", d.solve(["S#", "#C"]))

    def test_custom_sensor_cell_is_used(self):
        res = d.solve(d.CITY_MAP, alert_cell=(3, 4))
        self.assertEqual(res["alert_cell"], (3, 4))

    def test_sensor_off_route_falls_back_to_default(self):
        res = d.solve(d.CITY_MAP, alert_cell=(5, 0))
        self.assertEqual(res["alert_cell"], (1, 3))


class TestWebAPI(unittest.TestCase):
    def test_bad_map_rejected(self):
        import app
        for bad in (["S.", "C"], ["S.X", "..C"], ["...", "..C"]):
            with self.assertRaises(ValueError):
                app.validate_map(bad)

    def test_solve_endpoint_is_json_safe(self):
        import json
        import app
        out = app.to_json(app.handle_solve({"alert": True}))
        self.assertEqual(json.loads(json.dumps(out))["action"], "Reroute")


if __name__ == "__main__":
    unittest.main(verbosity=2)
