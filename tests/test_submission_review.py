"""Meaningful invariants for the aggregate-score review."""
import sys,unittest
from pathlib import Path
import numpy as np
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from ap2026q2_submission_review import (
    cohort_posterior,score_count_candidates,displayed_score,net_distribution,
    predictive_net,shared_population_grid,load_snapshot,joint_integer_bound,
)
class SubmissionReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.d=load_snapshot(ROOT)
    def test_ambiguous_single_zero_is_private_not_a_label(self):
        p=cohort_posterior(1,0)
        self.assertAlmostEqual(p["expected_public_rows"],0)
        self.assertAlmostEqual(p["theta_mean"],.5)
        np.testing.assert_allclose(p["finite_correct_count_pmf"],[.5,.5])
    def test_extreme_observation_forces_finite_success_not_population_perfection(self):
        p=cohort_posterior(2,2)
        self.assertAlmostEqual(p["expected_public_rows"],2)
        self.assertAlmostEqual(p["theta_mean"],.75)
        self.assertAlmostEqual(p["finite_cohort_accuracy_mean"],1)
        np.testing.assert_allclose(p["theta_ci95"],beta.ppf([.025,.975],3,1))
        q=cohort_posterior(2,-2)
        self.assertAlmostEqual(q["theta_mean"],.25)
        self.assertAlmostEqual(q["finite_cohort_accuracy_mean"],0)
    def test_display_roundtrip_and_all_historical_differences(self):
        observed=[r for r in self.d["history"] if r["status"]=="observed"]
        control=next(r for r in observed if r["name"]=="submission_bayes_verified_only.csv")
        differences=[]
        for n,mode in [(21000,"truncate"),(20998,"round")]:
            base=score_count_candidates(control["public_score"],n,mode)
            self.assertEqual(len(base),1);ds=[]
            for r in observed:
                values=score_count_candidates(r["public_score"],n,mode)
                self.assertEqual(len(values),1)
                self.assertAlmostEqual(displayed_score(int(values[0]),n,mode),r["public_score"])
                ds.append(int(values[0]-base[0]))
            differences.append(ds)
        self.assertEqual(differences[0],differences[1])
    def test_one_case_predictive_law(self):
        np.testing.assert_allclose(net_distribution([.5],.7),[.35,.3,.35])
    def test_real_group_milp_bounds(self):
        system=self.d["joint_system"]
        atom=next(r["atom"] for r in self.d["joint_atoms"] if r["rows"]==8 and r["forced_public_net_min"]==4)
        self.assertEqual(joint_integer_bound(system,atom),(4,4))
    def test_control_combination_uses_disjoint_rescue(self):
        rows={r["name"]:r for r in self.d["history"]}
        masks=lambda n:set(rows[n]["flips_vs_verified_control"])
        original=masks("submission_03_probe_red_discrepa_10.csv")
        rescue=masks("submission_06_rescata_consenso_8.csv")-original
        best=masks("submission_07_extiende_estable_8.csv")
        self.assertEqual(len(rescue),8);self.assertFalse(rescue&best)
        self.assertEqual(len(rescue|best),29)
    def test_plan_identifies_aggregate_cohorts_and_sentinel(self):
        ids=np.array(self.d["plan_cases"]["ids"]);cohort=np.array(self.d["plan_cases"]["cohort_ids"])
        m=np.zeros((75,len(ids)),int)
        for j,ix in enumerate(self.d["plan_membership_local_indices"]):m[j,ix]=1
        first=[np.flatnonzero(cohort==j)[0] for j in range(32)]
        a=m[:32,first]
        self.assertEqual(np.linalg.matrix_rank(a),32)
        self.assertEqual(np.linalg.matrix_rank(m.astype(float)),75)
        self.assertTrue(m[:15].any(axis=0).all())
        rng=np.random.default_rng(123);z=rng.integers(-1,2,len(ids));d=m@z
        group=np.linalg.solve(a,d[:32])
        truth=np.array([z[cohort==j].sum() for j in range(32)])
        np.testing.assert_allclose(group,truth,atol=1e-8)
        target=np.flatnonzero(ids==self.d["plan_manifest"]["sentinel_case_ID"])[0]
        self.assertAlmostEqual(float(d[32:].sum()-4*group.sum()),float(z[target]))
if __name__=="__main__":unittest.main(verbosity=2)

