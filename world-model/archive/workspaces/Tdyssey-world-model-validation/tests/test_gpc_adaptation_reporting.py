"""Pure numeric summaries: invalid rows/unknown stops cannot disappear."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from tdyssey.gpc_adaptation_reporting import RETENTION_RULES, summarize
from tdyssey.wm_measurement_contract import coverage_measurement


def rows():
    result=[]
    for candidate,value in (("hold",.1),("toward",.3),("side",.2),("away",0.)):
        for seed in (5478,5479,5480):
            for horizon in (1,2,3,4):
                result.append(dict(role="adapted",stratum="independent",shape="L",anchor="IL01",candidate=candidate,
                    seed=seed,horizon=horizon,mode="AR",status="completed",contact_by_h4=True,
                    rng_entry=f"fixture-{seed}-H{horizon}",rng_exit=f"fixture-{seed}-H{horizon}-exit",
                    truth=dict(pose=dict(position=[30,40],angle=.2),pusher=[10,20],coverage=value),
                    latest=dict(pose=dict(position=[20,40],angle=0)),
                    measurement=dict(state=dict(valid=True,pose=dict(position=[30,40],angle=.2)),shape=dict(valid=True),
                        pusher=dict(valid=True,position=[10,20]),coverage=coverage_measurement(value,half_width=.03,
                        uncertainty_kind="development_absolute_error_bound"))))
    return result


def paired_rows():
    result=[]
    for stratum,anchor in (("regression","T12"),("independent","IT01")):
        for role in ("frozen","adapted"):
            for original in rows():
                row=deepcopy(original);row.update(stratum=stratum,anchor=anchor,shape="T",role=role)
                result.append(row)
                if row["horizon"] in (2,4):
                    tf=deepcopy(row);tf["mode"]="TF";result.append(tf)
    return result


def declared_cases(data):
    result={}
    for row in data:
        key=(row["stratum"],row["shape"],row["anchor"],row["candidate"])
        result[key]=dict(role=row["stratum"],shape=row["shape"],anchor=row["anchor"],candidate=row["candidate"],
            contact_by_h4=row["contact_by_h4"],truth=[deepcopy(row["truth"])]*4)
    return list(result.values())


def mixed_rows(values=None):
    """Perfect informative anchors plus complete no-contact physical controls."""
    informative=paired_rows();controls=[]
    values=values or dict.fromkeys(("hold","toward","side","away"),.2)
    for original in informative:
        row=deepcopy(original);row["anchor"]="T00" if row["stratum"]=="regression" else "IT00"
        row["contact_by_h4"]=False;row["truth"]["coverage"]=values[row["candidate"]]
        row["latest"]["pose"]=deepcopy(row["truth"]["pose"])
        row["measurement"]["coverage"]=coverage_measurement(values[row["candidate"]],half_width=.03,
            uncertainty_kind="development_absolute_error_bound")
        controls.append(row)
    return informative+controls


def cell(report,mode="AR",horizon=4,stratum="independent"):
    return next(row for row in report["paired_retention"]["cells"] if row["mode"]==mode and row["horizon"]==horizon and row["stratum"]==stratum)


class AdaptationReportChecks(unittest.TestCase):
    def test_complete_uninformative_controls_do_not_block_supported_ranking(self):
        data=mixed_rows();report=summarize(data,planned_cases=declared_cases(data))
        self.assertEqual(report["paired_retention"]["T"]["overall"]["status"],"pass")
        for ranked in report["paired_retention"]["ranking"]:
            self.assertEqual((ranked["planned_anchors"],ranked["informative_anchors"],ranked["uninformative_anchors"]),(2,1,1))
            self.assertEqual((ranked["common_comparable_anchors"],ranked["planned_informative_pairs"],ranked["common_separable_pairs"]),(1,6,6))
            self.assertEqual((ranked["planned_rows_per_role"],ranked["status"]),(144,"pass"))
            self.assertEqual(ranked["frozen_regret_on_common"]["count"],1)
            control=next(row for row in ranked["anchors"] if row["information"]=="uninformative")
            self.assertEqual(control["status"],"uninformative");self.assertIsNone(control["passed"])
            self.assertTrue(control["matched_noise_complete"])
            self.assertEqual(ranked["uninformative_anchor_ids"],[control["anchor"]])
        with patch.dict(RETENTION_RULES,minimum_rank_comparable_anchors=2):
            insufficient=summarize(data,planned_cases=declared_cases(data))
        self.assertEqual(insufficient["paired_retention"]["T"]["ranking"]["status"],"insufficient")

    def test_informative_missing_or_invalid_anchor_cannot_leave_the_comparison_pool(self):
        original=mixed_rows();plan=declared_cases(original)
        for failure in ("missing_adapted","missing_both","invalid_adapted","invalid_both"):
            with self.subTest(failure=failure):
                data=deepcopy(original)
                if failure.startswith("missing"):
                    data=[row for row in data if not(row["anchor"]=="IT01" and (failure=="missing_both" or row["role"]=="adapted"))]
                else:
                    for row in data:
                        if row["anchor"]=="IT01" and row["candidate"]=="hold" and row["mode"]=="AR" and row["horizon"]==2 and (failure=="invalid_both" or row["role"]=="adapted"):
                            row["measurement"]["state"]["valid"]=False;row["measurement"]["shape"]["valid"]=False
                report=summarize(data,planned_cases=plan)
                ranked=next(row for row in report["paired_retention"]["ranking"] if row["stratum"]=="independent")
                self.assertEqual(ranked["informative_anchor_ids"],["IT01"])
                self.assertEqual((ranked["planned_anchors"],ranked["planned_informative_pairs"],ranked["common_comparable_anchors"]),(2,6,0))
                self.assertNotEqual(ranked["status"],"pass")
                self.assertNotEqual(report["paired_retention"]["T"]["overall"]["status"],"pass")

    def test_uninformative_control_state_regression_or_new_invalid_still_fails_T(self):
        original=mixed_rows();plan=declared_cases(original)
        for failure in ("pose_regression","new_invalid"):
            with self.subTest(failure=failure):
                data=deepcopy(original)
                for row in data:
                    if row["role"]=="adapted" and row["anchor"]=="IT00":
                        if failure=="pose_regression":row["measurement"]["state"]["pose"]["position"][0]+=4
                        else:
                            row["measurement"]["state"]["valid"]=False;row["measurement"]["shape"]["valid"]=False
                report=summarize(data,planned_cases=plan)
                self.assertEqual(report["paired_retention"]["T"]["ranking"]["status"],"pass")
                self.assertEqual(report["paired_retention"]["T"]["AR"]["status"],"fail")
                self.assertEqual(report["paired_retention"]["T"]["overall"]["status"],"fail")

    def test_missing_control_tf_row_still_blocks_complete_grid_condition(self):
        original=mixed_rows();plan=declared_cases(original)
        data=[row for row in original if not(row["role"]=="adapted" and row["anchor"]=="IT00" and row["candidate"]=="hold" and row["seed"]==5478 and row["mode"]=="TF" and row["horizon"]==2)]
        report=summarize(data,planned_cases=plan)
        ranked=next(row for row in report["paired_retention"]["ranking"] if row["stratum"]=="independent")
        self.assertEqual((ranked["common_comparable_anchors"],ranked["common_separable_pairs"]),(1,6))
        self.assertFalse(ranked["grid_complete"]);self.assertEqual(ranked["incomplete_grid_anchors"],["IT00"])
        self.assertEqual(ranked["status"],"insufficient")
        self.assertEqual(report["paired_retention"]["T"]["AR"]["status"],"pass")
        self.assertEqual(report["paired_retention"]["T"]["overall"]["status"],"insufficient")

    def test_information_uses_declared_separation_not_contact_or_predicted_validity(self):
        for values,expected,count in ((dict(hold=.2,toward=.21,side=.22,away=.24),"uninformative",1),
                                      (dict(hold=.2,toward=.2,side=.25,away=.2),"informative",2)):
            with self.subTest(expected=expected):
                data=mixed_rows(values);report=summarize(data,planned_cases=declared_cases(data))
                ranked=next(row for row in report["paired_retention"]["ranking"] if row["stratum"]=="independent")
                control=next(row for row in ranked["anchors"] if row["anchor"]=="IT00")
                self.assertEqual(control["information"],expected)
                self.assertEqual(ranked["informative_anchors"],count)
                self.assertEqual(ranked["status"],"pass")
        data=mixed_rows();plan=declared_cases(data)
        next(case for case in plan if case["anchor"]=="IT01")["truth"]=[]
        report=summarize(data,planned_cases=plan)
        ranked=next(row for row in report["paired_retention"]["ranking"] if row["stratum"]=="independent")
        self.assertEqual(ranked["unknown_truth_anchors"],["IT01"])
        self.assertEqual(ranked["uninformative_anchor_ids"],["IT00"])
        self.assertEqual(ranked["status"],"insufficient")

    def test_low_coverage_mae_cannot_hide_failed_original_p90_gate(self):
        data=rows()
        for row in data:
            if row["candidate"]=="hold":row["measurement"]["coverage"]=coverage_measurement(row["truth"]["coverage"]+.16,half_width=.03)
        report=summarize(data)
        h4=next(row for row in report["cells"] if row["horizon"]==4)
        self.assertAlmostEqual(h4["conditional_errors"]["coverage"]["mean"],.04)
        self.assertAlmostEqual(h4["conditional_errors"]["coverage"]["p90"],.16)
        self.assertFalse(h4["limited_error_screen"])

    def test_absolute_pass_does_not_allow_four_pixel_regression_from_frozen(self):
        data=paired_rows();plan=declared_cases(data)
        for row in data:
            if row["role"]=="adapted" and row["mode"]=="AR":row["measurement"]["state"]["pose"]["position"][0]+=4
        report=summarize(data,planned_cases=plan)
        self.assertTrue(all(row["limited_error_screen"] for row in report["cells"]))
        self.assertEqual(report["paired_retention"]["T"]["AR"]["status"],"fail")
        self.assertEqual(report["paired_retention"]["T"]["TF"]["status"],"pass")
        self.assertEqual(report["paired_retention"]["T"]["overall"]["status"],"fail")
        self.assertEqual(cell(report)["errors_on_same_common_subset"]["position"]["paired_error_increase"]["max"],4)

    def test_new_invalid_hard_case_cannot_improve_retention_by_leaving_error_subset(self):
        data=paired_rows();plan=declared_cases(data)
        for row in data:
            if row["candidate"]=="hold":
                if row["role"]=="frozen":row["measurement"]["state"]["pose"]["position"][0]+=4
                else:
                    row["measurement"]["state"]["valid"]=False;row["measurement"]["shape"]["valid"]=False
        report=summarize(data,planned_cases=plan);paired=cell(report)
        self.assertEqual(paired["status"],"fail")
        self.assertEqual(paired["validity"]["object_shape"]["frozen_only_valid"],3)
        self.assertEqual(paired["validity"]["object_shape"]["delta_fraction"],-.25)
        self.assertEqual(paired["validity"]["object_shape"]["new_invalid_branches"],1)
        error=paired["errors_on_same_common_subset"]["position"]
        self.assertEqual((error["planned"],error["common_valid"]),(12,9))
        self.assertEqual((error["frozen_on_common"]["mean"],error["adapted_on_common"]["mean"]),(0,0))
        self.assertEqual(error["status"],"insufficient")

    def test_both_roles_missing_rows_keep_full_planned_denominator_and_cannot_pass(self):
        data=paired_rows();plan=declared_cases(data)
        data=[row for row in data if not(row["stratum"]=="independent" and row["candidate"]=="hold" and row["mode"]=="AR" and row["horizon"]==4)]
        report=summarize(data,planned_cases=plan);paired=cell(report)
        self.assertEqual((paired["planned"],paired["planned_physical_branches"],paired["planned_anchors"]),(12,4,1))
        self.assertEqual(paired["completion"]["frozen"]["missing"],3)
        self.assertEqual(paired["completion"]["adapted"]["missing"],3)
        self.assertEqual(paired["status"],"insufficient");self.assertIsNone(paired["passed"])
        self.assertEqual(report["paired_retention"]["T"]["absolute_AR"]["status"],"insufficient")
        self.assertNotEqual(report["paired_retention"]["T"]["overall"]["status"],"pass")

    def test_entire_missing_horizon_cannot_make_observed_absolute_cells_pass(self):
        data=paired_rows();plan=declared_cases(data)
        data=[row for row in data if not(row["mode"]=="AR" and row["horizon"]==2)]
        report=summarize(data,planned_cases=plan)
        absolute=report["paired_retention"]["T"]["absolute_AR"]
        self.assertEqual(absolute["status"],"insufficient")
        missing=[row for row in absolute["cells"] if row["horizon"]==2]
        self.assertTrue(missing)
        self.assertTrue(all(row["planned"]==12 and row["observed_adapted"]==0 and row["passed"] is None for row in missing))

    def test_omitted_noise_seed_cannot_shrink_the_predeclared_denominator(self):
        data=paired_rows();plan=declared_cases(data)
        data=[row for row in data if row["seed"]!=5480]
        report=summarize(data,seeds=(5478,5479),planned_cases=plan);paired=cell(report)
        self.assertEqual(paired["planned"],12)
        self.assertEqual(paired["completion"]["adapted"]["missing"],4)
        self.assertFalse(report["paired_retention"]["noise_protocol_matches"])
        self.assertEqual(report["paired_retention"]["T"]["overall"]["status"],"insufficient")

    def test_identity_with_declared_plan_passes_and_no_plan_cannot_certify_retention(self):
        data=paired_rows()
        report=summarize(data,planned_cases=declared_cases(data))
        self.assertEqual(report["paired_retention"]["T"]["overall"]["status"],"pass")
        unplanned=summarize(data)
        self.assertEqual(unplanned["paired_retention"]["T"]["overall"]["status"],"insufficient")
        self.assertFalse(report["paired_retention"]["termination_certified"])

    def test_no_separable_actions_are_insufficient_for_ranking_even_with_zero_regret(self):
        data=paired_rows()
        for row in data:
            row["truth"]["coverage"]=.2;row["measurement"]["coverage"]=coverage_measurement(.2,half_width=.03,uncertainty_kind="development_absolute_error_bound")
        report=summarize(data,planned_cases=declared_cases(data))
        self.assertEqual(report["paired_retention"]["T"]["AR"]["status"],"pass")
        self.assertEqual(report["paired_retention"]["T"]["ranking"]["status"],"insufficient")
        self.assertIsNone(report["paired_retention"]["ranking"][0]["frozen_accuracy_on_common"])
        for ranked in report["paired_retention"]["ranking"]:
            self.assertEqual((ranked["informative_anchors"],ranked["uninformative_anchors"]),(0,1))
            self.assertEqual(ranked["status"],"insufficient")
        self.assertEqual(report["paired_retention"]["T"]["overall"]["status"],"insufficient")

    def test_ranking_compares_same_pairs_and_catches_regret_loss_at_equal_frame_error(self):
        data=paired_rows();values=dict(hold=.1,toward=.15,side=.2,away=0.)
        for row in data:
            value=values[row["candidate"]];row["truth"]["coverage"]=value
            delta=-.04 if row["role"]=="adapted" and row["candidate"]=="side" else .04
            row["measurement"]["coverage"]=coverage_measurement(value+delta,half_width=.03,uncertainty_kind="development_absolute_error_bound")
        report=summarize(data,planned_cases=declared_cases(data))
        self.assertEqual(report["paired_retention"]["T"]["AR"]["status"],"pass")
        ranked=report["paired_retention"]["ranking"][0]
        self.assertEqual((ranked["common_comparable_anchors"],ranked["common_separable_pairs"]),(1,6))
        self.assertEqual(ranked["status"],"fail")
        self.assertAlmostEqual(ranked["frozen_accuracy_on_common"],1.)
        self.assertAlmostEqual(ranked["adapted_accuracy_on_common"],5/6)
        self.assertAlmostEqual(ranked["adapted_regret_on_common"]["mean"],.05)

    def test_numeric_tolerance_is_only_roundoff_not_a_physical_degradation_allowance(self):
        data=paired_rows();plan=declared_cases(data)
        for row in data:
            if row["role"]=="adapted":row["measurement"]["state"]["pose"]["position"][0]+=.5e-6
        self.assertEqual(summarize(data,planned_cases=plan)["paired_retention"]["T"]["AR"]["status"],"pass")
        for row in data:
            if row["role"]=="adapted":row["measurement"]["state"]["pose"]["position"][0]+=2e-6
        self.assertEqual(summarize(data,planned_cases=plan)["paired_retention"]["T"]["AR"]["status"],"fail")

    def test_duplicate_unplanned_noise_and_unmatched_truth_or_noise_cannot_pass(self):
        data=paired_rows();plan=declared_cases(data)
        with self.assertRaises(ValueError):summarize(data+[deepcopy(data[0])],planned_cases=plan)
        wrong=deepcopy(data);wrong[0]["seed"]=9999
        with self.assertRaisesRegex(ValueError,"Unplanned"):summarize(wrong,planned_cases=plan)
        for field in ("truth","rng_entry"):
            wrong=deepcopy(data)
            row=next(r for r in wrong if r["role"]=="adapted" and r["mode"]=="AR" and r["horizon"]==2)
            if field=="truth":row[field]["pose"]["position"][0]+=1
            else:row[field]="different-noise"
            self.assertEqual(cell(summarize(wrong,planned_cases=plan),horizon=2,stratum="regression")["status"],"insufficient")
        wrong=deepcopy(data)
        for row in wrong:
            if row["mode"]=="AR" and row["horizon"]==2:row["truth"]["pose"]["position"][0]+=1
        self.assertEqual(cell(summarize(wrong,planned_cases=plan),horizon=2,stratum="regression")["status"],"insufficient")
        self.assertEqual(summarize([],planned_cases=[])["paired_retention"]["T"]["overall"]["status"],"insufficient")

    def test_tf_is_excluded_and_physical_counts_do_not_treat_noise_as_new_scenes(self):
        data=rows();tf=deepcopy(data[0]);tf.update(mode="TF",horizon=2,candidate="away")
        tf["measurement"]["coverage"]=coverage_measurement(.99,half_width=.03)
        report=summarize(data+[tf]);ranking=report["ranking"][0]
        self.assertEqual((ranking["selected"],ranking["selected_regret"]),("toward",0))
        self.assertEqual((ranking["correct_pairs"],ranking["actual_separable_pairs"]),(6,6))
        cell=next(c for c in report["cells"] if c["mode"]=="AR" and c["horizon"]==4)
        self.assertEqual((cell["total"],cell["physical_branches"],cell["anchors"]),(12,4,1))

    def test_invalid_prediction_retains_denominator_and_abstains_missing_noise_candidate(self):
        data=rows();bad=next(r for r in data if r["candidate"]=="toward" and r["seed"]==5478 and r["horizon"]==3)
        bad["measurement"]["shape"]["valid"]=False
        report=summarize(data);ranking=report["ranking"][0]
        self.assertFalse(ranking["eligible"]);self.assertIsNone(ranking["selected"])
        self.assertEqual((ranking["actual_separable_pairs"],ranking["abstained_pairs"]),(6,3))
        self.assertEqual(ranking["accuracy_including_abstentions"],.5)
        cell=next(c for c in report["cells"] if c["horizon"]==3)
        self.assertEqual((cell["total"],cell["invalid_count"]),(12,1))

    def test_unknown_success_interval_never_becomes_perfect_precision(self):
        data=rows()
        for row in data:
            row["truth"]["coverage"]=.96
            row["measurement"]["coverage"]=coverage_measurement(.95,half_width=.03)
        report=summarize(data);dynamics=report["dynamics"][0]
        self.assertEqual((dynamics["interval_success_tp"],dynamics["interval_success_fp"]),(0,0))
        self.assertIsNone(dynamics["interval_success_precision"])
        self.assertEqual(dynamics["interval_unknown"],12)
        self.assertFalse(dynamics["independent_termination_precision_validated"])
        self.assertFalse(report["ranking"][0]["eligible"])
