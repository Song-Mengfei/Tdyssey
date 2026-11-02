"""Fixed development36 H1/H4 raw-feedback orchestration; model-free factories.

Only an explicitly supplied predictor callback can execute sampling. Truth,
geometry, masks and labels are confined to preparation/scoring, never feedback.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD_EVALUATION_SHA="dbd7cd9be081a579708f4760ecf917a1fee0b2b2e2c5794eebdbc3c5d6729c10"
OLD_SCORE_SHA="9a165c71befa83d190d29311c265e2795ae24205464310d26fd0e64100ff49d6"
OLD_PROTOCOL_SHA="353ba2e644842a7006d3b4b2e4fe913167a41e08a9f16e7cfbb92d767ae64428"
OLD_CALIBRATION_FILE_SHA="fae795cd325ca4b88aa23faa127f9e84036f06bec485b169d12b902887c45609"
OLD_CALIBRATION_INTERNAL_SHA="cd2ed20812e8e53f138579939c45934b19329ef1afd111ec777b20ec0588e20c"
LETTERS=tuple("ABCEFHIJLMOPSTUVXY")

GALLERY_TEMPLATE=dict(main_columns=["输入图像","真实下一帧","当前微调模型预测"],
    H4_main_columns=["输入图像","真实第4帧","当前微调模型预测"],
    baseline_columns=["输入图像","真实下一帧","原始GPC（未微调，错误颜色基线）"],
    baseline_page_title="原始GPC（未微调，错误颜色基线）",
    baseline_on_separate_pages=True,order="冻结DEV36原顺序，各horizon全36，不按结果挑图",rows_per_page=6,
    horizon_labels={1:"一步预测 H1（0.1秒）",4:"四步自回归 H4（0.4秒）"},
    invalid_label="无效/未生成：原始数组另存，未修补",prediction_repair=False)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_validation_plan(dataset,development_windows,paired_batches,*,real_histories):
    """Read only saved metadata/actions and cached real RGB; no render call."""
    import numpy as np
    windows=list(development_windows)
    if len(windows)!=36:raise ValueError("Exactly the frozen36 development windows are required")
    histories=np.asarray(real_histories)
    if histories.shape!=(36,4,3,96,96) or not np.isfinite(histories).all() or histories.min()<0 or histories.max()>1:
        raise ValueError("Four real observed RGB frames must be finite in[0,1]")
    rows=[];sequences=[]
    for selected in windows:
        index=int(selected["dataset_index"]);trajectory_index,anchor=dataset.windows[index]
        entry,arrays=dataset.trajectory(trajectory_index)
        tid=entry["trajectory_id"];identifier=f"{tid}:{anchor}"
        if (entry.get("split")!="seen_val" or selected.get("split")!="seen_val" or entry["shape"] not in LETTERS
                or selected["letter"]!=entry["shape"] or identifier!=selected["window_id"] or anchor!=selected["anchor"]
                or selected["npz_sha256"]!=entry["sha256"] or selected["metadata_sha256"]!=entry["metadata_sha256"]):
            raise ValueError("Development index/split/anchor identity differs")
        valid=np.asarray(arrays["valid_transition"],dtype=bool)
        if anchor<3 or anchor+4>len(valid) or not valid[anchor-3:anchor+4].all():
            raise ValueError("All seven history/future transitions must be valid")
        sequence=np.asarray(arrays["actions"])[anchor-3:anchor+4].copy()
        if sequence.shape!=(7,2) or not np.isfinite(sequence).all() or sequence.min()<0 or sequence.max()>511:
            raise ValueError("Seven raw absolute actions must align with this anchor")
        sequences.append(sequence)
        rows.append(dict(window_id=identifier,letter=entry["shape"],trajectory_index=trajectory_index,dataset_index=index,
            anchor=anchor,history_state_indices=list(range(anchor-3,anchor+1)),action_indices=list(range(anchor-3,anchor+4)),
            h1_target_state_index=anchor+1,h4_target_state_index=anchor+4,
            h4_target_state=np.asarray(arrays["states"])[anchor+4].copy(),
            npz_sha256=entry["sha256"],metadata_sha256=entry["metadata_sha256"],split="seen_val"))
    batches=paired_batches["cohorts"]["dev36"] if isinstance(paired_batches,dict) else paired_batches
    _validate_batches(batches,[row["window_id"] for row in rows])
    return dict(windows=copy.deepcopy(windows),window_metadata=rows,window_ids=[row["window_id"] for row in rows],
        real_histories=histories.copy(),absolute_actions=np.stack(sequences),batches=copy.deepcopy(batches),
        preparation_render_calls=0,preparation_physics_controls=0,future_truth_in_feedback=False,
        scoring_targets_are_deferred=True,gallery_template=copy.deepcopy(GALLERY_TEMPLATE))


def _validate_batches(batches,window_ids):
    if len(window_ids)!=36 or len(set(window_ids))!=36 or len(batches)!=9:raise ValueError("Frozen36 windows require9 full batches")
    flattened=[]
    for index,batch in enumerate(batches):
        positions=list(batch["positions"])
        if positions!=list(range(index*4,index*4+4)) or batch["batch_size"]!=4 or batch["batch_index"]!=index:
            raise ValueError("Original batch4 order/membership must remain fixed")
        if batch["window_ids"]!=[window_ids[i] for i in positions]:raise ValueError("Batchwindow IDs differ")
        if type(batch["batch_seed"]) is not int or not 0<=batch["batch_seed"]<2**32:raise ValueError("Original batchseed must be uint32")
        flattened.extend(positions)
    if flattened!=list(range(36)):raise ValueError("Batch coverage differs")


def _numeric_status(image):
    import numpy as np
    if image.shape!=(3,96,96):raise ValueError("wrong_prediction_shape")
    if not np.isfinite(image).all():raise ValueError("nonfinite_prediction")
    return "out_of_range_native_RGB" if image.min()<0 or image.max()>1 else "ok"


def load_h1_cache(path,*,expected_sha256,window_ids,batches):
    """Hash-bound published H1 reference; does not call a sampler or renderer."""
    import numpy as np
    if _sha(path)!=expected_sha256:raise ValueError("Published H1 cache hash differs")
    with np.load(path,allow_pickle=False) as source:
        result={key:source[key].copy() for key in ("prediction","status","window_ids")}
    if list(result["window_ids"])!=list(window_ids) or result["prediction"].shape!=(36,3,96,96):
        raise ValueError("Published H1 cache order/shape differs")
    _validate_batches(batches,list(window_ids))
    result.update(batches=copy.deepcopy(batches),source_path=str(Path(path).resolve()),source_sha256=expected_sha256)
    return result


def autoregressive_batches(predict_batch,real_histories,absolute_actions,batches,*,window_ids,cached_h1=None,before_call=None,after_call=None):
    """Only real histories/actions enter; no future-truth argument exists.

    before_call(context) runs before sampling. after_call(context,raw_batch)
    receives unmodified output before validity checks, allowing failure retention.
    Any real exception stops the whole evaluation; finite OOR is fed back as-is.
    """
    import numpy as np
    histories=np.asarray(real_histories);actions=np.asarray(absolute_actions)
    if histories.shape!=(36,4,3,96,96) or not np.isfinite(histories).all() or histories.min()<0 or histories.max()>1:
        raise ValueError("Initial real observations must be[0,1] andfinite")
    if actions.shape!=(36,7,2) or not np.isfinite(actions).all() or actions.min()<0 or actions.max()>511:
        raise ValueError("Expected36 aligned seven-action sequences")
    _validate_batches(batches,list(window_ids))
    predictions=np.full((36,4,3,96,96),np.nan,dtype=np.result_type(histories.dtype,np.float32))
    status=np.full((36,4),"not_generated",dtype="U512");calls=[];raw_failures=[];fatal=None
    if cached_h1 is not None:
        cached=np.asarray(cached_h1["prediction"])
        if cached.shape!=(36,3,96,96) or list(cached_h1["window_ids"])!=list(window_ids):
            raise ValueError("CachedH1 shape/order differs from frozen inputs")
        if not np.issubdtype(cached.dtype,np.floating):raise ValueError("CachednativeH1 must retain rawfloating RGB")
        if "batches" not in cached_h1 or cached_h1["batches"]!=batches:raise ValueError("CachedH1 seed/membership identity missing or different")
        predictions=predictions.astype(np.result_type(predictions.dtype,cached.dtype),copy=False)
        predictions[:,0]=cached
        try:
            for i,raw in enumerate(cached):
                measured=_numeric_status(raw);status[i,0]=measured
                if str(cached_h1["status"][i])!=measured:raise ValueError("CachedH1 status does not describe raw values")
        except Exception as error:
            fatal=dict(stage="cached_h1",error=f"{type(error).__name__}: {error}",whole_work_stop=True)
    for batch in batches:
        if fatal:break
        positions=batch["positions"];history=histories[positions].copy()
        for k in range(1,5):
            context=dict(batch_index=batch["batch_index"],step=k,positions=list(positions),window_ids=list(batch["window_ids"]),
                         batch_seed=(batch["batch_seed"]+k-1)%2**32,allow_generated_history=k>1,cached=k==1 and cached_h1 is not None)
            raw=None
            try:
                if context["cached"]:raw=np.asarray(cached_h1["prediction"])[positions].copy()
                else:
                    if before_call is not None:before_call(copy.deepcopy(context))
                    calls.append(dict(copy.deepcopy(context),outcome="attempted"))
                    raw=np.asarray(predict_batch(history.copy(),actions[positions,k-1:k+3].copy(),
                        batch_seed=context["batch_seed"],allow_generated_history=k>1))
                    calls[-1]["outcome"]="raw_returned"
                    if after_call is not None:after_call(copy.deepcopy(context),raw.copy())
                if raw.shape!=(4,3,96,96):raise ValueError("wrong_prediction_batch_shape")
                if not np.issubdtype(raw.dtype,np.floating):raise ValueError("Native RGB prediction must be rawfloating point")
                if raw.dtype!=predictions.dtype:
                    promoted=np.result_type(predictions.dtype,raw.dtype)
                    if promoted!=predictions.dtype:predictions=predictions.astype(promoted)
                predictions[positions,k-1]=raw
                numeric_errors=[]
                for i,image in zip(positions,raw):
                    try:status[i,k-1]=_numeric_status(image)
                    except Exception as numeric_error:
                        status[i,k-1]="nonfinite_native_RGB";numeric_errors.append(str(numeric_error))
                if numeric_errors:raise ValueError("nonfinite_prediction")
                history=np.concatenate((history[:,1:],raw[:,None]),axis=1)
            except Exception as error:
                if raw is not None:raw_failures.append(dict(context=context,raw=raw.copy()))
                fatal=dict(stage="rollout",batch_index=batch["batch_index"],step=k,error=f"{type(error).__name__}: {error}",whole_work_stop=True)
                for i in positions:
                    if status[i,k-1]=="not_generated":status[i,k-1]="fatal_"+type(error).__name__
                break
    complete=fatal is None and not np.any(status=="not_generated")
    return dict(prediction=predictions,status=status,window_ids=list(window_ids),batches=copy.deepcopy(batches),
        call_log=calls,predictor_calls=len(calls),cached_h1_reused=cached_h1 is not None,
        raw_failures=raw_failures,fatal_error=fatal,evaluation_complete=bool(complete),future_truth_in_feedback=False,
        no_clipping_or_affine_mapping=True)


def verify_scoring_migration(snapshot_path=None):
    """Explicit source migration: only generated-history interfaces may differ."""
    from . import native_rgb_evaluation as current
    snapshot=Path(snapshot_path or ROOT/"artifacts/native-rgb-supervised-20261029/stage3c/source-snapshot/tdyssey/native_rgb_evaluation.py")
    if _sha(snapshot)!=OLD_EVALUATION_SHA:raise ValueError("Original metric-source snapshot identity differs")
    old=ast.parse(snapshot.read_text(encoding="utf-8"));new=ast.parse(Path(current.__file__).read_text(encoding="utf-8"))
    allowed={"native_sampler_batch_predictor","validate_native_batch_inputs","NativePredictor"}
    def stable(tree):return [ast.dump(n,include_attributes=False) for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef)) or n.name not in allowed]
    if stable(old)!=stable(new):raise ValueError("Scoring math/constants changed outside the authorized feedback interface")
    old_cls=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=="NativePredictor")
    new_cls=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=="NativePredictor")
    if [ast.dump(n,include_attributes=False) for n in old_cls.body if not isinstance(n,ast.FunctionDef) or n.name!="batch_predict"] != [ast.dump(n,include_attributes=False) for n in new_cls.body if not isinstance(n,ast.FunctionDef) or n.name!="batch_predict"]:
        raise ValueError("Native model construction orother interfaces changed")
    def sampler_parts(tree):
        outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="native_sampler_batch_predictor")
        inner=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=="predict")
        return outer,inner
    a,b=sampler_parts(old);c,d=sampler_parts(new)
    if ([ast.dump(n,include_attributes=False) for n in a.body if n is not b] != [ast.dump(n,include_attributes=False) for n in c.body if n is not d]
            or [ast.dump(n,include_attributes=False) for n in b.body[4:]] != [ast.dump(n,include_attributes=False) for n in d.body[3:]]):
        raise ValueError("Sampler math/normalization/RNG changed")
    return dict(original_evaluation_sha256=OLD_EVALUATION_SHA,active_evaluation_sha256=_sha(current.__file__),
        scoring_math_AST_identical=True,sampler_math_AST_identical=True,
        authorized_changes=["pure batch input provenance validator","allow_generated_history keywordforwarding"],
        historical_protocol_or_snapshot_modified=False)


def make_original_scorer(*,snapshot_path=None,protocol_path=None,calibration_path=None):
    """Old numerical score with an explicit, verified active-source migration."""
    from . import native_rgb_evaluation as current
    from scripts.native_rgb_stage3_score import score_image
    protocol=Path(protocol_path or ROOT/"configs/native_rgb_stage3_score.json")
    calibration_path=Path(calibration_path or ROOT/"artifacts/native-rgb-20261028/frozen-renderer-calibration.json")
    if _sha(protocol)!=OLD_PROTOCOL_SHA or _sha(ROOT/"scripts/native_rgb_stage3_score.py")!=OLD_SCORE_SHA or _sha(calibration_path)!=OLD_CALIBRATION_FILE_SHA:
        raise ValueError("Original protocol/scorer/calibration bytes changed")
    migration=verify_scoring_migration(snapshot_path)
    calibration=current.validate_calibration(calibration_path)
    if calibration["sha256"]!=OLD_CALIBRATION_INTERNAL_SHA:raise ValueError("Original calibration hash differs")
    def score(prediction,truth,last,*,letter,status,truth_full_mask=None,last_full_mask=None):
        return score_image(prediction,truth,last,calibration,letter=letter,status=status,
                           truth_full_mask=truth_full_mask,last_full_mask=last_full_mask)
    return score,dict(protocol_sha256=OLD_PROTOCOL_SHA,scorer_sha256=OLD_SCORE_SHA,calibration_sha256=OLD_CALIBRATION_INTERNAL_SHA,source_migration=migration)


def native_predictor_factory(context):
    """Concrete future-model bridge; called only after the executionguard."""
    from .native_rgb_evaluation import native_sampler_batch_predictor
    from .gpc_world_model import world_model_spec
    expected=world_model_spec()["sampler"]
    if context["config"].get("sampler",expected)!=expected:raise ValueError("Original sampler configuration must remain unchanged")
    return native_sampler_batch_predictor(context["model"],context["source"],sampler_config=expected)


def make_validation_callback(plan,*,predictor_factory=native_predictor_factory,truth_provider,row_scorer,execution_guard,raw_writer,
                             formal_experiment_id,selection_catalog=None,assess_candidate=None,
                             before_call=None,after_call=None,cached_h1=None,reference_only=False):
    """Future execution callback. The caller owns authorization/budget/model mode.

    truth_provider(horizon,positions,window_ids,context=...,rollout_token=...) is
    invoked only after raw rollout and returns rgb/window_ids/horizon and optional
    full_object_mask/last_full_mask for offline scoring. raw_writer
    persists everyraw/partial output and returns only small artifact references.
    """
    if not callable(execution_guard) or not callable(raw_writer):raise ValueError("Explicit executionguard andraw persistence callback required")
    def callback(context):
        execution_guard(context)
        update=context["progress"]["updates"]
        role=context.get("validation_role")
        if (type(update) is not int or (not reference_only and (role not in ("scheduled","early_stop")
                or role=="scheduled" and update not in (250,500) or role=="early_stop" and not 1<=update<500))):
            raise ValueError("Only formal250/500 orone positive early-stop final may be evaluated")
        predictor=predictor_factory(context)
        def check_call(info):
            context["check"]()
            if before_call is not None:before_call(info)
        result=autoregressive_batches(predictor,plan["real_histories"],plan["absolute_actions"],plan["batches"],
            window_ids=plan["window_ids"],cached_h1=cached_h1,before_call=check_call,after_call=after_call)
        artifacts=raw_writer(result,context)
        token=dict(authorized=True,rollout_finished=True,evaluation_complete=result["evaluation_complete"],
            fatal_error=result["fatal_error"],formal_experiment_id=formal_experiment_id,update=update,window_ids=list(plan["window_ids"]))
        records={}
        for horizon in (1,4):
            truth=truth_provider(horizon,list(range(36)),list(plan["window_ids"]),context=context,rollout_token=copy.deepcopy(token))
            if truth.get("window_ids")!=list(plan["window_ids"]) or truth.get("horizon")!=horizon:
                raise ValueError("Offline truth-provider horizon/order identity differs")
            rows=[]
            for i,window in enumerate(plan["windows"]):
                full=truth.get("full_object_mask");prior=truth.get("last_full_mask")
                measured=row_scorer(result["prediction"][i,horizon-1],truth["rgb"][i],plan["real_histories"][i,-1],
                    letter=window["letter"],status=str(result["status"][i,horizon-1]),
                    truth_full_mask=None if full is None else full[i],last_full_mask=None if prior is None else prior[i])
                hold=row_scorer(plan["real_histories"][i,-1],truth["rgb"][i],plan["real_histories"][i,-1],
                    letter=window["letter"],status="ok",truth_full_mask=None if full is None else full[i],
                    last_full_mask=None if prior is None else prior[i])
                rows.append(dict(measured,window_id=window["window_id"],horizon=horizon,hold_metrics=hold))
            records["h"+str(horizon)]=rows
        candidate=None;assessment=None
        if not reference_only:
            candidate=dict(formal_experiment_id=formal_experiment_id,candidate_role=role,
                update=update,evaluation_complete=result["evaluation_complete"],**records)
            if assess_candidate is not None:assessment=assess_candidate(candidate,selection_catalog,formal_experiment_id=formal_experiment_id)
        return dict(formal_experiment_id=formal_experiment_id,formal_update=update,
            candidate_role="published_reference" if reference_only else role,evaluation_complete=result["evaluation_complete"],
            candidate=candidate,selection=assessment,**records,raw_artifacts=artifacts,
            fatal_error=result["fatal_error"],call_log=result["call_log"],cached_h1_reused=result["cached_h1_reused"],
            stop_model_work=result["fatal_error"] is not None,
            published_reference_only=bool(reference_only),future_truth_in_feedback=False,gallery_template=copy.deepcopy(GALLERY_TEMPLATE))
    return callback


def make_baseline_h4_callback(plan,*,cached_h1,**kwargs):
    if cached_h1 is None:raise ValueError("Published H1 cache is required; baselineH1 must not be regenerated")
    return make_validation_callback(plan,cached_h1=cached_h1,reference_only=True,**kwargs)
