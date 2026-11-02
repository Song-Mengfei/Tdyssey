"""Explicit NumPy-only, injected-callback native RGB H1/H4 evaluation.

No model, simulator, renderer or training iterator is constructed here. Four
generated raw RGB frames slide through the same history interface; future truth,
state, pose and masks are deliberately absent from the rollout signature.
"""
from __future__ import annotations
import hashlib
import json


def paired_seed(seed, cohort, batch_index, window_ids, step):
    key = dict(domain="r2c0.native-rollout-batch.v1", seed=int(seed), cohort=str(cohort),
        batch_index=int(batch_index), window_ids=list(window_ids), step=int(step))
    return int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).digest()[:4], "little")


def _sha_array(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def normalize_actions(absolute_actions):
    """Exactly the existing float32 absolute0..511 -> [-1,1] model interface."""
    import numpy as np
    values = np.asarray(absolute_actions)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 511).any():
        raise ValueError("Actual absolute actions must be finite in0..511")
    return values.astype(np.float32)/511*2-1


def rollout_raw(predict_callback, histories, absolute_actions, *, window_ids,
                seed=5478, cohort, batch_size=4):
    """Run four future steps only through the explicitly supplied callback.

    callback(history[N,4,3,96,96], normalized_actions[N,4,2],
             seed=...,cohort=...,step=0..3,window_ids=...)->raw RGB[N,3,96,96].
    Finite out-of-range values are retained and fed back unchanged. Wrong shape,
    nonfinite or callback exception stops the cohort; denominators remain N.
    H1 reuses the first H4 call. No global NumPy RNG is consumed or reseeded.
    """
    import numpy as np
    histories, actions = np.asarray(histories), np.asarray(absolute_actions)
    n = len(histories)
    ids = list(window_ids)
    if (histories.shape != (n,4,3,96,96) or len(ids) != n or len(set(ids)) != n
            or not np.issubdtype(histories.dtype, np.floating) or not np.isfinite(histories).all()
            or (histories < 0).any() or (histories > 1).any() or actions.shape != (n,7,2)
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("Invalid saved real history/action audit prefix or window identities")
    normalized = normalize_actions(actions)
    # Float64 containers preserve every float32/float64 callback value exactly;
    # returned dtype is also recorded. Missing endpoints remain explicit NaN.
    h1, h4 = np.full((n,3,96,96), np.nan), np.full((n,3,96,96), np.nan)
    s1, s4 = ["not_attempted"]*n, ["not_attempted"]*n
    records, failures = [], []
    hard_stop = False
    for batch_index, start in enumerate(range(0,n,batch_size)):
        end = min(n,start+batch_size)
        history = histories[start:end].copy()
        for step in range(4):
            action4 = normalized[start:end,step:step+4].copy()
            callback_seed = paired_seed(seed,cohort,batch_index,ids[start:end],step)
            rec = dict(batch_index=batch_index, step=step, window_ids=ids[start:end], seed=callback_seed,
                input_history_sha256=_sha_array(history), input_history_dtype=str(history.dtype),
                normalized_actions_sha256=_sha_array(action4), action_prefix_indices=list(range(step,step+4)))
            try:
                raw = predict_callback(history.copy(), action4, seed=callback_seed, cohort=cohort,
                    step=step, window_ids=ids[start:end])
                raw = np.asarray(raw)
            except Exception as exc:
                rec.update(status="callback_exception", error=f"{type(exc).__name__}: {exc}")
                records.append(rec)
                failures.append(dict(**rec, raw=None))
                s4[start:end] = ["callback_exception"]*(end-start)
                if step == 0:s1[start:end] = ["callback_exception"]*(end-start)
                hard_stop = True
                break
            rec.update(raw_shape=list(raw.shape), raw_dtype=str(raw.dtype), raw_sha256=_sha_array(raw))
            numeric = np.issubdtype(raw.dtype,np.floating)
            if raw.shape != (end-start,3,96,96) or not numeric:
                rec["status"] = "wrong_shape_or_dtype"
                records.append(rec)
                failures.append(dict(**rec,raw=raw.copy()))
                s4[start:end] = ["wrong_shape_or_dtype"]*(end-start)
                if step == 0:s1[start:end] = ["wrong_shape_or_dtype"]*(end-start)
                hard_stop = True
                break
            finite = np.isfinite(raw).all(axis=(1,2,3))
            statuses = ["nonfinite" if not finite[i] else "finite_out_of_range" if raw[i].min()<0 or raw[i].max()>1 else "ok" for i in range(len(raw))]
            rec.update(statuses=statuses, status="nonfinite" if not finite.all() else "finite",
                finite_min=None if not np.isfinite(raw).any() else float(raw[np.isfinite(raw)].min()),
                finite_max=None if not np.isfinite(raw).any() else float(raw[np.isfinite(raw)].max()))
            records.append(rec)
            if step == 0:
                h1[start:end] = raw
                s1[start:end] = statuses
            if step == 3:
                h4[start:end] = raw
                s4[start:end] = statuses
            if not finite.all():
                failures.append(dict(**rec,raw=raw.copy()))
                hard_stop = True
                if step != 3:s4[start:end] = ["hard_stop_nonfinite_before_H4"]*(end-start)
                break
            # No clipping, recoloring, repair or future real RGB in feedback.
            history = np.concatenate((history[:,1:],raw[:,None]),axis=1)
        if hard_stop:
            break
    return dict(window_ids=ids, h1=h1, h4=h4, status_h1=s1, status_h4=s4,
        step_records=records, raw_failures=failures, hard_stop=hard_stop,
        planned_windows=n, callback_calls=len(records), h1_reuses_first_h4_call=True,
        finite_out_of_range_feedback="raw unchanged", future_truth_in_feedback=False,
        action_normalization="float32/511*2-1 exactly once inside this interface")


def score_endpoints(result, truth_h1, truth_h4, last, metadata, calibration):
    """Deferred CPU scoring, with unchanged raw-visible object/pusher metrics."""
    from scripts.native_rgb_stage3_score import score_image
    n = result["planned_windows"]
    if not all(len(v)==n for v in (truth_h1,truth_h4,last,metadata)):
        raise ValueError("Truth/last/metadata denominators differ from frozen cohort")
    rows = []
    for horizon, truth, images, statuses in ((1,truth_h1,result["h1"],result["status_h1"]),(4,truth_h4,result["h4"],result["status_h4"])):
        for i, row in enumerate(metadata):
            # Existing score treats finite out-of-range as hard failure while
            # rollout retains its original array and still uses finite feedback.
            status = "ok" if statuses[i] in ("ok","finite_out_of_range") else statuses[i]
            score = score_image(images[i],truth[i],last[i],calibration,letter=row["letter"],status=status)
            ambiguous = score.get("occlusion_ambiguous") or score.get("prediction_object_border_touch") or score.get("truth_object_border_touch")
            if ambiguous:
                score.update(legacy_shape_assessable=False,legacy_shape_deformation_failure=None,
                    legacy_shape_unknown_reason="visible_occlusion_or_viewport_ambiguity")
            score["shape_outcome"] = ("unknown" if not score.get("legacy_shape_assessable") else
                "failure" if score.get("legacy_shape_deformation_failure") else "assessable_no_detected_failure")
            rows.append(dict(window_id=result["window_ids"][i],horizon=horizon,native_numeric_status=statuses[i],**score))
    return rows
