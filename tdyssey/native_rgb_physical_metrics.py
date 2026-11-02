"""Offline occlusion-aware pose measurement of *unchanged* native RGB outputs.

This is an evaluation instrument, not a predictor. No fitted mask, translation,
angle, geometry, or simulator label is returned to the sampler. Calibration can
fail; unavailable rows remain in the physical qualification denominator.
"""

from collections import Counter, OrderedDict, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from .native_rgb_evaluation import (
    TRAIN_LETTERS, WORLD_SCALE, _rgb, _stable_seed, palette_masks, _json_value,
)


PROTOCOL = dict(version="offline-occlusion-aware-pose.v1", image_size=96,
    coarse_angle_degrees=10, refined_angle_degrees=1,
    centroid_iterations=4, subpixel_grid=[-.5, 0., .5],
    minimum_visible_fraction=.55, minimum_fit_iou=.75,
    maximum_missing_fraction=.20, maximum_extra_fraction=.20,
    blue_exclusion_dilation_pixels=1,
    maximum_calibration_translation_error_px=6.,
    maximum_calibration_angle_error_deg=6.,
    required_motion_improvement=.20,
    required_gate_horizons=[1,4],
    require_absolute_gain_above_twice_calibration_pose_error=True,
    fitted_images_enter_prediction=False)


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True,
                               allow_nan=False)+"\n", encoding="utf-8")


class PredictionTrace:
    """Pass through the original result object, keeping bounded raw CPU copies.

    max_bytes bounds retained float32 RGB bytes, independently for each model.
    Metadata has a separate finite record limit. An overflow changes measurement
    availability only. It never changes a sampler argument, result, or exception.
    """

    def __init__(self, predictor, max_bytes=128*1024**2, max_records=4096):
        if type(max_bytes) is not int or max_bytes < 0 or type(max_records) is not int or max_records < 1:
            raise ValueError("Trace requires nonnegative bytes and a finite positive record limit")
        self.predictor, self.max_bytes, self.max_records = predictor, max_bytes, max_records
        self.records, self._raw = [], {}
        self.retained_bytes = self.dropped_raw_outputs = self.metadata_overflows = self.total_calls = 0
        self.instrumentation_errors = 0

    def _safe_record(self,*args,**kwargs):
        try:
            self._record(*args,**kwargs)
        except Exception:
            # An instrument failure must not change a prediction or replace the
            # predictor's original exception. Its missing row is scored unavailable.
            self.instrumentation_errors += 1

    def _record(self, mode, seed, history, actions, result=None, error=None):
        import numpy as np
        call_id = self.total_calls
        self.total_calls += 1
        if len(self.records) >= self.max_records:
            self.metadata_overflows += 1
            return
        expected = 1 if mode == "single" else len(history)
        histories = np.asarray(history)[None] if mode == "single" else np.asarray(history)
        controls = np.asarray(actions)[None] if mode == "single" else np.asarray(actions)
        row = dict(id=call_id, mode=mode, seed=int(seed), count=expected,
            history_sha256=[_hash_array(value) for value in histories],
            actions_sha256=[_hash_array(np.asarray(value, dtype=np.float32)) for value in controls],
            status="exception" if error is not None else "returned", error=error,
            raw_cached=False, item_valid=[False]*expected, item_sha256=[None]*expected)
        if error is None:
            values = np.asarray(result)
            if mode == "single":
                values = values[None]
            row["returned_shape"] = list(np.asarray(result).shape)
            if values.shape == (expected,3,96,96):
                # Conversion affects the trace copy only; NativePredictor already
                # returns float32. The caller receives its exact original object.
                values = np.asarray(values, dtype=np.float32)
                row["item_valid"] = [bool(np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1)
                                     for value in values]
                row["item_sha256"] = [_hash_array(value) for value in values]
                if self.retained_bytes+values.nbytes <= self.max_bytes:
                    saved = values.copy()
                    saved.flags.writeable = False
                    self._raw[call_id] = saved
                    self.retained_bytes += saved.nbytes
                    row["raw_cached"] = True
                else:
                    self.dropped_raw_outputs += expected
            else:
                row["shape_invalid"] = True
        self.records.append(row)

    def __call__(self, history, absolute_actions, seed=5478):
        try:
            result = self.predictor(history, absolute_actions, seed)
        except Exception as exc:
            self._safe_record("single",seed,history,absolute_actions,error=type(exc).__name__+": "+str(exc)[:300])
            raise
        self._safe_record("single",seed,history,absolute_actions,result)
        return result

    def forward(self, history, absolute_actions, seed=5478):
        function=getattr(self.predictor,"forward",self.predictor)
        try:
            result=function(history,absolute_actions,seed)
        except Exception as exc:
            self._safe_record("single",seed,history,absolute_actions,error=type(exc).__name__+": "+str(exc)[:300])
            raise
        self._safe_record("single",seed,history,absolute_actions,result)
        return result

    def batch_predict(self, histories, absolute_actions, *, batch_seed=5478):
        try:
            result = self.predictor.batch_predict(histories,absolute_actions,batch_seed=batch_seed)
        except Exception as exc:
            self._safe_record("batch",batch_seed,histories,absolute_actions,error=type(exc).__name__+": "+str(exc)[:300])
            raise
        self._safe_record("batch",batch_seed,histories,absolute_actions,result)
        return result

    def raw(self, call_id, item=0):
        value = self._raw.get(call_id)
        return None if value is None or not 0 <= item < len(value) else value[item]

    def snapshot(self):
        return dict(max_raw_bytes=self.max_bytes, retained_raw_bytes=self.retained_bytes,
            max_records=self.max_records, retained_records=len(self.records), total_calls=self.total_calls,
            dropped_raw_outputs=self.dropped_raw_outputs, metadata_overflows=self.metadata_overflows,
            instrumentation_errors=self.instrumentation_errors,
            outputs_modified=False, raw_images_written_to_disk=False)

    def clear(self):
        self._raw.clear()
        self.records.clear()
        self.retained_bytes = 0

    def close(self):
        # Keep copies until the caller has completed offline scoring/clear().
        self.predictor.close()


def _centroid(mask):
    import numpy as np
    y,x = np.nonzero(mask)
    return None if not len(x) else np.asarray([x.mean(),y.mean()],dtype=np.float64)


def _edge(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())


def equivalent_angle_degrees(angle, symmetry_order):
    if symmetry_order not in (1,2,4):
        raise ValueError("Unsupported declared geometry symmetry")
    period = 360/symmetry_order
    return (float(angle)+period/2) % period-period/2


def _transform(mask, angle, shift, center):
    import cv2
    import numpy as np
    radians = math.radians(angle)
    matrix = np.asarray([[math.cos(radians),-math.sin(radians)],
                         [math.sin(radians),math.cos(radians)]])
    offset = center+np.asarray(shift)-matrix@center
    return cv2.warpAffine(mask.astype(np.float32),np.column_stack((matrix,offset)),
        (96,96),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def fit_physical_pose(prediction, truth_full_mask, palette_calibration, *, symmetry_order=1, protocol=None):
    """Fit the complete truth shape to raw visible grey, excluding predicted blue.

    The returned translation is the fitted full-area-centroid displacement from
    the true future full-area centroid, in the 512-coordinate world frame.
    No fitted mask or RGB is exposed to a prediction caller.
    """
    import numpy as np
    from scipy import ndimage
    protocol = PROTOCOL if protocol is None else protocol
    grey,blue = palette_masks(_rgb(prediction),palette_calibration)
    truth = np.asarray(truth_full_mask,dtype=np.float32)
    alpha = palette_calibration["palette"]["minimum_alpha"]
    if truth.shape != (96,96) or not np.isfinite(truth).all():
        return dict(status="unmeasurable",reason="invalid_full_truth_mask")
    full = truth >= alpha
    if not grey.any() or not full.any():
        return dict(status="unmeasurable",reason="empty_grey_or_full_truth_mask")
    if _edge(full) or _edge(grey):
        return dict(status="unmeasurable",reason="viewport_clipping")
    excluded = ndimage.binary_dilation(blue,iterations=protocol["blue_exclusion_dilation_pixels"])
    observed = grey & ~excluded
    center,seen_center = _centroid(full),_centroid(observed)
    if seen_center is None:
        return dict(status="unmeasurable",reason="all_observed_grey_excluded")
    initial = seen_center-center
    candidates = []
    def candidate(angle, shift=None, iterate=True):
        angle = equivalent_angle_degrees(angle,symmetry_order)
        shift = initial.copy() if shift is None else np.asarray(shift,dtype=np.float64).copy()
        for _ in range(protocol["centroid_iterations"] if iterate else 0):
            warped = _transform(truth,angle,shift,center) >= alpha
            visible = warped & ~excluded
            current = _centroid(visible)
            if current is None:
                break
            adjustment = seen_center-current
            shift += adjustment
            if float(np.linalg.norm(adjustment)) < .01:
                break
        warped = _transform(truth,angle,shift,center) >= alpha
        visible = warped & ~excluded
        union = (visible|observed).sum()
        iou = float((visible&observed).sum()/union) if union else 0.
        visible_fraction = float(visible.sum()/max(1,warped.sum()))
        missing = float((visible&~observed).sum()/max(1,visible.sum()))
        extra = float((observed&~visible).sum()/max(1,visible.sum()))
        candidates.append(dict(fit_iou=iou,angle=angle,shift=shift,
            visible_fraction=visible_fraction,missing_fraction=missing,extra_fraction=extra,
            clipped=_edge(warped),full_area=int(warped.sum())))
    period = int(360/symmetry_order)
    for angle in range(-period//2,period//2,protocol["coarse_angle_degrees"]):
        candidate(angle)
    best = max(candidates,key=lambda row:(row["fit_iou"],-abs(row["angle"])))
    coarse = best["angle"]
    for angle in range(int(coarse)-10,int(coarse)+11,protocol["refined_angle_degrees"]):
        candidate(angle)
    best = max(candidates,key=lambda row:(row["fit_iou"],-abs(row["angle"])))
    for angle in (best["angle"]-1,best["angle"],best["angle"]+1):
        for dx in protocol["subpixel_grid"]:
            for dy in protocol["subpixel_grid"]:
                candidate(angle,best["shift"]+(dx,dy),iterate=False)
    best = max(candidates,key=lambda row:(row["fit_iou"],-abs(row["angle"])))
    reasons = []
    if best["clipped"]:reasons.append("fitted_shape_clipped")
    if best["visible_fraction"] < protocol["minimum_visible_fraction"]:reasons.append("insufficient_visible_shape")
    if best["fit_iou"] < protocol["minimum_fit_iou"]:reasons.append("shape_fit_below_fixed_floor")
    if best["missing_fraction"] > protocol["maximum_missing_fraction"]:reasons.append("missing_shape")
    if best["extra_fraction"] > protocol["maximum_extra_fraction"]:reasons.append("extra_shape")
    translation = best["shift"]*WORLD_SCALE
    radius = float(np.sqrt(((np.argwhere(full)-center[::-1])**2).sum(1).mean())*WORLD_SCALE)
    return dict(status="unmeasurable" if reasons else "ok",reason=";".join(reasons),
        translation_error_px=float(np.linalg.norm(translation)),translation_delta_px=translation.tolist(),
        angle_error_deg=abs(float(best["angle"])),angle_delta_deg=float(best["angle"]),
        physical_motion_error_px=float(np.linalg.norm(translation))+radius*math.radians(abs(best["angle"])),
        symmetry_order=symmetry_order,object_radius_px=radius,
        fit_iou=best["fit_iou"],visible_fraction=best["visible_fraction"],
        missing_fraction=best["missing_fraction"],extra_fraction=best["extra_fraction"],
        scoring_only=True,physical_position="full-area centroid",output_repaired=False)


def calibrate_physical_measurement(dataset, palette_calibration, *, letters=TRAIN_LETTERS, output=None):
    """Finite known-pose renders from recorded seen states; zero physics steps.

    Acceptance precision is fixed in PROTOCOL before any model result. Severe
    synthetic blue occlusions are rendering stress cases, not simulated controls.
    """
    import numpy as np
    from .envs.official import official_shape
    if output is not None and Path(output).exists():
        raise FileExistsError("Do not overwrite a measured physical calibration")
    if set(letters)-set(TRAIN_LETTERS):
        raise ValueError("Physical measurement calibration cannot inspect unseen letters")
    indices = {entry["shape"]:i for i,entry in enumerate(dataset.entries)}
    rows = []
    cases = ((0.,0.,0.,False),(0.,0.,0.,True),(8.,-5.,0.,True),(0.,0.,15.,True),(5.,4.,-20.,False),(0.,0.,180.,False))
    for letter in letters:
        if letter not in indices:
            raise ValueError("Missing seen calibration letter: "+letter)
        entry,arrays = dataset.trajectory(indices[letter])
        original = np.asarray(arrays["states"][0],dtype=np.float64).copy()
        original[:2] = (30,480)
        shape = official_shape(letter)
        local = np.asarray(shape.centroid,dtype=np.float64)
        c,s = math.cos(original[4]),math.sin(original[4])
        center = original[2:4]+np.asarray(((c,-s),(s,c)))@local
        # A recorded wall scene rotated in place can legitimately leave the
        # viewport. Canonicalize its *known* area centroid to the viewport center
        # for positive precision fixtures; this is rendering, never a reset/step.
        original[2:4] += np.asarray((256.,256.))-center
        center = np.asarray((256.,256.))
        reference = dataset.render_layers(entry,original)["full_object_mask"]
        for dx,dy,angle,occlude in cases:
            state = original.copy()
            state[4] += math.radians(angle)
            c,s = math.cos(state[4]),math.sin(state[4])
            state[2:4] = center+(dx,dy)-np.asarray(((c,-s),(s,c)))@local
            if occlude:
                full = dataset.render_layers(entry,state)["full_object_mask"] > .8
                coordinates = np.argwhere(full)
                if len(coordinates):
                    target = _centroid(full)[::-1]
                    pixel = coordinates[np.argmin(((coordinates-target)**2).sum(1))]
                    state[:2] = (pixel[::-1]+.5)*WORLD_SCALE
            prediction = dataset.render_layers(entry,state)["rgb"]
            fit = fit_physical_pose(prediction,reference,palette_calibration,symmetry_order=shape.symmetry_order)
            translation_error = None if fit["status"] != "ok" else float(np.linalg.norm(
                np.asarray(fit["translation_delta_px"])-(dx,dy)))
            angle_error = None if fit["status"] != "ok" else abs(equivalent_angle_degrees(
                fit["angle_delta_deg"]-angle,shape.symmetry_order))
            rows.append(dict(letter=letter,known_translation_px=[dx,dy],known_angle_deg=angle,
                rendered_blue_occlusion=occlude,raw_sha256=_hash_array(_rgb(prediction)),
                fit=fit,measurement_translation_error_px=translation_error,
                measurement_angle_error_deg=angle_error,
                passed=bool(fit["status"] == "ok" and translation_error <= PROTOCOL["maximum_calibration_translation_error_px"]
                            and angle_error <= PROTOCOL["maximum_calibration_angle_error_deg"])))
    valid = [row for row in rows if row["passed"]]
    result = dict(kind="native-rgb-physical-measurement-calibration.v1", protocol=dict(PROTOCOL),
        physical_source_sha256=_hash_file(__file__),palette_calibration=palette_calibration,
        letters=list(letters),rows=rows,physical_measurement_calibrated=len(valid)==len(rows),
        calibration_cases=len(rows),calibration_passed=len(valid),physics_steps=0,gpu_calls=0,
        fixture_recentring="known translation of recorded initial pose to full-area centroid(256,256); render-only",
        model_outputs_used=False,acceptance_limits_fixed_before_model_outputs=True,
        maximum_observed_translation_error_px=max((r["measurement_translation_error_px"] for r in rows
                                                  if r["measurement_translation_error_px"] is not None),default=None),
        maximum_observed_angle_error_deg=max((r["measurement_angle_error_deg"] for r in rows
                                             if r["measurement_angle_error_deg"] is not None),default=None))
    result["sha256"] = hashlib.sha256(json.dumps(_json_value(result),sort_keys=True).encode()).hexdigest()
    if output is not None:_write(output,result)
    return result


def physical_metric_identity(calibration):
    if isinstance(calibration,(str,Path)):
        calibration=json.loads(Path(calibration).read_text(encoding="utf-8"))
    raw={key:value for key,value in calibration.items() if key!="sha256"}
    expected=hashlib.sha256(json.dumps(_json_value(raw),sort_keys=True).encode()).hexdigest()
    if (calibration.get("sha256")!=expected or calibration.get("physical_source_sha256")!=_hash_file(__file__)
            or calibration.get("protocol")!=PROTOCOL or calibration.get("model_outputs_used") is not False):
        raise ValueError("Physical measurement calibration/source/protocol identity differs")
    return dict(physical_source_sha256=_hash_file(__file__),physical_calibration_sha256=expected,
                physical_measurement_calibrated=calibration["physical_measurement_calibrated"])


def _raw_addresses(core_report,dataset,trace,base_seed):
    """Map core window/step to the same returned trace items, including shrink."""
    import numpy as np
    entries={entry.get("trajectory_id",entry.get("file",str(i))):i for i,entry in enumerate(dataset.entries)}
    result={}
    def actions_hash(window,step):
        identifier,anchor=window.rsplit(":",1)
        _,arrays=dataset.trajectory(entries[identifier])
        anchor=int(anchor)
        return _hash_array(np.asarray(arrays["actions"])[anchor-3+step:anchor+1+step].astype(np.float32))
    maximum=max(core_report["horizons"])
    if int(core_report.get("inference_batch_size",1)) == 1:
        for window in {row["window"] for row in core_report["rows"]}:
            for step in range(maximum):
                seed=(_stable_seed(window,base_seed)+step)%2**32
                matches=[record for record in trace.records if record["mode"]=="single" and record["seed"]==seed
                         and record["actions_sha256"]==[actions_hash(window,step)]]
                if len(matches)==1:result[window,step+1]=(matches[0]["id"],0)
        return result
    for batch in core_report.get("batches",[]):
        active=list(batch["windows"])
        for step in range(maximum):
            if not active:break
            seed=(int(batch["batch_seed"])+step)%2**32
            expected=[actions_hash(window,step) for window in active]
            matches=[record for record in trace.records if record["mode"]=="batch" and record["seed"]==seed
                     and record["actions_sha256"]==expected]
            if len(matches)!=1:break
            record=matches[0]
            if record["status"]!="returned" or record.get("shape_invalid"):break
            for index,window in enumerate(active):result[window,step+1]=(record["id"],index)
            active=[window for window,valid in zip(active,record["item_valid"]) if valid]
    return result


def _summarize(rows):
    import numpy as np
    valid=[row for row in rows if row["status"]=="ok"]
    result=dict(total=len(rows),measurable=len(valid),unmeasurable=len(rows)-len(valid),
        reasons=dict(Counter(row["reason"] for row in rows if row["status"]!="ok")))
    for metric in ("translation_error_px","angle_error_deg","physical_motion_error_px","fit_iou"):
        values=[row[metric] for row in valid]
        result[metric]=dict(mean=float(np.mean(values)) if values else None,
                           p95=float(np.percentile(values,95)) if values else None)
    return result


def score_from_trace(core_report,dataset,traces,calibration,*,base_seed=5478,output=None,allow_final=False):
    """Score identical raw outputs, without rerunning or changing the predictor."""
    import numpy as np
    from .envs.official import official_shape
    if dataset.split in ("seen_test","unseen_test") and not allow_final:
        raise ValueError("Final physical measurements require explicit server/frozen authorization")
    if isinstance(calibration,(str,Path)):
        calibration=json.loads(Path(calibration).read_text(encoding="utf-8"))
    identity=physical_metric_identity(calibration)
    base_seed=int(core_report.get("evaluation_protocol",{}).get("seed",base_seed))
    palette=calibration["palette_calibration"]
    indices={entry.get("trajectory_id",entry.get("file",str(i))):i for i,entry in enumerate(dataset.entries)}
    addresses={method:_raw_addresses(core_report,dataset,trace,base_seed) for method,trace in traces.items()}
    layers_cache=OrderedDict()
    rows=[]
    for original in core_report["rows"]:
        row={key:original[key] for key in ("window","method","horizon","letter","anchor","trajectory_id",
                                           "contact","rotation","boundary","stationary") if key in original}
        row.update(status="unmeasurable",reason="core_prediction_failed")
        try:
            if original["status"]!="ok":
                rows.append(row)
                continue
            key=(original["window"],original["horizon"])
            if key not in layers_cache:
                entry,arrays=dataset.trajectory(indices[original["trajectory_id"]])
                anchor,horizon=original["anchor"],original["horizon"]
                truth=dataset.render_layers(entry,arrays["states"][anchor+horizon])
                hold=dataset.render_state(entry,arrays["states"][anchor])
                layers_cache[key]=(truth,hold)
                while len(layers_cache)>8:layers_cache.popitem(last=False)
            truth,hold=layers_cache[key]
            method=original["method"]
            if method=="hold":
                prediction=_rgb(hold)
                row["raw_sha256"]=_hash_array(prediction)
            else:
                trace=traces.get(method)
                address=addresses.get(method,{}).get(key)
                prediction=None if trace is None or address is None else trace.raw(*address)
                if prediction is None:
                    row["reason"]="trace_raw_missing_or_cache_limit"
                    rows.append(row)
                    continue
                row["trace_call_id"],row["trace_item_index"]=address
                row["raw_sha256"]=_hash_array(prediction)
            row.update(fit_physical_pose(prediction,truth["full_object_mask"],palette,
                symmetry_order=official_shape(row["letter"]).symmetry_order,protocol=calibration["protocol"]))
            if "rgb_mse" in original:
                mse=float(((_rgb(prediction)-_rgb(truth["rgb"]))**2).mean())
                row["same_raw_core_rgb_mse"] = abs(mse-float(original["rgb_mse"])) <= 1e-8
                if not row["same_raw_core_rgb_mse"]:
                    row.update(status="unmeasurable",reason="trace_output_does_not_match_core_rgb_error")
        except Exception as exc:
            row.update(status="unmeasurable",reason=type(exc).__name__+": "+str(exc)[:200])
        rows.append(row)
    summary,qualification={},{}
    for method in core_report["methods"]:
        summary[method],qualification[method]={},{}
        for horizon in core_report["horizons"]:
            chosen=[row for row in rows if row["method"]==method and row["horizon"]==horizon]
            summary[method][str(horizon)]=dict(overall=_summarize(chosen),
                per_letter={letter:_summarize([r for r in chosen if r["letter"]==letter])
                            for letter in sorted({r["letter"] for r in chosen})},
                subsets={name:_summarize([r for r in chosen if r.get(name)]) for name in ("contact","rotation","boundary","stationary")})
            if method=="hold":continue
            hold={row["window"]:row for row in rows if row["method"]=="hold" and row["horizon"]==horizon}
            checks={}
            for name in ("contact","rotation","boundary"):
                selected=[row for row in chosen if row.get(name)]
                paired=[(row,hold[row["window"]]) for row in selected if row["status"]=="ok"
                    and row["window"] in hold and hold[row["window"]]["status"]=="ok"]
                baseline=float(np.mean([b["physical_motion_error_px"] for _,b in paired])) if paired else None
                error=float(np.mean([a["physical_motion_error_px"] for a,_ in paired])) if paired else None
                improvement=(baseline-error)/baseline if baseline is not None and baseline>1e-8 else None
                translation_precision=calibration.get("maximum_observed_translation_error_px")
                angle_precision=calibration.get("maximum_observed_angle_error_deg")
                translation_precision=PROTOCOL["maximum_calibration_translation_error_px"] if translation_precision is None else translation_precision
                angle_precision=PROTOCOL["maximum_calibration_angle_error_deg"] if angle_precision is None else angle_precision
                uncertainty=float(np.mean([2*(translation_precision+b["object_radius_px"]*math.radians(angle_precision))
                                           for _,b in paired])) if paired else None
                absolute_gain=baseline-error if baseline is not None else None
                checks[name]=dict(total=len(selected),paired_measurable=len(paired),hold_error_px=baseline,
                    model_error_px=error,relative_improvement=improvement,
                    absolute_improvement_px=absolute_gain,finite_calibration_gain_uncertainty_px=uncertainty,
                    passed=bool(identity["physical_measurement_calibrated"] and selected and len(paired)==len(selected)
                                and improvement is not None and improvement>=PROTOCOL["required_motion_improvement"]
                                and absolute_gain>uncertainty))
            qualification[method][str(horizon)]=dict(subsets=checks,
                physical_motion_passed=bool(chosen and all(row["status"]=="ok" for row in chosen)
                    and all(value["passed"] for value in checks.values())),
                total=len(chosen),unmeasurable=sum(row["status"]!="ok" for row in chosen))
    tuned=qualification.get("fine_tuned",{})
    result=dict(kind="native-rgb-offline-physical-metrics.v1",identity=identity,split=dataset.split,
        physical_measurement_calibrated=identity["physical_measurement_calibrated"],
        physical_dynamics_verified=bool(identity["physical_measurement_calibrated"]
            and {str(h) for h in PROTOCOL["required_gate_horizons"]}.issubset(tuned)
            and core_report.get("all_letter_coverage_complete",False)
            and all(value["physical_motion_passed"] for value in tuned.values())),
        future_truth_used_only_for_offline_scoring=True,fitted_output_entered_sampler=False,
        raw_prediction_reused_without_rerun=True,rows=rows,summary=summary,qualification=qualification,
        traces={method:trace.snapshot() for method,trace in traces.items()},
        limitations="Finite 96px pose-fitting calibration; low-fit, clipped and missing-output rows remain unmeasurable in original denominators")
    if output is not None:
        output=Path(output)
        output.mkdir(parents=True,exist_ok=True)
        if (output/"physical-report.json").exists():raise FileExistsError("Preserve existing physical measurements")
        _write(output/"physical-report.json",result)
        fields=sorted({key for row in rows for key in row})
        with (output/"physical-metrics.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
    return result
