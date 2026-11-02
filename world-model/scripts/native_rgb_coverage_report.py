"""Derive static coverage figures from frozen native-RGB data; no physics/model."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(data):
    import numpy as np
    from tdyssey.envs.official import official_shape
    from tdyssey.native_rgb_data import TRAIN_LETTERS
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf8"))
    audit=json.loads((data/"physics-audit.json").read_text(encoding="utf8"))
    if audit["dataset_manifest_sha256"]!=sha256(data/"manifest.json"):
        raise ValueError("Audit/manifest identity does not match")
    rows=[];target_rows=[];sources={"manifest.json":sha256(data/"manifest.json"),"physics-audit.json":sha256(data/"physics-audit.json")}
    for split in ("train","seen_val"):
        path=data/"coverage"/split/"coverage.json";coverage=json.loads(path.read_text(encoding="utf8"))
        sources[str(path.relative_to(data))]=sha256(path)
        for letter in TRAIN_LETTERS:
            segments=coverage["segments"][letter];reachability=coverage["canonical_reachability"][letter]
            reachable={row["segment_id"] for row in reachability if row["status"].startswith("reachable")}
            closed={row["segment_id"] for row in reachability if row["status"]=="proven_closed_hole_from_exterior"}
            unconfirmed=set(range(len(segments)))-reachable-closed
            hit={row["id"] for row in segments if row["contact_count"]}
            observed_reachable=reachable|hit
            observed_unknown=set(range(len(segments)))-observed_reachable-closed
            conflict=hit&closed
            selected=[entry for entry in manifest["entries"] if entry["split"]==split and entry["shape"]==letter]
            trajectory_hit_bins={}
            for segment in segments:
                for trajectory_id in segment["trajectory_ids"]:
                    trajectory_hit_bins.setdefault(trajectory_id,set()).add(segment["id"])
            letter_targets=[]
            vertices=np.asarray([point for rectangle in official_shape(letter).components for point in rectangle.vertices])
            proximity=np.zeros(8,dtype=np.int64);frames=0;contacts=0;controls=0;trajectories=0;point_count=0
            for entry in selected:
                with np.load(data/entry["file"],allow_pickle=False) as arrays:
                    state=arrays["states"];angles=state[:,4];c=np.cos(angles)[:,None];s=np.sin(angles)[:,None]
                    x=c*vertices[None,:,0]-s*vertices[None,:,1]+state[:,None,2]
                    y=s*vertices[None,:,0]+c*vertices[None,:,1]+state[:,None,3]
                    left=x.min(1)<23;right=x.max(1)>488;top=y.min(1)<23;bottom=y.max(1)>488
                    masks=(left,right,top,bottom,left&top,right&top,left&bottom,right&bottom)
                    proximity+=np.asarray([mask.sum() for mask in masks]);frames+=len(state)
                    count=int((arrays["contact"]>0).sum());contacts+=count;controls+=len(arrays["actions"])
                    trajectories+=int(count>0);point_count+=len(arrays["contact_points"])
                    planned=entry.get("staging_segment");actual_bins=trajectory_hit_bins.get(entry["trajectory_id"],set())
                    valid_planned=type(planned)is int and 0<=planned<len(segments)
                    old_staging=(entry.get("reset_records") or [{}])[-1].get("reachability") or {}
                    target=dict(split=split,letter=letter,trajectory_id=entry["trajectory_id"],planned_arc=planned,
                        any_real_contact=count>0,planned_arc_actually_hit=planned in actual_bins,
                        distinct_real_hit_bins=len(actual_bins),real_hit_bin_ids=" ".join(map(str,sorted(actual_bins))),
                        planned_arc_is_closed_hole=bool(valid_planned and segments[planned]["ring"]>0),
                        planned_arc_canonical_classification=reachability[planned]["status"] if valid_planned else "missing_or_invalid",
                        recorded_staging_reachability_status=old_staging.get("status","not_recorded"),
                        actual_contact_controls=count)
                    letter_targets.append(target);target_rows.append(target)
            angle_bins=[sum(segment["actual_contact_angle_bins"].get(str(index),0) for segment in segments) for index in range(8)]
            missing=sorted(reachable-hit)
            row=dict(split=split,letter=letter,total_arc_bins=len(segments),all_hit_bins=len(hit),
                all_hit_percent=100*len(hit)/len(segments),reachable_bins=len(reachable),reachable_hit_bins=len(reachable&hit),
                reachable_hit_percent=100*len(reachable&hit)/max(1,len(reachable)),
                closed_hole_bins=len(closed),closed_hole_percent=100*len(closed)/len(segments),
                unconfirmed_bins=len(unconfirmed),unconfirmed_percent=100*len(unconfirmed)/len(segments),
                observed_reachable_lower_bound_bins=len(observed_reachable),
                observed_reachable_hit_percent=100*len(hit)/max(1,len(observed_reachable)),
                observed_unknown_bins=len(observed_unknown),observed_unknown_percent=100*len(observed_unknown)/len(segments),
                canonical_only_unhit_bins=len(reachable-hit),
                proven_closed_without_hit_bins=len(closed-hit),closed_hole_hit_conflict_bins=" ".join(map(str,sorted(conflict))),
                observed_reachability_evidence_consistent=not bool(conflict),
                independent_trajectories=len(selected),independent_contact_trajectories=trajectories,
                controls=controls,contact_controls=contacts,contact_control_percent=100*contacts/max(1,controls),
                recorded_contact_points=point_count,mapped_contact_points=sum(angle_bins),recorded_frames=frames,
                missing_reachable_bins=" ".join(map(str,missing)))
            planned_hits=sum(target["planned_arc_actually_hit"] for target in letter_targets)
            row.update(planned_arc_hit_trajectories=planned_hits,
                planned_arc_hit_percent=100*planned_hits/max(1,len(letter_targets)),
                planned_closed_hole_trajectories=sum(target["planned_arc_is_closed_hole"] for target in letter_targets),
                planned_closed_hole_actually_hit_trajectories=sum(target["planned_arc_is_closed_hole"] and target["planned_arc_actually_hit"] for target in letter_targets))
            row.update({f"angle_bin_{index}":value for index,value in enumerate(angle_bins)})
            row.update({f"near_{name}_frames":int(count) for name,count in zip(("left","right","top","bottom","top_left","top_right","bottom_left","bottom_right"),proximity)})
            rows.append(row)
    return rows,target_rows,manifest,audit,sources


def render_figures(rows,output):
    import numpy as np
    os.environ.setdefault("MPLCONFIGDIR",str(output/".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    figure,axes=plt.subplots(1,2,figsize=(16,10),sharey=False)
    for axis,split,title in zip(axes,("train","seen_val"),("Training: 22–23 independent trajectories / letter","Seen-letter validation: 6 independent trajectories / letter")):
        selected=[row for row in rows if row["split"]==split];y=np.arange(len(selected))
        axis.barh(y-.17,[row["all_hit_percent"] for row in selected],height=.30,color="#3264a8",label="All contour bins hit / all bins")
        axis.barh(y+.17,[row["reachable_hit_percent"] for row in selected],height=.30,color="#14834e",label="Verified reachable bins hit / reachable bins")
        axis.set_yticks(y,[f"{row['letter']}   {row['independent_contact_trajectories']} contact traj." for row in selected])
        axis.invert_yaxis();axis.set_xlim(0,116);axis.set_xticks((0,25,50,75,100))
        axis.set_xlabel("Actual contact coverage (%) — denominators differ")
        axis.set_title(title,fontsize=12,pad=14)
        axis.set_axisbelow(True);axis.grid(axis="x",color="#dddddd",linewidth=.7)
        for index,row in enumerate(selected):
            axis.text(101,index+.17,f"{row['reachable_hit_bins']}/{row['reachable_bins']}",va="center",fontsize=8,color="#12643d")
        axis.legend(loc="lower left",bbox_to_anchor=(0,-.14),fontsize=9,frameon=False)
    figure.suptitle("Fixed 18-letter contact coverage · 8 world-pixel arc bins",fontsize=17,y=.97)
    figure.text(.5,.025,"Canonical reachability: closed holes excluded; staging, contact center and normal path checked in conservative exterior free space.\nTraining misses: L bin 2, O bin 55, P bin 55. Finite-budget gaps remain; validation coverage is substantially lower.",ha="center",fontsize=10)
    figure.subplots_adjust(left=.12,right=.98,top=.91,bottom=.16,wspace=.38)
    figure.savefig(output/"coverage-overview.png",dpi=160,facecolor="white");plt.close(figure)

    figure,axes=plt.subplots(1,2,figsize=(16,10))
    categories=(("all_hit_bins","Actual contact bins H","#3264a8"),
        ("canonical_only_unhit_bins","Canonical reachable, unhit R − H","#65ad83"),
        ("proven_closed_without_hit_bins","Proven closed holes (without hits)","#9a81b5"),
        ("observed_unknown_bins","Unresolved after R ∪ H","#d4d7dc"))
    for axis,split,title in zip(axes,("train","seen_val"),("Training","Seen-letter validation")):
        selected=[row for row in rows if row["split"]==split];y=np.arange(len(selected));left=np.zeros(len(selected))
        for key,label,color in categories:
            width=np.asarray([100*row[key]/row["total_arc_bins"] for row in selected])
            axis.barh(y,width,left=left,height=.66,label=label,color=color);left+=width
        axis.set_yticks(y,[row["letter"] for row in selected]);axis.invert_yaxis()
        axis.set_xlim(0,126);axis.set_xticks((0,25,50,75,100));axis.set_xlabel("Partition of all fixed contour bins (%)")
        axis.set_title(title+" · annotation = |H| / |R ∪ H|",fontsize=12)
        for index,row in enumerate(selected):
            note=f"{row['all_hit_bins']}/{row['observed_reachable_lower_bound_bins']} ({row['observed_reachable_hit_percent']:.1f}%)"
            axis.text(102,index,note,va="center",fontsize=8,
                color="#a32626" if row["closed_hole_hit_conflict_bins"] else "#222222")
    axes[0].legend(loc="lower left",bbox_to_anchor=(-.03,-.19),fontsize=9,frameon=False)
    figure.suptitle("Observed reachability lower bound: R ∪ H",fontsize=17,y=.97)
    figure.text(.5,.045,"R = canonical normal-path reachable bins. H = distinct bins hit by recorded physically valid exterior trajectories.\nR ∪ H is a global lower bound across observed poses within each split; it is not all physically reachable contour.\nRepeated touches do not enlarge H. Unresolved bins are not declared unreachable. Any hit/closed-hole conflict is a reported risk.",ha="center",fontsize=10)
    figure.subplots_adjust(left=.05,right=.98,top=.91,bottom=.20,wspace=.15)
    figure.savefig(output/"observed-reachability-lower-bound.png",dpi=160,facecolor="white");plt.close(figure)

    angle=np.asarray([[row[f"angle_bin_{i}"] for i in range(8)] for row in rows],dtype=float)
    angle/=np.maximum(angle.sum(1,keepdims=True),1)
    names=("left","right","top","bottom","top_left","top_right","bottom_left","bottom_right")
    near=np.asarray([[100*row[f"near_{name}_frames"]/max(1,row["recorded_frames"]) for name in names] for row in rows])
    trajectory=np.asarray([[row["independent_contact_trajectories"]] for row in rows])
    figure,axes=plt.subplots(1,3,figsize=(16,13),gridspec_kw={"width_ratios":[1.5,6,6]})
    labels=[("T" if row["split"]=="train" else "V")+" · "+row["letter"] for row in rows]
    maps=[axes[0].imshow(trajectory,cmap="viridis",vmin=0,vmax=24,aspect="auto"),
        axes[1].imshow(angle*100,cmap="YlGnBu",vmin=0,vmax=100,aspect="auto"),
        axes[2].imshow(near,cmap="OrRd",vmin=0,vmax=100,aspect="auto")]
    for axis in axes:
        axis.set_yticks(np.arange(len(rows)),labels,fontsize=8);axis.axhline(17.5,color="#333333",linewidth=1.5)
    axes[0].set_title("Independent\ncontact trajectories",fontsize=11)
    axes[0].set_xticks([0],["count"])
    for index,row in enumerate(rows):axes[0].text(0,index,str(row["independent_contact_trajectories"]),ha="center",va="center",fontsize=8,color="white" if row["independent_contact_trajectories"]<12 else "black")
    axes[1].set_title("Actual mapped contact approach-angle distribution (%)",fontsize=11)
    axes[1].set_xticks(range(8),["−180..−135","−135..−90","−90..−45","−45..0","0..45","45..90","90..135","135..180"],rotation=55,ha="right",fontsize=8)
    axes[2].set_title("Actual object proximity at recorded frames (%)",fontsize=11)
    axes[2].set_xticks(range(8),["Left","Right","Top","Bottom","Top-left","Top-right","Bottom-left","Bottom-right"],rotation=55,ha="right",fontsize=9)
    for index in (1,2):figure.colorbar(maps[index],ax=axes[index],fraction=.035,pad=.025,label="%")
    figure.suptitle("Contact and boundary diagnostics · T = training, V = seen-letter validation",fontsize=16,y=.98)
    figure.text(.5,.025,"Angles use pre-solve relative velocity against the actual collision normal; counts repeat at 100 Hz, not independent contacts.\nBoundary proximity: object bounding box within 16 world px of inner walls at 10 Hz, initial frames included. It is proximity, not wall collision.\nNear-corner means simultaneous proximity to its two adjacent walls. Wall-arbiter points were not retained; wall collision counts are unavailable.\nAll counts and denominator categories are in summary.csv; planned scene locations are not used as actual collision counts.",ha="center",fontsize=9)
    figure.subplots_adjust(left=.055,right=.97,top=.93,bottom=.16,wspace=.35)
    figure.savefig(output/"contact-boundary-diagnostics.png",dpi=160,facecolor="white");plt.close(figure)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"artifacts/native-rgb-20261028/data")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/native-rgb-20261028/report-coverages")
    args=parser.parse_args(argv);data=args.data.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    rows,target_rows,manifest,audit,sources=derive(data)
    with (output/"summary.csv").open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/"target-contact-comparison.csv").open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(target_rows[0]));writer.writeheader();writer.writerows(target_rows)
    target_summary={}
    for split in ("train","seen_val"):
        selected=[row for row in target_rows if row["split"]==split]
        planned_hits=sum(row["planned_arc_actually_hit"] for row in selected)
        target_summary[split]=dict(independent_trajectories=len(selected),
            independent_real_contact_trajectories=sum(row["any_real_contact"] for row in selected),
            planned_arc_hit_trajectories=planned_hits,planned_arc_hit_percent=100*planned_hits/max(1,len(selected)),
            planned_closed_hole_trajectories=sum(row["planned_arc_is_closed_hole"] for row in selected),
            planned_closed_hole_hit_trajectories=sum(row["planned_arc_is_closed_hole"] and row["planned_arc_actually_hit"] for row in selected),
            planned_miss_but_other_real_contact_trajectories=sum(row["any_real_contact"] and not row["planned_arc_actually_hit"] for row in selected))
    (output/"target-contact-summary.json").write_text(json.dumps(dict(
        semantics="Preselected object contour arc hit; not pusher click-target arrival",summary=target_summary,
        planned_closed_hole_cases=[row for row in target_rows if row["planned_arc_is_closed_hole"]]),indent=2)+"\n",encoding="utf8")
    render_figures(rows,output)
    text=["# Frozen-data contact coverage","", "This is an offline report. It performs no physics controls, model inference or GPU work, and changes no input data or denominator.","",
        f"Physical audit passed: **{audit['physics_collection_gate_passed']}**. Data: **{manifest['actual_controls']:,} controls, {manifest['actual_trajectories']} independent trajectories**.","",
        "![Coverage overview](coverage-overview.png)","","![Observed reachability lower bound](observed-reachability-lower-bound.png)","","![Contact and boundary diagnostics](contact-boundary-diagnostics.png)","",
        "| Split | Letter | All contour hit | Verified reachable hit | Closed-hole bins | Unconfirmed bins | Contact trajectories |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['split']} | {row['letter']} | {row['all_hit_bins']}/{row['total_arc_bins']} ({row['all_hit_percent']:.1f}%) | {row['reachable_hit_bins']}/{row['reachable_bins']} ({row['reachable_hit_percent']:.1f}%) | {row['closed_hole_bins']} ({row['closed_hole_percent']:.1f}%) | {row['unconfirmed_bins']} ({row['unconfirmed_percent']:.1f}%) | {row['independent_contact_trajectories']} |")
    text.extend(["", "## Reachability evidence from recorded contacts", "",
        "Let **R** be the canonical normal-path reachable-bin set, **H** the distinct actual collision-hit-bin set, and **C** the proven closed-hole-bin set. The physically audited trajectories start in the exterior; each actual hit is evidence that its bin was reachable in at least that observed pose. Repeated touches of one bin add no new member to H.","",
        "The global observed-poses reachability lower bound is **R ∪ H**, computed independently within training and validation. Its hit coverage is **|H| / |R ∪ H|**; unresolved bins are **all bins − (R ∪ H) − C**. This is a lower bound of demonstrated reachability, not a complete classification of all physically reachable bins. The original canonical denominator remains unchanged.","",
        "| Split | Letter | R ∪ H bins | Hit / observed reachable lower bound | Unresolved bins | H ∩ closed-hole conflict |",
        "|---|---|---:|---:|---:|---|"])
    for row in rows:
        text.append(f"| {row['split']} | {row['letter']} | {row['observed_reachable_lower_bound_bins']} | {row['all_hit_bins']}/{row['observed_reachable_lower_bound_bins']} ({row['observed_reachable_hit_percent']:.1f}%) | {row['observed_unknown_bins']} ({row['observed_unknown_percent']:.1f}%) | {row['closed_hole_hit_conflict_bins'] or 'none'} |")
    conflicts=[f"{row['split']}/{row['letter']}: {row['closed_hole_hit_conflict_bins']}" for row in rows if row["closed_hole_hit_conflict_bins"]]
    text.extend(["", "**Evidence conflict risk:** "+("; ".join(conflicts)+". Preserve and investigate mapping/topology/trajectory validity before treating these bins as reachability evidence." if conflicts else "No actual hit bin intersects a proven closed-hole bin in either split. This consistency check does not prove all remaining bins reachable or unreachable.")])
    text.extend(["", "## Planned contact arc versus actual collision bins", "",
        "The planned arc is the collector's preselected **object contour segment** (`staging_segment`). This metric is not the pusher click-target arrival rate. Actual hit bins come from real collision-point mappings for that trajectory; repeated contact with one bin contributes one distinct hit-bin member.","",
        "| Split | Independent trajectories | Any real contact | Planned arc actually hit | Planned closed-hole arcs | Closed-hole arcs actually hit |",
        "|---|---:|---:|---:|---:|---:|"])
    for split,summary in target_summary.items():
        text.append(f"| {split} | {summary['independent_trajectories']} | {summary['independent_real_contact_trajectories']} | {summary['planned_arc_hit_trajectories']} ({summary['planned_arc_hit_percent']:.1f}%) | {summary['planned_closed_hole_trajectories']} | {summary['planned_closed_hole_hit_trajectories']} |")
    text.extend(["", "The original sampler checked the staging center's exterior component without the later normal-contact-path and closed-hole-priority checks. Therefore a planned arc can be in a closed hole while its staging center is outside; contacts elsewhere on the object do not count as hitting that planned arc. These misses remain in the report and were not replaced or recollected.","",
        "[Per-trajectory planned/actual comparison](target-contact-comparison.csv) lists every trajectory, planned arc, any-real-contact flag, planned-arc-hit flag, number and identities of distinct actual hit bins, and the recorded versus reviewed classification. [Target-contact summary](target-contact-summary.json) retains all planned closed-hole cases."])
    text.extend(["", "The all-contour denominator includes outer and hole boundaries. Verified reachable coverage uses only canonical bins with conservative exterior-path evidence. Closed holes are inaccessible from the exterior. Unconfirmed paths are not claimed impossible; some have recorded contacts under other approaches/poses.","",
        "Training retains three verified-reachable misses: **L bin 2, O bin 55 and P bin 55**. Each has zero mapped contact at that bin despite contacts on the rest of the letter. Their precise failure causes are unconfirmed. The finite collection budget is exhausted; no replacement or extra collection was performed.","",
        "Validation uses six independent trajectories per letter and has much lower coverage. This report does not treat the small validation set as complete contour coverage or as an unseen-letter test.","",
        "Contact-angle histograms use mapped real collision points and pre-solve relative velocities, not commanded directions. Repeated 100 Hz contact points are correlated; independent contact-trajectory counts are reported separately.","",
        "Boundary statistics are independently derived from recorded states and the fixed existing geometry. They measure 16px bounding-box proximity to the four inner walls and simultaneous proximity at each corner, including initial frames. Wall-arbiter contact points were not retained, so actual wall-collision counts are unavailable and are not claimed. Planned wall/corner locations are never substituted for actual collisions.","",
        "[Full compact CSV](summary.csv) contains raw counts, percentages, all eight angle bins, all four walls/corners, and remaining reachable-bin IDs. Original per-letter SVGs remain under `../data/coverage/train/` and `../data/coverage/seen_val/`.","",
        f"Input manifest SHA256: `{sources['manifest.json']}`."])
    (output/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf8")
    record=dict(kind="native-rgb-derived-coverage-report-v1",input_data=str(data),source_sha256=sources,
        new_physics_controls=0,model_inference=False,gpu_used=False,input_files_modified=False,
        observed_reachability_lower_bound="R union H, independently per split; not complete physical reachability",
        closed_hole_hit_conflicts=conflicts,
        planned_contact_arc_summary=target_summary,wall_arbiter_contact_points_retained=False,
        actual_wall_collision_counts_available=False,
        generated_files=["coverage-overview.png","observed-reachability-lower-bound.png","contact-boundary-diagnostics.png","summary.csv","target-contact-comparison.csv","target-contact-summary.json","REPORT.md"])
    (output/"report-sources.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf8")
    print(json.dumps(record))


if __name__=="__main__":main()
