"""Freeze the directly requested recovery, preserving all original evidence."""
import ast,ctypes,json,sys
from ctypes import wintypes
from pathlib import Path
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
from continuation_io import install
from common import ROOT,read,sha,utc

def main():
    io=install();assert not (ROOT/'continuation-config.json').exists() and not (ROOT/'continuation-execution-claim.json').exists()
    original=read(ROOT/'frozen-config.json');done=read(ROOT/'completion.json')
    assert sha(ROOT/'frozen-config.json')=='8439a8108ee606e9ecd7bdc61de92645e68d30cbfe6404c283fde0adc2f24e56'
    assert done['failed'] and done['failure']['type']=='PermissionError' and done['ledger']['updates']=={'BASE':128,'SNR':128}
    auth=read(ROOT/'CONTINUATION_HUMAN_AUTHORIZATION.json');assert auth['resume_authorized'] and auth['only_final_report']
    assert read(ROOT/'continuation-static-checks.json')['passed']
    reused=[];paths=[]
    for bid in (0,4):
        files=[]
        for slot in range(4):
            p=ROOT/f'evaluation/SNR/real_history-b{bid:02d}-s{slot}.npz';assert p.is_file() and p.stat().st_size==1769732
            files.append(dict(slot=slot,path=str(p),bytes=p.stat().st_size,sha256=sha(p)));paths.append(p)
        reused.append(dict(mode='real_history',batch_id=bid,files=files))
    paths += [ROOT/x for x in ('continuation_io.py','continuation_ledger.py','continuation_worker.py','continuation_checks.py','continuation_postrun_audit.py','prepare_continuation.py','launch_continuation.ps1','CONTINUATION_HUMAN_AUTHORIZATION.json','completion.json','failure.json','training/SNR-128.json','training/SNR-128.pt','evaluation/BASE/metrics.json')]
    for p in ROOT.glob('continuation_*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    assert sum(v['images_attempted'] for v in done['ledger']['counts'].values())==8276
    value=dict(run_id=original['run_id']+'-CONT1',original_run_id=original['run_id'],original_config_sha256=sha(ROOT/'frozen-config.json'),
        original_completion_sha256=sha(ROOT/'completion.json'),original_failure_sha256=sha(ROOT/'failure.json'),
        direct_user_authorization=auth['direct_user_messages'],final_only_report=True,
        recovery_scope='Fix bounded Windows replace retry and immutable cost snapshots, reuse8 persisted SNR archives, finish SNR eval/GDL/FFL unchanged. No BASE/SNR retraining, no new preflight, no loss/data/physics/precision/seed changes.',
        reused_batches=reused,original_charged_image_forwards=8276,original_completed_image_forwards=8272,remaining_new_image_forwards=14944,
        expected_final_charged_image_forwards=23220,expected_final_completed_image_forwards=23216,unchanged_total_image_cap=23296,unchanged_batch_cap=6016,
        expected_final_charged_batch_forwards=5901,expected_final_completed_batch_forwards=5900,
        unique_eval_predictions=4992,completed_sampler_predictions_with_lost_prefix_replay=5000,
        original_aborted_pre_body_reservation_charged=4,original_completed_replayed_denoiser_images=28,
        model_constructions_cumulative_cap=2,one_active_CUDA_process_at_a_time=True,preflight_E2_model_only_loads=1,continuation_SNR_model_only_loads=1,formal_E2_full_arm_restores=4,
        formal_native_exposures_each=2048,formal_attempts_each=128,formal_updates_total_cap=512,formal_backward_cap=2048,
        io_change='Same-temp PermissionError retry <=2s, no retry of model calls. Immutable before/after-forward journal avoids hot ledger.json replacement. All journal bytes remain in900MB root cap.',
        root_cap=900000000,shared_cap=10000000000,atomic_reserve=268435456,external_growth_guard=67108864,final_metadata_reserve=16777216,
        windows_task_name='Tdyssey-LLS-20261102-continue1',original_windows_task_name='Tdyssey-LLS-20261102',
        original_scoring_parity_reused=True,old_SNR_noise_receipt_provenance='Actual noise was checked by frozen capture_noise before files persisted; in-memory hash receipt lost. Independent same-seed B4 replay reconstructs canonical hashes, explicitly not a new native observation.',
        bindings=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(paths))],at_utc=utc())
    io.write_json('continuation-config.json',value,exclusive=True);digest=sha(ROOT/'continuation-config.json')
    io.write_json('continuation-release.json',dict(run_id=value['run_id'],config_sha256=digest,resume_authorized=True,source='Direct human request in this owner chat after original failure; finish all and report once',at_utc=utc()),exclusive=True)
    io.write_json('continuation-freeze-receipt.json',dict(run_id=value['run_id'],config_sha256=digest,bindings=len(paths),model_calls_during_repair=0,storage=io.guard(refresh=True),at_utc=utc()),exclusive=True)
    print(json.dumps(dict(run_id=value['run_id'],config_sha256=digest,remaining_image_forwards=14944,expected_cumulative_charged=23220,total_cap=23296,models_started_by_repair=0,storage=io.guard())))

if __name__=='__main__':main()

