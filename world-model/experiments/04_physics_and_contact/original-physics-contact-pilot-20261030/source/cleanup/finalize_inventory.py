import json
from pathlib import Path

d=Path(__file__).resolve().parent
p=d/'inventory.json'
value=json.loads(p.read_text(encoding='utf-8'))
receipt=json.loads((d/'checkpoint-deletion-receipt.json').read_text(encoding='utf-8'))
verify=json.loads((d/'retained-assets-verification.json').read_text(encoding='utf-8'))
value['status']='checkpoint_cleanup_executed_by_root_old_training_data_retained'
value['actual_deletion_receipt']=str(d/'checkpoint-deletion-receipt.json')
value['actual_deleted_files']=len(receipt['deleted'])
value['actual_deleted_logical_bytes']=receipt['actual_deleted_logical_bytes']
value['old_training_data_deleted']=False
value['old_training_data_retention_reason']='Full original loader/trajectory audit depends on full R3A manifest; root chose to preserve59MB after recovering569MiB from retired weights, rather than claim an unsupported minimal reproducible subset.'
value['retained_assets_verification']=str(d/'retained-assets-verification.json')
value['post_cleanup_verification_passed']=verify['passed']
value['shared_bytes_post_cleanup_snapshot']=verify['shared_bytes_snapshot']
value['headroom_after_atomic_reserve_bytes']=verify['headroom_after_atomic_reserve_bytes']
value['filesystem_post_cleanup_snapshot']=verify['filesystem_snapshot']
value['snapshot_limit']=verify['snapshot_limit']
p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'inventory':str(p),'status':value['status'],'actual_deleted_bytes':value['actual_deleted_logical_bytes'],
                 'verification_passed':verify['passed'],'shared_bytes':verify['shared_bytes_snapshot'],
                 'headroom_after_atomic_reserve_bytes':verify['headroom_after_atomic_reserve_bytes']},indent=2))
