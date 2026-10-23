"""CUDA training/evaluation; imported only after command-line/source-only checks."""

from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import digest, json_bytes
from .models.visual_dynamics import VisualDynamics
from .world_model_data import VisualDynamicsDataset, episode_order, fit_motion_stats
from .world_model_protocol import MetricTotals
from .training_data import WindowSpec
from .tracking import RunLogger

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = 'visual-residual-cnn-v1'
LOSS_WEIGHTS = {'image':1.,'object_roi':2.,'agent_motion':.1,'object_motion':.1,'contact':.05,'reward':.1}


def cuda_setup(seed):
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required; no automatic CPU training fallback')
    torch.set_num_threads(4)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.cuda.reset_peak_memory_stats()
    return torch.device('cuda')


def to_device(batch, device):
    return {key:{name:value.to(device) for name,value in batch[key].items()}
            for key in ('observations','targets','labels')} | {'actions':batch['actions'].to(device)}


def training_loss(model, prediction, batch):
    error=(prediction['image']-batch['targets']['image']).square().mean(2)
    roi=batch['labels']['object_roi'][:, :, 0].to(error.dtype)
    normalized=(batch['labels']['motion']-model.motion_mean)/model.motion_scale
    values={'image':error.mean(),'object_roi':((error*roi).sum((-1,-2))/roi.sum((-1,-2)).clamp_min(1)).mean(),
            'agent_motion':F.mse_loss(prediction['normalized_motion'][:, :, :2],normalized[:, :, :2]),
            'object_motion':F.mse_loss(prediction['normalized_motion'][:, :, 2:],normalized[:, :, 2:]),
            'contact':F.binary_cross_entropy_with_logits(prediction['contact_logit'],(batch['labels']['n_contacts']>0).float()),
            'reward':F.mse_loss(prediction['reward'],batch['labels']['reward'])}
    return sum(LOSS_WEIGHTS[key]*value for key,value in values.items()),values


def persistence(batch, stats):
    b,f=batch['actions'].shape[:2]
    device=batch['actions'].device
    return {'image':batch['observations']['image'][:, -1:, :].expand(-1,f,-1,-1,-1),
            'agent_pos':batch['observations']['agent_pos'][:, -1:].expand(-1,f,-1),
            'motion':torch.zeros((b,f,5),device=device),
            'reward':torch.full((b,f),stats['mean_reward'],device=device),
            'contact_probability':torch.full((b,f),stats['contact_frequency'],device=device)}


def metric_values(prediction, batch):
    error=(prediction['image']-batch['targets']['image']).square().mean(2)
    roi=batch['labels']['object_roi'][:, :, 0].to(error.dtype)
    # At most four horizons: explicit ordered sums also work in CUDA's strict
    # deterministic mode, unlike its cumsum kernel in the locked torch profile.
    increments=prediction['motion'][:, :, 2:].unbind(1)
    positions=[increments[0]]
    for increment in increments[1:]:
        positions.append(positions[-1]+increment)
    cumulative=torch.stack(positions,dim=1)
    delta=cumulative-batch['labels']['object_displacement']
    angle=torch.atan2(torch.sin(delta[:, :, 2]),torch.cos(delta[:, :, 2])).abs()
    contact=(batch['labels']['n_contacts']>0).float()
    probability=prediction['contact_probability'] if 'contact_probability' in prediction else prediction['contact_logit'].sigmoid()
    return {'image_mse':error.mean((-1,-2)),
            'object_roi_mse':(error*roi).sum((-1,-2))/roi.sum((-1,-2)).clamp_min(1),
            'agent_position_error_pixels':(prediction['agent_pos']-batch['targets']['agent_pos']).norm(dim=2),
            'object_position_error_pixels':delta[:, :, :2].norm(dim=2),
            'object_angle_error_degrees':angle*180/torch.pi,
            'reward_mae':(prediction['reward']-batch['labels']['reward']).abs(),
            'contact_brier':(probability-contact).square(),
            'contact_accuracy':((probability>=.5)==contact.bool()).float()}


def add_metrics(totals, values, contact):
    arrays={key:value.detach().cpu().numpy() for key,value in values.items()}
    contact=contact.detach().cpu().numpy()
    for h in range(contact.shape[1]):
        for name,mask in (('all',np.ones(len(contact),dtype=np.bool_)),('contact',contact[:, h]),('noncontact',~contact[:, h])):
            sums={key:float(value[mask,h].sum(dtype=np.float64)) for key,value in arrays.items()}
            totals.add(h+1,name,int(mask.sum()),sums)


@torch.no_grad()
def evaluate(model, dataset, stats, batch_size=8, deadline=None):
    model.eval()
    totals,baseline=MetricTotals(dataset.spec.future),MetricTotals(dataset.spec.future)
    count=0
    for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,
                            generator=torch.Generator().manual_seed(0)):
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('World-model wall-time limit reached during validation')
        batch=to_device(batch,next(model.parameters()).device)
        prediction=model(batch['observations'],batch['actions'])
        contact=batch['labels']['n_contacts']>0
        add_metrics(totals,metric_values(prediction,batch),contact)
        add_metrics(baseline,metric_values(persistence(batch,stats),batch),contact)
        count+=len(batch['actions'])
    return {'windows':count,'horizons':totals.result(),'persistence':baseline.result()}


def save_json(path,value,max_bytes):
    payload=json_bytes(value)
    if len(payload)+sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file() and p!=path)>max_bytes:
        raise ValueError('World-model output byte budget exceeded')
    temporary=path.with_suffix('.tmp')
    temporary.write_bytes(payload);temporary.replace(path)


def save_checkpoint(path,model,stats,config,catalog_hash,epoch,training_state=None):
    payload={'model_version':MODEL_VERSION,'state_dict':model.state_dict(),'normalization':stats,
             'config':asdict(config),'catalog_sha256':catalog_hash,'epoch':epoch,
             'model_inputs':['two RGB images','two pusher positions','goal pose','future actions'],
             'privileged_supervision':['object motion','geometric image ROI','reward','contact']}
    if training_state is not None:
        payload['training_state']=training_state
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    if sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())>config.max_bytes:
        temporary.unlink();raise ValueError('World-model output byte budget exceeded')
    temporary.replace(path)


def load_checkpoint(path,device):
    payload=torch.load(path,map_location=device,weights_only=True)
    if payload['model_version']!=MODEL_VERSION:
        raise ValueError('Unknown visual world-model checkpoint version')
    model=VisualDynamics(payload['normalization']).to(device)
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.eval(),payload


@torch.no_grad()
def save_preview(model,dataset,output):
    """First validation window with contact; never choose by prediction quality."""
    from PIL import Image,ImageDraw
    sample=None
    for index in range(len(dataset)):
        candidate=dataset[index]
        if sample is None:sample=candidate
        if (candidate['labels']['n_contacts']>0).any():
            sample=candidate;break
    batch=torch.utils.data.default_collate([sample])
    gpu=to_device(batch,next(model.parameters()).device)
    prediction=model(gpu['observations'],gpu['actions'])['image'][0].cpu().numpy()
    rows=dataset.spec.future
    canvas=Image.new('RGB',(4*192,rows*220),(245,247,250));draw=ImageDraw.Draw(canvas)
    for h in range(rows):
        frames=[sample['observations']['image'][-1],sample['targets']['image'][h],prediction[h],sample['observations']['image'][-1]]
        for col,(label,frame) in enumerate(zip(('Input','Actual','Predicted','Persistence'),frames)):
            image=Image.fromarray(np.rint(frame.transpose(1,2,0).clip(0,1)*255).astype(np.uint8)).resize((192,192))
            canvas.paste(image,(col*192,h*220+25));draw.text((col*192+5,h*220+5),f'{label} +{h+1}',fill=(20,30,40))
    canvas.save(output)
    return {'episode_id':sample['episode_id'],'anchor':sample['anchor'],'selection':'first validation window with any contact; fallback first window'}


def check_training_resume(payload,config,catalog_hash,source_hashes):
    if payload.get('model_version')!=MODEL_VERSION or 'training_state' not in payload:
        raise ValueError('Resume requires a world-model last.pt recovery checkpoint')
    allowed={'epochs','max_steps','seconds','max_bytes'}
    for name,value in payload['config'].items():
        if name not in allowed and value!=getattr(config,name):
            raise ValueError(f'Resume changes world-model protocol: {name}')
    state=payload['training_state']
    if payload['catalog_sha256']!=catalog_hash or state['report']['provenance']['source_sha256']!=source_hashes:
        raise ValueError('Resume requires the same data catalog and training sources')
    if config.max_steps < state['report']['steps'] or config.epochs < state['completed_epochs']:
        raise ValueError('Resume cap precedes the checkpoint')
    return state


def train(catalog,output,config,tracking=None,resume=None):
    started=time.monotonic();deadline=started+config.seconds
    device=cuda_setup(config.seed)
    catalog,output=Path(catalog).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Output exists; choose a new directory')
    spec=WindowSpec(2,config.future)
    training=VisualDynamicsDataset(catalog,'train',spec)
    validation=VisualDynamicsDataset(catalog,'validation',spec)
    if not len(training) or not len(validation):raise ValueError('Training/validation need nonempty windows')
    stats=fit_motion_stats(training)
    model=VisualDynamics(stats).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.0001)
    paths=['tdyssey/world_model_data.py','tdyssey/world_model_protocol.py','tdyssey/world_model_training.py',
           'tdyssey/models/visual_dynamics.py','tdyssey/training_data.py','tdyssey/dataset.py','scripts/train_world_model.py',
           'tdyssey/tracking.py','pyproject.toml','uv.lock']
    report={'schema_version':1,'kind':'visual_world_model_pilot_not_policy_adaptation','status':'running',
            'config':asdict(config),'model_version':MODEL_VERSION,'parameters':sum(p.numel() for p in model.parameters()),
            'catalog_sha256':digest(catalog.read_bytes()),'normalization':stats,'loss_weights':LOSS_WEIGHTS,
            'device':str(device),'gpu':torch.cuda.get_device_name(),'torch_version':str(torch.__version__),
            'provenance':{'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                          'dirty':bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True)),
                          'source_sha256':{p:digest((ROOT/p).read_bytes()) for p in paths}},
            'train':training.summary(),'validation':validation.summary(),'model_test_used':False,
            'selection':'lowest validation contact object-position error at final horizon; all-step fallback if no contact',
            'steps':0,'training_windows_seen':0,'forecast_steps_trained':0,'epochs':[]}
    best=None;completed=0;cursor=0;epoch_losses=[]
    if resume:
        payload=torch.load(resume,map_location='cpu',weights_only=True)
        state=check_training_resume(payload,config,report['catalog_sha256'],report['provenance']['source_sha256'])
        if payload['normalization']!=stats:
            raise ValueError('Training-only normalization differs from checkpoint')
        model.load_state_dict(payload['state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        report=state['report'];report.update(status='running',config=asdict(config))
        report.pop('error',None)
        completed,cursor=state['completed_epochs'],state['epoch_cursor']
        epoch_losses=state['epoch_losses'];best=state['best_score']
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
        # Charge completed work recorded after the recovery point separately.
        sidecar=Path(resume).parent/'report.json'
        audit={'checkpoint_sha256':digest(Path(resume).read_bytes()),'post_boundary_costs_known':False}
        if sidecar.is_file():
            latest=json.loads(sidecar.read_text())
            if latest.get('last_checkpoint_sha256')==audit['checkpoint_sha256']:
                audit['discarded_optimizer_steps']=max(0,latest['steps']-report['steps'])
                audit['discarded_training_windows']=max(0,latest['training_windows_seen']-report['training_windows_seen'])
                audit['post_boundary_costs_known']=latest.get('status')=='complete'
        report.setdefault('resume_chain',[]).append(audit)
    output.mkdir(parents=True,exist_ok=False)
    if resume and best is not None:
        best_path=Path(resume).parent/'best.pt'
        if not best_path.is_file() or digest(best_path.read_bytes())!=report['best_checkpoint_sha256']:
            raise ValueError('Copy the matching best.pt beside the recovery checkpoint')
        shutil.copyfile(best_path,output/'best.pt')
    logger=RunLogger(output,tracking,{'trainer':'visual_world_model','config':asdict(config),
        'catalog_sha256':report['catalog_sha256'],'provenance':report['provenance']})
    report['tracking']=logger.state
    session={'start_steps':report['steps'],'status':'running'}
    report.setdefault('sessions',[]).append(session)
    def recovery():
        session.update(elapsed_seconds=time.monotonic()-started,status=report['status'])
        snapshot=json.loads(json.dumps(report))
        save_checkpoint(output/'last.pt',model,stats,config,report['catalog_sha256'],completed,
            {'optimizer':optimizer.state_dict(),'report':snapshot,'completed_epochs':completed,
             'epoch_cursor':cursor,'epoch_losses':epoch_losses,'best_score':best,
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device)})
        report['last_checkpoint_sha256']=digest((output/'last.pt').read_bytes())
    def log_validation(metrics):
        values={}
        for h,groups in metrics['horizons'].items():
            for name,row in groups.items():
                values.update({f'validation/h{h}/{name}/{k}':v for k,v in row.items() if v is not None})
        logger.log(values,report['steps'])
    try:
        if not resume:
            report['initial_validation']=evaluate(model,validation,stats,config.batch_size,deadline)
            log_validation(report['initial_validation'])
        recovery()
        save_json(output/'report.json',report,config.max_bytes)
        stop=False
        for epoch in range(completed+1,config.epochs+1):
            if stop:break
            model.train();losses=list(epoch_losses)
            order=list(episode_order(training.offsets,config.seed+epoch))
            loader=DataLoader(training,batch_size=config.batch_size,
                              sampler=order[cursor:],num_workers=0,
                              generator=torch.Generator().manual_seed(config.seed+epoch))
            for batch in loader:
                if time.monotonic()>=deadline:raise TimeoutError('World-model wall-time limit reached')
                if report['steps']>=config.max_steps:
                    stop=True;break
                batch=to_device(batch,device)
                optimizer.zero_grad(set_to_none=True)
                prediction=model(batch['observations'],batch['actions'])
                loss,components=training_loss(model,prediction,batch)
                if not torch.isfinite(loss):raise ValueError('Non-finite world-model loss')
                loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(gradient):raise ValueError('Non-finite model gradient')
                optimizer.step()
                report['steps']+=1;report['training_windows_seen']+=len(batch['actions'])
                report['forecast_steps_trained']+=len(batch['actions'])*config.future
                losses.append(float(loss.detach()))
                epoch_losses=losses
                cursor+=len(batch['actions'])
                if report['steps'] % logger.config.log_every == 0:
                    logger.log({'train/loss':float(loss.detach()),'train/gradient_norm':float(gradient),
                        **{f'train/{k}':float(v.detach()) for k,v in components.items()},
                        'runtime/peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                        'progress/training_windows_seen':report['training_windows_seen']},report['steps'])
                if report['steps'] % config.checkpoint_every == 0:
                    recovery()
                    save_json(output/'report.json',report,config.max_bytes)
            if not losses:break
            metrics=evaluate(model,validation,stats,config.batch_size,deadline)
            final=metrics['horizons'][str(config.future)]
            chosen=final['contact'] if final['contact']['forecast_steps'] else final['all']
            score=chosen['object_position_error_pixels']
            epoch_complete=cursor>=len(order)
            row={'epoch':epoch,'steps':report['steps'],'train_loss':sum(losses)/len(losses),
                 'epoch_complete':epoch_complete,'windows_in_epoch':cursor,
                 'validation':metrics,'selection_score':score}
            report['epochs'].append(row)
            log_validation(metrics)
            logger.log({'train/epoch_mean_loss':row['train_loss'],'progress/epoch':epoch},report['steps'])
            if best is None or score<best:
                best=score;report['best_epoch']=epoch
                save_checkpoint(output/'best.pt',model,stats,config,report['catalog_sha256'],epoch)
                report['best_checkpoint_sha256']=digest((output/'best.pt').read_bytes())
            if epoch_complete:
                completed=epoch;cursor=0;epoch_losses=[]
            recovery()
            save_json(output/'report.json',report,config.max_bytes)
            print(json.dumps({'epoch':epoch,'steps':report['steps'],'loss':row['train_loss'],'validation_contact_position_error_px':score}),flush=True)
        if best is None:raise ValueError('No trained epoch completed validation')
        model,payload=load_checkpoint(output/'best.pt',device)
        report['best_validation']=evaluate(model,validation,stats,config.batch_size,deadline)
        report['preview']=save_preview(model,validation,output/'prediction-preview.png')
        report.update(status='complete',stop_reason='max_steps' if stop else 'epochs',
                      elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      checkpoint_sha256=digest((output/'best.pt').read_bytes()))
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        session.update(elapsed_seconds=time.monotonic()-started,status=report['status'],
                       optimizer_steps=report['steps']-session['start_steps'])
        report['counts_are_lower_bounds']=report['status']!='complete'
        logger.finish(report['status'])
        save_json(output/'report.json',report,config.max_bytes)
    return report
