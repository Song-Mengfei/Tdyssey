"""Identity stays exact in JSON; every basename uses a Windows-safe hash key."""
import hashlib,json,re

def panel_name(sample_id,mode,label='all',extension='png'):
    if extension not in ('png','json','npz','pt','txt','html'):raise ValueError('Unsupported output suffix')
    identity=dict(sample_id=str(sample_id),mode=str(mode),label=str(label),extension=extension)
    encoded=json.dumps(identity,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    digest=hashlib.sha256(encoded).hexdigest()
    slug=re.sub('[^A-Za-z0-9_-]+','_',str(sample_id)).strip('_')[:28] or 'case'
    return 'p-'+slug+'-'+digest[:24]+'.'+extension,dict(identity=identity,identity_sha256=digest)

def reserve_panel(io,mapping,sample_id,mode,label='all',extension='png'):
    name,record=panel_name(sample_id,mode,label,extension)
    if name in mapping and mapping[name]!=record:raise RuntimeError('Basename collision; no overwrite')
    mapping[name]=record
    io.write_json('report/basename-map.json',mapping)
    return io.checked('report/'+name)

