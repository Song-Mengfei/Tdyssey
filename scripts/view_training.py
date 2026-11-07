"""Serve local live training charts; no account, downloads or extra packages."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Tdyssey live training</title><style>
body{font:16px system-ui;background:#101723;color:#e2e8f0;margin:32px auto;max-width:1200px;padding:0 18px}
h1{font-size:25px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
section{background:#1c2838;border:1px solid #34445a;border-radius:12px;padding:18px}
select{max-width:100%;background:#101723;color:#e2e8f0;padding:8px;border:1px solid #53657c}
svg{width:100%;height:210px}p{color:#abbcd1}small{color:#a8b6c9} #status{white-space:pre-wrap}
</style><h1>Tdyssey live training</h1><p id="status">Loading local events…</p>
<p>Fixed noise loss is a diagnostic. Check per-letter success and final coverage for actual pushing improvement.</p>
<div class="grid" id="charts"></div><p>Refreshes every 5 seconds. X axis: optimizer minibatches. No smoothing.</p>
<script>
const wanted=['train/update_mean_loss','probe/L/fixed_noise_mse','eval/L/final_coverage','rollout/L/contact_fraction'];
const choices=[], holders=[];
for(let i=0;i<4;i++){const section=document.createElement('section');const select=document.createElement('select');
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 500 210');
 const label=document.createElement('small');section.append(select,svg,label);document.querySelector('#charts').append(section);
 choices.push(select);holders.push([svg,label]);select.onchange=()=>draw();}
let events=[];
function draw(){choices.forEach((select,i)=>{const key=select.value;const points=events.filter(e=>typeof e.metrics[key]==='number')
 .map(e=>[e.metrics['progress/optimizer_step'],e.metrics[key]]);const [svg,label]=holders[i];svg.replaceChildren();
 if(!points.length){label.textContent='Waiting for this metric';return;}
 let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const poly=document.createElementNS(svg.namespaceURI,'polyline');poly.setAttribute('fill','none');poly.setAttribute('stroke','#58c6ff');
 poly.setAttribute('stroke-width','2');poly.setAttribute('points',points.map(([x,y])=>`${25+450*(x-xmin)/(xmax-xmin||1)},${185-160*(y-ymin)/(ymax-ymin||1)}`).join(' '));svg.append(poly);
 const dot=document.createElementNS(svg.namespaceURI,'circle');const [x,y]=points.at(-1);
 dot.setAttribute('cx',25+450*(x-xmin)/(xmax-xmin||1));dot.setAttribute('cy',185-160*(y-ymin)/(ymax-ymin||1));
 dot.setAttribute('r','4');dot.setAttribute('fill','#a4e2ff');svg.append(dot);
 label.textContent=`Latest ${ys.at(-1).toPrecision(5)} · range ${ymin.toPrecision(4)}–${ymax.toPrecision(4)} · steps ${xmin}–${xmax}`;
 });}
async function poll(){try{const response=await fetch('/events',{cache:'no-store'});const data=await response.json();events=data.events;
 const keys=[...new Set(events.flatMap(e=>Object.keys(e.metrics)))].filter(k=>k!=='progress/optimizer_step').sort();
 choices.forEach((select,i)=>{const previous=select.value; if([...select.options].map(o=>o.value).join('|')!==keys.join('|')){
 select.replaceChildren(...keys.map(k=>{const o=document.createElement('option');o.textContent=k;o.value=k;return o;}));
 select.value=keys.includes(previous)?previous:keys.includes(wanted[i])?wanted[i]:keys[i%Math.max(1,keys.length)];}});
 document.querySelector('#status').textContent=`${data.name} · ${data.status} · ${events.length} recent events · ${new Date().toLocaleTimeString()}`;draw();
 }catch(error){document.querySelector('#status').textContent='Waiting for metrics.jsonl; '+error.message;}}
poll();setInterval(poll,5000);
</script></html>'''


def read_events(run, limit=5000):
    rows = []
    path = run/'metrics.jsonl'
    if path.exists():
        # Producer has a 16 MiB default cap. Tolerate a concurrently written tail.
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if 'metrics' in row:
                        rows.append(row)
                except json.JSONDecodeError:
                    continue
    status = 'waiting'
    try:
        status = json.loads((run/'tracking.json').read_text())['status']
    except (OSError, ValueError, KeyError):
        pass
    return {'name': run.name, 'status': status, 'events': rows[-limit:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    run = args.run.resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                payload, kind = PAGE.encode(), 'text/html; charset=utf-8'
            elif self.path == '/events':
                payload, kind = json.dumps(read_events(run), allow_nan=False).encode(), 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Live charts: http://{args.host}:{server.server_port} (run: {run})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
