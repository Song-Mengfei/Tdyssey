"""Render the four predeclared snapshots without controls or search."""

import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
os.environ.setdefault("SDL_AUDIODRIVER","dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")

import cv2
import numpy as np
from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.official import load_official_source
from tdyssey.state import Snapshot


def main():
    base=Path(__file__).resolve().parent
    inventory=json.loads((base/"scene_inventory.json").read_text(encoding="utf-8"))
    original=make_official_shape_env_class(load_official_source(ROOT/"third_party/diffusion_policy"))
    header,image_size,footer,gap,banner=56,512,52,12,48
    tile_height=header+image_size+footer
    canvas=np.full((banner+2*tile_height+3*gap,2*image_size+3*gap,3),248,dtype=np.uint8)
    cv2.putText(canvas,"D teacher validation: fixed initial scenes",(gap,30),
                cv2.FONT_HERSHEY_SIMPLEX,.75,(35,35,35),2,cv2.LINE_AA)
    cv2.putText(canvas,"RENDER ONLY",(815,30),cv2.FONT_HERSHEY_SIMPLEX,.58,(80,80,80),1,cv2.LINE_AA)
    counts={"preview_reset_physics_substeps":0,"snapshot_restores":0,
            "actual_controls":0,"search_evaluations":0,"forecast_controls":0}
    for index,scene in enumerate(inventory["scenes"]):
        state=Snapshot.from_dict(scene["initial_state"])
        env=original(shape="D",legacy=True,render_size=512)
        try:
            env.seed(scene["accepted_seed"])
            env.reset()
            counts["preview_reset_physics_substeps"]+=1
            env.restore_state(state)
            counts["snapshot_restores"]+=1
            frame=cv2.cvtColor(env.render(mode="rgb_array"),cv2.COLOR_RGB2BGR)
            if frame.shape!=(512,512,3):raise AssertionError(frame.shape)
            if env.capture_state()!=state:raise AssertionError("Rendering changed snapshot")
            x=gap+(index%2)*(image_size+gap)
            y=banner+gap+(index//2)*(tile_height+gap)
            canvas[y+header:y+header+image_size,x:x+image_size]=frame
            cv2.putText(canvas,f"Scene {scene['accepted_seed']}",(x,y+22),cv2.FONT_HERSHEY_SIMPLEX,
                        .65,(20,20,20),2,cv2.LINE_AA)
            cv2.putText(canvas,f"D angle {math.degrees(state.object_body.pose.angle):.1f} deg  ->  goal 45.0 deg",
                        (x,y+44),cv2.FONT_HERSHEY_SIMPLEX,.50,(40,40,40),1,cv2.LINE_AA)
            fy=y+header+image_size+20
            cv2.putText(canvas,f"Green goal: ({state.goal.position[0]:.1f}, {state.goal.position[1]:.1f})",
                        (x,fy),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,90,30),1,cv2.LINE_AA)
            cv2.putText(canvas,f"Blue pusher: ({state.pusher_body.pose.position[0]:.0f}, {state.pusher_body.pose.position[1]:.0f})"
                        f"  |  coverage {100*scene['metrics']['coverage']:.1f}%",
                        (x,fy+22),cv2.FONT_HERSHEY_SIMPLEX,.48,(130,65,15),1,cv2.LINE_AA)
        finally:
            env.close()
    path=base/"initial-scenes.png"
    if not cv2.imwrite(str(path),canvas):raise RuntimeError("PNG output failed")
    report={"kind":"initial_scene_render_preview_not_experiment",
            "inventory":"scene_inventory.json","output":path.name,
            "counts_separate_from_inventory_and_matrix":counts,
            "torch_loaded":"torch" in sys.modules,"image_shape":list(canvas.shape),
            "snapshot_states_unchanged":True}
    (base/"initial-scenes-preview.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
