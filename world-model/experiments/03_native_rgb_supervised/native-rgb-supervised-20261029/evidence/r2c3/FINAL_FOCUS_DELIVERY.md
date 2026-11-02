# Additional direct human focus-gallery request

The parent relayed the human's explicit request: after6000 training,show the
same letters next to1000/3000 and inspect whether O/C corners recover. Current
training continues unchanged; no additional old-point inference or scoring.

Fixed val_motion windows,never swap to prettier samples:
- O: R2B-new-25-O-left-bottom:12
- B: R2B-new-48-B-top:5
- C: R2B-new-28-C-left:6
- X: R2B-new-38-X-right:13

H1(.1s) andH4(.4s) separate main images,strict four columns:
real future / training1000updates / training3000updates / training6000updates.
All predictions use3 native denoising steps with the frozen paired seeds,
actions and window.16steps remains in the separate scientific matrix.

Additional large O/C detail comparisons,prefer2rows×4columns. Each row uses
identical coordinate crop and integer nearest-neighbour magnification across
all four columns. No rotation/translation alignment,sharpening,smoothing,shape
repair or clipping away a predicted deformation. Show outer straight edges,
turns,O inner hole,C opening and both stroke ends. Preserve all original arrays;
only display-encoding copies,clearly labelled for any range saturation.

Reuse parent-view-1000/3000 rendering/provenance approach and retain old images.
New images count toward400MiB/shared3GiB. No duplicate large arrays. Record raw
SHA/window/step/seed/crop/magnification and link image-page+absolute paths in
the actual final callback. Full36-window report remains;4focus examples are
visual diagnostics,not aggregate statistical samples.

Describe O and C individually as recovered,partly recovered or unrecovered from
actual viewed pixels. IoU or no-triggered-failure is not a substitute for corners.
Limit rasterization/occlusion uncertainty to the specific affected region;
unknown cannot conceal visible whole-object deformation.

Parent also provided cold parent-view-1000(1,439,831B) andparent-view-3000
(1,550,126B),with no model/physics/scoring.1000's24unknown are a1px gray-mask
dilation/blue-mask intersection proxy,all object bordersfalse,not proof that24
are actually occluded. B holes/outer corners andX branching improve,while O/C
remain problematic at3000. Do not change frozen scoring mid-training.
