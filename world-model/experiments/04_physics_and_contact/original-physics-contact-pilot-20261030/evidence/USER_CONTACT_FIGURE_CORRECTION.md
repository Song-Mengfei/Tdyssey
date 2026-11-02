# User correction: do not claim contact from separated frames

Human: “你这右侧的图片明显不对吧 这不是根本就没碰到吗”. The attached image is the saved B-v6-before-after-control32.png.

The user is correct: new-protocol controls31/32 in that prefix have separated pusher/object positions. Same initial state/request prefix has produced different trajectories. The image is retained as a trajectory-divergence example and is explicitly excluded from proof of collision while in contact. No formal or calibration collection had begun at the correction.

Replacement proof: physics-design/contact-proof/B-TL-exact-contact-proof.png and exact-frame-proof.json. At exactlyt=.320/.330s the v6 saved microsteps have positive post-solve AO impulses and simultaneous OW contact. Geometry confirms AO overlap .00861159/.00896069worldpx and outside pusher center. A whole-control contact duration is not evidence for every frame; the separately saved.340s sample without AO remains reported.

Independent continuous-motion proof: B-free-sustained-contact-motion.png. The(.30,.60] interval has600/600 actual positive-AO microsteps, .3000/.3000seconds contact, actual H1 centroid motion around2.1–3.4imagepx. No new physics/model calls were made to render saved states.

The 96-case physics gate and UI gate use actual geometry/events, not the rejected picture. Wall-normal saturation is not counted as effective pushing unless actual object motion/contact is measured. Current7499 is still old-physics-trained; no model improvement or training is claimed.
