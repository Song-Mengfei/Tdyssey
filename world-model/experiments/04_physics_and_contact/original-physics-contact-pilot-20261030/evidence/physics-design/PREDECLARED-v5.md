# Candidate5 before one remaining-corner replay

Root v4 sweep has one known remaining C-corner_tr failure (rawv2): after kinematic velocity was already stopped, a constrained native impulse generates actual dynamic twist(-944,+458,13.6rad/s); the next.001s integration moves a side into the static pusher by1.65px. Same face/wall constraints cannot undo this already committed dynamic integration.

V5 keeps geometry/body/damping/PD/solver/epsilon/valid1world and command constraints unchanged. Quantize integration to integer.000125s ticks, with exactly80ticks per.01s PD interval. Step sizes only1/2/4/8ticks (.000125/.00025/.0005/.001s), chosen for target max.25world boundary-point travel using actual body speed+omega radius and requested pusher speed. Minimum tick is explicit; target may be exceeded for a larger shock, which must still pass unchanged actual geometry validity checks. dt growth at most2x across steps prevents tiny-remainder warm-start amplification. No fractional remainders or cache reset. New protocolv5; actualdt/substep count variable and recorded.

Run only the root frozen C-corner_tr24-control initial/action sequence once now, preserve all raw/cost/RGB. No tolerance widening and no extra broad sweep by this agent. Root will decide next full frozen96-case gate. No models/GPU/training.
