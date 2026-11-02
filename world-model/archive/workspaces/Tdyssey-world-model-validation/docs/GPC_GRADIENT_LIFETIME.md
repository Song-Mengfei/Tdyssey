# Release the temporary gradient-check list

After [pilot02](GPC_SUPERVISED_PILOT_02.md), the sole production change releases
local `grads` immediately after finite-gradient validation. Parameter `p.grad`
remains available for the original clipping and optimizer step. At the next
`zero_grad(set_to_none=True)`, the old list no longer keeps those gradients alive.
Loss, norm, optimizer, precision, architecture, configuration and budgets do not
change. No cache clearing, garbage collection or CPU transfer is introduced.

Twelve focused stdlib/marker checks passed in one6.062s command, within the
60s-command/180s-total/three-command limits. They invoke the actual old production
function from sealed `3974e84` source and the repaired production function for
two rounds. Old gradients remain alive at the second forward; repaired gradients
are released. Both runs produce parameters `[3.88,-2.16,7]`, identical metrics
and operation traces, and correct gradient-object identities at clip/step.
Missing/nonfinite gradient and nonfinite updated-parameter rejection, failed
spent reservations,500-entry consumption and checkpoint cursor recovery pass.
Only the one deletion differs in the function AST.

Real Torch/NumPy/native/process/network/model/physics imports or operations were
blocked with zero attempts. Evidence is under
`artifacts/gpc-gradient-lifetime-review-20261026/`, including the sealed old source,
`validation-01.json` and its log. Temporary review files have a4MiB allowance
inside the16MiB total evidence cap. No production attempt, model ledger, new
candidate or configuration was created. Prior results and claims remain spent.
Closing checks preserve all7,867 old files and14 claim documents, including both
configs, the500list and weights. Eight held files use manifest/stat checks only,
with zero content opens/hashes; the compact reconciliation passes.

The reference-lifetime issue is repaired; the real CUDA peak has not been
retested. This establishes neither the size of any GPU saving nor training
feasibility under512MiB. Return for main review before any real follow-up.
