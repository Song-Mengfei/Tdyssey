# Explicit human approval required after this stage

Received2026-10-30 through the planning chat,quoting the human:

> 他再回传给你报告之后你不要自动下发了，先给我审批再说。

This overrides the earlier convention that the planning chat could autonomously
release successive implementation stages. The already released CONTACT-PILOT2
scope and its failure-stop conditions remain active through its current report.
After the next actual stage callback(including any early-failure report),stop and
wait for explicit human approval. Do not start a new repair/retest/collection,
resume R3B or expand data automatically. Sending a report is not authorization
to continue. The planning chat may read/review and organize results/recommendations
for the human,but cannot dispatch implementation before the human approves it.

The standing authorization to proactively send stage reports,experiment summaries
and local paths to the planning chat remains valid. Save the actual callback
receipt as usual;the human does not need to forward the report manually. No
separate acknowledgment message to the planning chat is needed for this rule.
