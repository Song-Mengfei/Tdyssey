"""Fixed identity for the separately authorized third comparison attempt.

This is one new non-resumable account, not a selectable attempt/retry mechanism.
The spent v1/v2 claims and scientific protocol remain unchanged.
"""
from pathlib import Path

ATTEMPT_ID = "wm-input-compat-eval-v3"
REGISTRY = ATTEMPT_ID+"-registry"
OUTPUT = ATTEMPT_ID+"-20261025"
LEDGER = ATTEMPT_ID+"-ledger-20261025"
AUTHORIZATION_ID = ATTEMPT_ID+"-20261025-user-authorized"


def validate_execution_identity(repo, output, ledger, authorization_id):
    from .wm_compat_storage import unlinked_path
    artifacts = Path(repo)/"artifacts"
    if (unlinked_path(output) != artifacts/OUTPUT or ledger is None
            or unlinked_path(ledger) != artifacts/LEDGER or authorization_id != AUTHORIZATION_ID):
        raise ValueError("Execution requires the fixed v3 output, ledger and authorization identity")
