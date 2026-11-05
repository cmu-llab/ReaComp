"""
Reward for the environment-setup domain (SetupBench, EnvBench-Python).

Follows the rewards/ registry contract
    reward(result, execution_ok, entry, **kwargs) -> {"value": float, "message": str}
with value 1.0 iff the task's hidden verifier passes.

``result`` may be
  * {"container": DockerTaskContainer}  - verify live in that container (SetupBench)
  * {"script": "<bash>"} or a bash string - replay the script in a fresh container, then verify
"""
from typing import Any, Dict


def reward(result: Any, execution_ok: bool, entry: Dict, **kwargs) -> Dict:
    if not execution_ok:
        return {"value": 0.0, "message": "Solution raised a runtime error during execution."}
    from domains.envsetup.verifier import verify
    container = result.get("container") if isinstance(result, dict) else None
    if container is not None:
        return verify(entry, container=container, **kwargs)
    script = result.get("script") if isinstance(result, dict) else result
    if not isinstance(script, str):
        return {"value": 0.0, "message": "Expected a bash script string or {'script': ...}."}
    return verify(entry, script=script, **kwargs)
