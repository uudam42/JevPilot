"""UAV materials, end to end: natural-language request → engineering report.

    python examples/uav_end_to_end.py                 # OFFLINE_DEMO, built-in request
    python examples/uav_end_to_end.py "your request"  # OFFLINE_DEMO, your request

Same as ``jevpilot uav-materials --demo``. Runs without credentials or network:
requirements are interpreted by the deterministic offline parser and the
workflow is routed by a scripted reference policy through the JevRouter
pipeline. For live Claude + Jev, use ``jevpilot uav-materials --live``.
"""

from __future__ import annotations

import sys

from apps.uav_materials import DEMO_REQUEST, run_uav_material_workflow


def main() -> None:
    request = " ".join(sys.argv[1:]) or DEMO_REQUEST
    result = run_uav_material_workflow(request)
    print(result.markdown)
    print(f"\n[{result.mode.value}] decision: {result.decision}; report: {result.report.status}")


if __name__ == "__main__":
    main()
