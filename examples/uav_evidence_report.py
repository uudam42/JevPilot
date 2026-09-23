"""UAV materials iteration 4: why can no existing material be selected?

python examples/uav_evidence_report.py

Same synthetic target and real 46-material dataset as uav_material_search.py.
Separates engineering failures (a hard constraint violated by sufficient
evidence) from evidence gaps (a hard constraint that cannot be decided), and
ranks the missing evidence by how many potentially viable candidates it blocks.
"""

from __future__ import annotations

from uav_material_search import target  # sibling example (examples/ is on sys.path)

from domains.uav_materials.datasets import real_candidates
from domains.uav_materials.evidence import GapClass, evidence_report, format_report


def main() -> None:
    report = evidence_report(target(), real_candidates())
    viable = [
        a.candidate_id for a in report.assessments if a.classification is not GapClass.INFEASIBLE
    ]
    print(format_report(report, detail_ids=viable))


if __name__ == "__main__":
    main()
