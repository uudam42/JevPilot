from pathlib import Path

from jevpilot import (
    InMemoryTraceSink,
    JsonlTraceSink,
    TraceEventType,
    Tracer,
    WorkflowState,
    read_jsonl_trace,
)


def test_tracer_sequences_and_fans_out(tmp_path: Path, state: WorkflowState) -> None:
    mem = InMemoryTraceSink()
    path = tmp_path / "trace.jsonl"
    t = Tracer([mem, JsonlTraceSink(path)])
    t.emit("wf", 0, TraceEventType.WORKFLOW_STARTED, goal=state.goal)
    t.emit("wf", 0, TraceEventType.STATE_SNAPSHOT, state=state)
    assert [e.seq for e in mem.events] == [0, 1]
    assert mem.events[0].payload["goal"]["description"] == "test goal"
    assert read_jsonl_trace(path) == mem.events
