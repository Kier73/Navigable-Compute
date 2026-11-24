"""Use the actual pinned IFS operation-span recorder and profile verifier."""
import json
from pathlib import Path
import sys
from operation_spans import (OperationSpanRecorder,build_operation_cost_profiles,
                             validate_operation_cost_profiles)
from reflective_routing_demo import campaign


def main():
    recorder=OperationSpanRecorder(clock_actor_id="runner",clock_segment_id="one")
    report=campaign(recorder)
    spans=list(recorder.spans)
    profiles=build_operation_cost_profiles(spans,minimum_samples=3)
    assert validate_operation_cost_profiles(profiles,spans)==profiles
    assert not recorder.measurement_errors
    assert len(spans)==132
    assert all(p["readiness"]=="descriptive_only" for p in profiles["profiles"])
    out=Path("route-observation-artifacts")
    out.mkdir(exist_ok=True)
    (out/"spans.json").write_text(json.dumps(spans,sort_keys=True),encoding="utf-8")
    (out/"profiles.json").write_text(json.dumps(profiles,sort_keys=True),encoding="utf-8")
    (out/"campaign.json").write_text(json.dumps(report,sort_keys=True),encoding="utf-8")
    print("IFS profile reconstruction:",len(spans),"spans",len(profiles["profiles"]),"profiles")


if __name__=="__main__":
    main()
