"""Process page alarms (pure rules over the status readings)."""

from zettlecast.system import alarms

LLM_UP = {"backend": "llama-server", "server": {"installed": True, "state": "ready"}}
LLM_DOWN = {"backend": "llama-server", "server": {"installed": True, "state": "stopped"}}


def queue(**by_status):
    return {"podcast": {"by_status": {"pending": 0, "processing": 0, "failed": 0, **by_status}}}


def ids(found):
    return {(a["id"], a["priority"]) for a in found}


def test_healthy_plant_has_no_alarms():
    run = {"phase": "enhance", "alive": True, "stage": "cleanup"}
    gpu = {"memory_used_mib": 28_300, "memory_total_mib": 32_607, "temperature": 62}
    assert alarms(run, gpu, LLM_UP, queue(pending=180, processing=1), 600) == []


def test_vram_near_and_past_the_spill_point():
    run = {"phase": "idle", "alive": False}
    assert ("vram", "medium") in ids(alarms(run, {"memory_used_mib": 31_000, "memory_total_mib": 32_607}, LLM_UP, queue(), 600))
    assert ("vram", "high") in ids(alarms(run, {"memory_used_mib": 32_000, "memory_total_mib": 32_607}, LLM_UP, queue(), 600))


def test_dead_run_stuck_items_failed_and_llm_down():
    dead = {"phase": "enhance", "alive": False, "episode_title": "Ep"}
    found = ids(alarms(dead, None, LLM_UP, queue(processing=2, failed=1), 600))
    assert {("run-died", "high"), ("stuck", "medium"), ("failed", "medium")} <= found
    live = {"phase": "enhance", "alive": True, "stage": "suspects"}
    assert ("llm-down", "high") in ids(alarms(live, None, LLM_DOWN, queue(), 600))
    # between LLM steps (saving, indexing) a stopped server is fine
    assert ("llm-down", "high") not in ids(alarms({**live, "stage": "index"}, None, LLM_DOWN, queue(), 600))


def test_low_disk_and_stop_request():
    found = ids(alarms({"phase": "enhance", "alive": True, "stop_requested": True}, None, LLM_UP, queue(), 3))
    assert {("disk", "high"), ("stopping", "low")} <= found


def test_ingests_are_tracked_while_they_run():
    from zettlecast import system

    system.web_run.update(in_flight=[], recent=[])
    with system.tracking_ingest("web", "example.com") as item:
        assert [i["title"] for i in system.web_run["in_flight"]] == ["example.com"]
        item["status"] = "success"
    assert system.web_run["in_flight"] == []
    assert [(r["kind"], r["status"]) for r in system.web_run["recent"]] == [("web", "success")]
    try:
        with system.tracking_ingest("pdf", "paper.pdf"):
            raise ValueError("unreadable")
    except ValueError:
        pass
    assert system.web_run["in_flight"] == [] and system.web_run["recent"][-1]["status"] == "failed"
