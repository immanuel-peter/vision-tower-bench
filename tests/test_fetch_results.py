from scripts import fetch_results


def test_fetch_always_reconciles_with_the_hub_and_marks_only_full_results(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(fetch_results, "snapshot_download", lambda *a, **kw: calls.append(kw["allow_patterns"]) or [])
    monkeypatch.setattr(fetch_results, "MARKER", tmp_path / "payload-revision.txt")
    fetch_results.fetch(["results/recognition/"])
    assert calls == [["results/recognition/*"]] and not fetch_results.fetched()
    fetch_results.fetch()
    assert fetch_results.fetched()
    fetch_results.fetch(["results/"])
    assert calls[1:] == [["results/*"], ["measurements/*"], ["results/*"]]
