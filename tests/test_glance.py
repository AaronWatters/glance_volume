from glance_volume import glance


def test_main_builds_component_and_runs(monkeypatch):
    seen = {"volume_paths": [], "component": None, "run_called": False}

    class FakeVolume:
        def __init__(self, path):
            seen["volume_paths"].append(path)
            self.path = path

    class FakeComponent:
        def __init__(self, volumes, segmentation=None):
            seen["component"] = {
                "volumes": volumes,
                "segmentation": segmentation,
            }

        def run(self):
            seen["run_called"] = True

    monkeypatch.setattr(glance, "Volume", FakeVolume)
    monkeypatch.setattr(glance, "NeuroglancerComponent", FakeComponent)

    rc = glance.main(["im_a.tiff", "im_b.tiff", "-s", "seg.tiff"])

    assert rc == 0
    assert seen["volume_paths"] == ["im_a.tiff", "im_b.tiff", "seg.tiff"]
    assert seen["component"] is not None
    assert [v.path for v in seen["component"]["volumes"]] == ["im_a.tiff", "im_b.tiff"]
    assert seen["component"]["segmentation"].path == "seg.tiff"
    assert seen["run_called"]
