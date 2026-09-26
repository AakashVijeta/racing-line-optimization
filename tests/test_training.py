import pytest

from sac_train import GeneralizationCallback, lap_score


def make_callback(proc_frac=0.5, adaptive=True, real=("a", "b"), proc=("proc-1-0", "proc-1-1")):
    return GeneralizationCallback(val_pool={}, widths={}, obs_version=3, run_dir=".",
                                  real_train_ids=real, proc_train_ids=proc,
                                  proc_frac=proc_frac, adaptive=adaptive)


def test_lap_score_orders_outcomes():
    fast = lap_score(True, 60.0, 5000.0, 1.0)
    slow = lap_score(True, 90.0, 5000.0, 1.0)
    almost = lap_score(False, 50.0, 5000.0, 0.99)
    early = lap_score(False, 5.0, 5000.0, 0.1)
    assert fast > slow > almost > early >= 0.0


def test_track_weights_respect_procedural_share():
    cb = make_callback(proc_frac=0.3)
    w = cb.track_weights()
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["proc-1-0"] + w["proc-1-1"] == pytest.approx(0.3)


def test_failing_tracks_get_sampled_more():
    cb = make_callback()
    cb.fail_rate["a"] = 1.0
    cb.fail_rate["b"] = 0.0
    w = cb.track_weights()
    assert w["a"] == pytest.approx(3 * w["b"])  # (0.5 + 1.0) vs (0.5 + 0.0)

    uniform = make_callback(adaptive=False)
    uniform.fail_rate["a"] = 1.0
    w = uniform.track_weights()
    assert w["a"] == pytest.approx(w["b"])


def test_weights_without_procedural_tracks():
    cb = make_callback(proc=())
    assert sum(cb.track_weights().values()) == pytest.approx(1.0)


def test_entropy_coefficient_is_capped():
    import math
    import torch

    class FakeModel:
        log_ent_coef = torch.tensor([math.log(0.5)])

    cb = GeneralizationCallback(val_pool={}, widths={}, obs_version=3, run_dir=".",
                                real_train_ids=["a"], proc_train_ids=[], proc_frac=0.0,
                                max_ent_coef=0.02)
    cb.model = FakeModel()
    cb._clamp_entropy()
    assert torch.exp(cb.model.log_ent_coef).item() == pytest.approx(0.02)

    FakeModel.log_ent_coef = torch.tensor([math.log(0.01)])
    cb._clamp_entropy()
    assert torch.exp(cb.model.log_ent_coef).item() == pytest.approx(0.01)  # below the cap: untouched


def test_latest_checkpoint_picks_highest_step(tmp_path):
    from sac_train import latest_checkpoint
    assert latest_checkpoint(str(tmp_path)) is None
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    for name in ["sac_racing_50000_steps.zip", "sac_racing_1000000_steps.zip",
                 "sac_racing_950000_steps.zip", "notes.txt"]:
        (ckpt / name).write_bytes(b"")
    path, steps = latest_checkpoint(str(tmp_path))
    assert steps == 1_000_000  # numeric, not lexicographic, order
    assert path.endswith("sac_racing_1000000_steps.zip")


def resume(cli, saved):
    from sac_train import build_parser, merge_resume_config
    parser = build_parser()
    args = parser.parse_args(["--run", "r", "--resume"] + cli)
    return args, merge_resume_config(args, saved, parser)


def test_resume_restores_saved_training_settings():
    saved = {"run": "old", "timesteps": 8_000_000, "proc_frac": 0.7, "lr": 1e-4,
             "resume": False, "refill": 100_000, "git_sha": "abc123"}
    args, warnings = resume([], saved)
    assert args.proc_frac == 0.7 and args.lr == 1e-4 and args.timesteps == 8_000_000
    assert args.run == "r" and args.resume is True
    assert warnings == []


def test_resume_keeps_launch_options_and_allows_extending():
    saved = {"timesteps": 8_000_000, "refill": 100_000, "resume": False}
    args, warnings = resume(["--refill", "5000", "--timesteps", "10000000"], saved)
    assert args.refill == 5000
    assert args.timesteps == 10_000_000
    assert warnings == []


def test_resume_warns_about_overridden_flags():
    args, warnings = resume(["--proc-frac", "0.9"], {"proc_frac": 0.5})
    assert args.proc_frac == 0.5
    assert len(warnings) == 1 and "--proc-frac" in warnings[0]
