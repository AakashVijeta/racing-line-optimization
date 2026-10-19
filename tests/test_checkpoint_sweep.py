from checkpoint_sweep import pace_tracks

TRACKS = [f"t{i}" for i in range(10)]


def test_collapsed_checkpoint_does_not_empty_the_pace_comparison():
    laps = {
        "good": dict.fromkeys(TRACKS, 60.0),
        "close": {**dict.fromkeys(TRACKS, 61.0), "t9": None},  # 9/10 laps: still a contender
        "collapsed": dict.fromkeys(TRACKS),                    # finishes nothing
    }
    assert pace_tracks(laps, TRACKS) == TRACKS[:9]


def test_pace_is_compared_on_every_track_when_all_candidates_finish():
    laps = {"a": dict.fromkeys(TRACKS, 60.0), "b": dict.fromkeys(TRACKS, 62.0)}
    assert pace_tracks(laps, TRACKS) == TRACKS


def test_no_laps_at_all_gives_nothing_to_compare():
    assert pace_tracks({"a": dict.fromkeys(TRACKS)}, TRACKS) == []
