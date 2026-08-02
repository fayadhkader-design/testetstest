import threading

import pytest

from nnscope.control import Controls


def gate_in_background(controls, results, timeout=2.0):
    """Run gate() on another thread, recording when it returns."""
    done = threading.Event()

    def run():
        results.append(controls.gate(timeout=timeout))
        done.set()

    threading.Thread(target=run, daemon=True).start()
    return done


def test_unpaused_gate_returns_immediately():
    assert Controls().gate(timeout=0.5) is True


def test_gate_blocks_while_paused_and_releases_on_resume():
    controls = Controls()
    controls.pause()
    results: list = []

    done = gate_in_background(controls, results)
    assert not done.wait(0.15), "gate should still be blocked while paused"

    controls.resume()

    assert done.wait(1.0), "resume did not wake the training thread"
    assert results == [True]


def test_gate_times_out_when_never_resumed():
    controls = Controls()
    controls.pause()

    assert controls.gate(timeout=0.1) is False


def test_single_step_admits_exactly_one_pass():
    controls = Controls()
    controls.step()

    assert controls.gate(timeout=0.5) is True
    assert controls.gate(timeout=0.1) is False, "budget of one is spent"


def test_step_budget_accumulates():
    controls = Controls()
    controls.step(3)

    assert [controls.gate(timeout=0.5) for _ in range(3)] == [True] * 3
    assert controls.gate(timeout=0.1) is False


def test_step_while_running_also_pauses():
    controls = Controls()
    assert not controls.paused

    controls.step()

    assert controls.paused


def test_step_rejects_nonpositive_counts():
    with pytest.raises(ValueError):
        Controls().step(0)


def test_release_unblocks_a_waiting_thread():
    controls = Controls()
    controls.pause()
    results: list = []

    done = gate_in_background(controls, results)
    assert not done.wait(0.15)

    controls.release()

    assert done.wait(1.0), "release did not wake the training thread"
    assert results == [True]
    assert controls.released


def test_release_is_permanent():
    controls = Controls()
    controls.release()
    controls.pause()

    assert controls.gate(timeout=0.1) is True, "released runs can never re-block"


def test_resume_clears_a_pending_step_budget():
    controls = Controls()
    controls.step(5)
    controls.resume()

    assert controls.snapshot()["stepBudget"] == 0


def test_learning_rate_is_consumed_once():
    controls = Controls()
    assert controls.take_learning_rate() is None

    controls.set_learning_rate(0.003)

    assert controls.take_learning_rate() == pytest.approx(0.003)
    assert controls.take_learning_rate() is None


def test_shock_is_consumed_once():
    controls = Controls()
    assert controls.take_shock() is None

    controls.shock(0.25)

    assert controls.take_shock() == pytest.approx(0.25)
    assert controls.take_shock() is None


def test_negative_settings_are_rejected():
    controls = Controls()
    with pytest.raises(ValueError):
        controls.set_learning_rate(-1)
    with pytest.raises(ValueError):
        controls.shock(-0.1)


def test_snapshot_reports_pause_state():
    controls = Controls()
    assert controls.snapshot() == {"paused": False, "stepBudget": 0}

    controls.step(2)

    assert controls.snapshot() == {"paused": True, "stepBudget": 2}


def test_starts_paused_when_requested():
    assert Controls(paused=True).paused
