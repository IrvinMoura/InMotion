"""Spec 001 — SafetyManager: nenhum caminho deixa input pressionado."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.input.backend import MOUSE_BUTTONS, KeyName
from inmotion.safety import (
    SafetyAction,
    SafetyConfig,
    SafetyManager,
    SafetyState,
    TrackedInput,
    register_atexit,
)
from tests.fakes import FakeInputBackend, SimulatedReleaseError

if TYPE_CHECKING:
    from inmotion.pipeline import Pipeline


# Exceções próprias: NotImplementedError herda de RuntimeError, então usar RuntimeError
# aqui faria o stub "passar" pelo pytest.raises pelo motivo errado.
class BoomError(Exception):
    pass


class FakeFailSafeError(Exception):
    """Imita pydirectinput.FailSafeException sem importar o pydirectinput."""


def make_manager() -> tuple[SafetyManager, FakeInputBackend]:
    fake = FakeInputBackend()
    manager = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig())
    return manager, fake


def press_then_raise(manager: SafetyManager, key: KeyName, exc: BaseException) -> None:
    with manager.guard():
        manager.input.press(key)
        raise exc


# --- TrackedInput.release_all (CA7, CA4b) ---------------------------------------


def test_release_all_empties_pressed() -> None:
    fake = FakeInputBackend()
    tracked = TrackedInput(fake, MOUSE_BUTTONS)
    tracked.press("space")
    tracked.press("w")

    tracked.release_all()

    assert tracked.pressed == frozenset()
    assert fake.pressed == set()


def test_release_all_twice_is_idempotent() -> None:
    fake = FakeInputBackend()
    tracked = TrackedInput(fake, MOUSE_BUTTONS)
    tracked.press("space")
    tracked.release_all()
    fake.events.clear()

    tracked.release_all()

    assert tracked.pressed == frozenset()
    assert set(fake.released()) == MOUSE_BUTTONS


def test_release_all_continues_after_partial_failure() -> None:
    fake = FakeInputBackend(fail_on_release={"w"})
    tracked = TrackedInput(fake, MOUSE_BUTTONS)
    for key in ("space", "w", "shift"):
        tracked.press(key)

    with pytest.raises(SimulatedReleaseError, match="w"):
        tracked.release_all()

    assert {"space", "shift"} <= set(fake.released())
    assert fake.pressed == {"w"}
    assert tracked.pressed == frozenset({"w"})  # continua registrada: dá para tentar de novo


def test_press_ignored_when_disabled() -> None:
    fake = FakeInputBackend()
    tracked = TrackedInput(fake, MOUSE_BUTTONS)
    tracked.set_enabled(False)

    tracked.press("space")

    assert not tracked.enabled
    assert tracked.pressed == frozenset()
    assert fake.events == []


# --- guard() (CA4, CA5) ------------------------------------------------------------


def test_guard_releases_on_exception_and_reraises() -> None:
    manager, fake = make_manager()

    with pytest.raises(BoomError):
        press_then_raise(manager, "mouse_left", BoomError())

    assert fake.pressed == set()


def test_guard_releases_on_keyboard_interrupt() -> None:
    manager, fake = make_manager()

    with pytest.raises(KeyboardInterrupt):
        press_then_raise(manager, "space", KeyboardInterrupt())

    assert fake.pressed == set()


def test_guard_releases_on_failsafe_like_exception() -> None:
    manager, fake = make_manager()

    with pytest.raises(FakeFailSafeError):
        press_then_raise(manager, "mouse_right", FakeFailSafeError())

    assert fake.pressed == set()


def test_guard_releases_on_normal_exit() -> None:
    manager, fake = make_manager()

    with manager.guard():
        manager.input.press("w")

    assert fake.pressed == set()


def test_guard_keeps_original_exception_if_release_fails() -> None:
    fake = FakeInputBackend(fail_on_release={"mouse_middle"})
    manager = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig())

    with pytest.raises(BoomError):
        press_then_raise(manager, "space", BoomError())

    assert "space" not in fake.pressed


def test_register_atexit_callback_releases_all() -> None:
    manager, fake = make_manager()
    callbacks: list[Callable[[], None]] = []

    register_atexit(manager, register=callbacks.append)
    manager.input.press("shift")
    assert len(callbacks) == 1

    callbacks[0]()

    assert fake.pressed == set()


# --- Perda de tracking (CA1, CA2) --------------------------------------------------


def test_lost_tracking_for_timeout_releases_all() -> None:
    manager, fake = make_manager()
    manager.on_frame(0.0, tracking_ok=True)
    manager.input.press("mouse_left")

    manager.on_frame(0.1, tracking_ok=False)
    assert fake.pressed == {"mouse_left"}  # 0,1 s < T

    manager.on_frame(0.3, tracking_ok=False)  # 0,3 s ≥ T = 0,2 s
    assert fake.pressed == set()
    assert manager.state is SafetyState.LOST


def test_short_tracking_loss_keeps_keys() -> None:
    manager, fake = make_manager()
    manager.on_frame(0.0, tracking_ok=True)
    manager.input.press("space")

    manager.on_frame(0.05, tracking_ok=False)
    manager.on_frame(0.15, tracking_ok=False)
    manager.on_frame(0.18, tracking_ok=True)

    assert fake.pressed == {"space"}
    assert fake.released() == []
    assert manager.state is SafetyState.ACTIVE


def test_tracking_recovery_reenables_input() -> None:
    manager, fake = make_manager()
    manager.on_frame(0.0, tracking_ok=True)
    manager.on_frame(0.3, tracking_ok=False)

    manager.input.press("w")
    assert fake.pressed == set()  # LOST: press ignorado

    manager.on_frame(0.4, tracking_ok=True)
    manager.input.press("w")

    assert manager.state is SafetyState.ACTIVE
    assert fake.pressed == {"w"}


def test_loss_from_first_frame_counts_from_it() -> None:
    manager, fake = make_manager()
    manager.input.press("mouse_left")

    manager.on_frame(0.0, tracking_ok=False)
    manager.on_frame(0.1, tracking_ok=False)
    assert fake.pressed == {"mouse_left"}

    manager.on_frame(0.25, tracking_ok=False)
    assert fake.pressed == set()


@pytest.mark.parametrize("t", [0.5, 0.4])
def test_non_increasing_time_raises(t: float) -> None:
    manager, _ = make_manager()
    manager.on_frame(0.5, tracking_ok=True)

    with pytest.raises(ValueError, match="crescente"):
        manager.on_frame(t, tracking_ok=True)


@pytest.mark.parametrize("bad_t", [math.nan, math.inf, -math.inf])
def test_non_finite_time_raises(bad_t: float) -> None:
    """Com t = NaN toda comparação é falsa: a perda de tracking nunca dispararia."""
    manager, _ = make_manager()
    with pytest.raises(ValueError, match="finito"):
        manager.on_frame(bad_t, tracking_ok=False)

    manager.on_frame(0.0, tracking_ok=True)
    with pytest.raises(ValueError, match="finito"):
        manager.on_frame(bad_t, tracking_ok=False)


class CountingTrackedInput(TrackedInput):
    """Conta as chamadas de release_all() feitas pelo SafetyManager."""

    def __init__(self, backend: FakeInputBackend) -> None:
        super().__init__(backend, MOUSE_BUTTONS)
        self.release_all_calls = 0

    def release_all(self) -> None:
        self.release_all_calls += 1
        super().release_all()


def long_loss_runs(times: list[float], oks: list[bool], timeout: float) -> int:
    """Oráculo ingênuo de S2': trechos maximais sem pose com duração ≥ T.

    A duração de um trecho é medida a partir do último frame com pose antes dele (ou do
    primeiro frame da sequência, se ainda não houve pose) até o último frame do trecho.
    """
    count = 0
    i = 0
    while i < len(oks):
        if oks[i]:
            i += 1
            continue
        start = i
        while i < len(oks) and not oks[i]:
            i += 1
        reference = times[start - 1] if start > 0 else times[0]
        if times[i - 1] - reference >= timeout:
            count += 1
    return count


frames = st.lists(
    st.tuples(st.floats(min_value=0.001, max_value=0.15), st.booleans()),
    min_size=1,
    max_size=80,
)


@given(frames)
def test_release_count_matches_long_loss_runs(sequence: list[tuple[float, bool]]) -> None:
    fake = FakeInputBackend()
    tracked = CountingTrackedInput(fake)
    config = SafetyConfig()
    manager = SafetyManager(tracked, config)
    times: list[float] = []
    t = 0.0
    for dt, _ in sequence:
        t += dt
        times.append(t)
    oks = [ok for _, ok in sequence]

    for t_n, ok in zip(times, oks, strict=True):
        manager.on_frame(t_n, tracking_ok=ok)

    assert tracked.release_all_calls == long_loss_runs(times, oks, config.lost_timeout_s)


# --- Pausa F8 e saída F9 (CA6, CA6b) --------------------------------------------


def test_f8_edge_toggles_pause_once() -> None:
    manager, fake = make_manager()
    manager.input.press("mouse_left")

    for _ in range(5):  # F8 segurado por 5 frames
        manager.on_hotkeys(frozenset({"f8"}))

    assert manager.state is SafetyState.PAUSED
    assert fake.pressed == set()


F8 = frozenset({"f8"})
F9 = frozenset({"f9"})
NONE: frozenset[str] = frozenset()


def test_press_ignored_while_paused() -> None:
    manager, fake = make_manager()
    manager.on_hotkeys(F8)

    manager.input.press("w")
    assert fake.pressed == set()

    manager.on_hotkeys(NONE)
    manager.on_hotkeys(F8)  # nova borda: retoma
    manager.input.press("w")

    assert manager.state is SafetyState.ACTIVE
    assert fake.pressed == {"w"}


def test_f9_edge_returns_exit_once() -> None:
    manager, _ = make_manager()

    actions = [manager.on_hotkeys(keys) for keys in (F9, F9, F9, NONE, F9)]

    assert actions == [
        SafetyAction.EXIT,
        SafetyAction.NONE,
        SafetyAction.NONE,
        SafetyAction.NONE,
        SafetyAction.EXIT,
    ]


def test_pause_has_priority_over_lost() -> None:
    manager, fake = make_manager()
    manager.on_frame(0.0, tracking_ok=True)
    manager.on_hotkeys(F8)
    manager.on_frame(0.5, tracking_ok=False)
    assert manager.state is SafetyState.PAUSED

    manager.on_hotkeys(NONE)
    manager.on_hotkeys(F8)  # retoma, mas continua sem pose
    assert manager.state is SafetyState.LOST
    manager.input.press("w")
    assert fake.pressed == set()

    manager.on_frame(0.6, tracking_ok=True)
    assert manager.state is SafetyState.ACTIVE


@given(st.lists(st.booleans(), max_size=60))
def test_toggle_count_equals_rising_edges(f8_held: list[bool]) -> None:
    """S3: alternâncias da pausa = bordas de subida de F8 (não frames pressionados)."""
    manager, _ = make_manager()
    toggles = 0
    previous_state = manager.state
    for held in f8_held:
        manager.on_hotkeys(F8 if held else NONE)
        if manager.state is not previous_state:
            toggles += 1
        previous_state = manager.state

    # [False, *f8_held] tem um elemento a mais de propósito: o estado antes do 1º frame.
    pairs = zip([False, *f8_held], f8_held, strict=False)
    rising_edges = sum(1 for prev, cur in pairs if cur and not prev)
    assert toggles == rising_edges


# --- Validação do SafetyConfig --------------------------------------------------------


@pytest.mark.parametrize("timeout", [0.0, -0.1, math.nan, math.inf])
def test_config_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="lost_timeout_s"):
        SafetyConfig(lost_timeout_s=timeout)


def test_config_rejects_same_pause_and_exit_key() -> None:
    with pytest.raises(ValueError, match="f8"):
        SafetyConfig(pause_key="f8", exit_key="f8")


@pytest.mark.parametrize("field", ["pause_key", "exit_key"])
def test_config_rejects_empty_key(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        SafetyConfig(**{field: ""})


def test_manager_rejects_always_release_mismatch() -> None:
    """always_release tem uma fonte só: a config tem de bater com o TrackedInput."""
    tracked = TrackedInput(FakeInputBackend(), always_release=frozenset())

    with pytest.raises(ValueError, match="always_release"):
        SafetyManager(tracked, SafetyConfig())


def test_tracked_input_exposes_always_release() -> None:
    tracked = TrackedInput(FakeInputBackend(), MOUSE_BUTTONS)

    assert tracked.always_release == MOUSE_BUTTONS


# ---------------------------------------------------------------------------
# Spec 002 T6 — on_stall (watchdog) e acesso de outra thread
# ---------------------------------------------------------------------------


def test_on_stall_releases_and_next_frame_recovers() -> None:
    manager, fake = make_manager()
    manager.on_frame(0.0, True)
    manager.input.press("w")
    manager.on_stall()
    assert "w" in fake.released()
    assert manager.input.pressed == frozenset()
    assert not manager.input.enabled
    manager.input.press("w")  # ignorado enquanto travado
    assert manager.input.pressed == frozenset()
    manager.on_frame(1.0, True)
    assert manager.input.enabled
    manager.input.press("w")
    assert manager.input.pressed == frozenset({"w"})


def test_on_stall_from_other_thread() -> None:
    import threading

    manager, fake = make_manager()
    started = threading.Event()
    errors: list[BaseException] = []

    def loop() -> None:
        try:
            for n in range(1, 3001):
                manager.on_frame(n * 0.001, True)
                manager.input.press("w")
                manager.input.release("w")
                manager.input.press("space")
                started.set()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=loop, daemon=True)
    worker.start()
    try:
        assert started.wait(timeout=2.0)
        for _ in range(200):
            manager.on_stall()
    finally:
        worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert errors == []
    # Com o loop encerrado, o último on_stall deixa tudo solto e desabilitado.
    manager.on_stall()
    assert manager.input.pressed == frozenset()
    assert fake.pressed == set()
    assert not manager.input.enabled


# --- Spec 017 CA8: perder o tracking com o agachar ativo solta o shift ---------------------


def _active_crouch_pipeline() -> tuple[Pipeline, SafetyManager, FakeInputBackend]:
    from inmotion.config.schema import Binding, GameProfile
    from inmotion.fps import FpsMeter
    from inmotion.gestures.base import GestureKind
    from inmotion.gestures.factory import build_body_gestures
    from inmotion.pipeline import Pipeline
    from tests.helpers import make_body

    fake = FakeInputBackend()
    safety = SafetyManager(
        TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig(lost_timeout_s=0.2, min_track_s=0.0)
    )
    profile = GameProfile("teste", {"crouch": Binding("shift", GestureKind.HOLD)})
    pipeline = Pipeline(
        safety, build_body_gestures({"crouch"}), FpsMeter(window=30), profile=profile
    )
    dt = 1 / 30
    for i in range(20):
        pipeline.step(make_body(t=i * dt), i * dt)
    for i in range(20, 26):  # agachado, com o quadril ainda no quadro: START
        pipeline.step(make_body(t=i * dt, shoulder_dy_px=120.0), i * dt)
    assert fake.pressed == {"shift"}
    return pipeline, safety, fake


def test_crouch_released_after_lost_timeout() -> None:
    """CA8: agachado e sem ombros (sem pose) por ``lost_timeout_s``, o ``shift`` é solto.
    O alívio da regra de tracking (RF4) vale só para o quadril, nunca para os ombros."""
    from tests.helpers import make_body

    pipeline, safety, fake = _active_crouch_pipeline()
    dt = 1 / 30
    t = 26 * dt
    pipeline.step(make_body(t=t, shoulder_dy_px=120.0, hips_visible=False), t)  # só ombros
    assert fake.pressed == {"shift"}
    gone = make_body(t=t, visibility=0.0)  # nem os ombros: pose fantasma
    for k in range(1, 12):
        pipeline.step(gone, t + k * dt)
    assert safety.state is SafetyState.LOST
    assert fake.pressed == set()
    assert fake.released().count("shift") == 1
