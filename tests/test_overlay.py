"""Spec 002 T8 — overlay mínimo (adaptador)."""

import numpy as np
import pytest

from inmotion.history import SeriesView
from inmotion.overlay import handedness_label, render
from inmotion.pipeline import StepResult
from inmotion.safety import SafetyState
from tests.fakes import make_pose

SIZE = (160, 120)  # (W, H)


def blank() -> np.ndarray:
    return np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)


def result_with_pose() -> StepResult:
    pose = make_pose({0: (80, 60)}, size=SIZE, t=1.0)
    return StepResult(t=1.0, fps=29.7, safety_state=SafetyState.ACTIVE, pose=pose)


def test_render_keeps_shape_and_dtype() -> None:
    img = blank()
    out = render(img, result_with_pose())
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out is not img


def test_render_without_pose() -> None:
    res = StepResult(t=0.0, fps=None, safety_state=SafetyState.LOST, pose=None)
    out = render(blank(), res)
    assert out.shape == (SIZE[1], SIZE[0], 3)
    assert out.any()  # o texto de estado foi desenhado


def test_overlay_ignores_missing_and_out_of_frame_coordinates() -> None:
    from dataclasses import replace

    result = result_with_pose()
    landmarks = result.pose.landmarks.copy()
    landmarks[0, :2] = np.nan
    landmarks[1, :2] = np.inf
    landmarks[2, :2] = 1e20
    pose = replace(result.pose, landmarks=landmarks)
    out = render(blank(), replace(result, pose=pose))
    assert out.shape == blank().shape


def test_render_draws_something_with_pose() -> None:
    out = render(blank(), result_with_pose())
    assert out[60, 80].any()  # ponto do nariz


@pytest.mark.parametrize("state", list(SafetyState))
def test_render_all_states(state: SafetyState) -> None:
    res = StepResult(t=0.0, fps=30.0, safety_state=state, pose=None)
    assert render(blank(), res).any()


def test_render_does_not_modify_input_or_result() -> None:
    img = blank()
    img.setflags(write=False)  # imagem somente leitura (view do FramePacket) também funciona
    res = result_with_pose()
    assert res.pose is not None
    landmarks_before = res.pose.landmarks.copy()
    render(img, res)
    assert not img.any()
    assert np.array_equal(res.pose.landmarks, landmarks_before)
    assert res.fps == 29.7


def test_render_draws_exit_zone_with_progress() -> None:
    from dataclasses import replace

    from inmotion.exit_zone import ExitZoneConfig, ExitZoneState

    cfg = ExitZoneConfig(x0=0.0, y0=0.5, x1=0.5, y1=1.0)
    base = StepResult(t=0.0, fps=None, safety_state=SafetyState.ACTIVE, pose=None)
    idle = render(blank(), replace(base, exit_zone=ExitZoneState(cfg, False, 0.0, False)))
    half = render(blank(), replace(base, exit_zone=ExitZoneState(cfg, True, 0.5, False)))
    zone = (slice(60, 120), slice(0, 80))  # [y0·H, y1·H) × [x0·W, x1·W)
    assert idle[zone].any()  # contorno do bloco
    assert half[zone].sum() > idle[zone].sum()  # barra de progresso preenchida


def test_handedness_label_names_the_mode_and_the_attacking_arm() -> None:
    """Spec 017 RF6: o texto diz Destro/Canhoto e qual braço ataca (ASCII: Hershey)."""
    assert handedness_label("right") == "Destro - ataca: braco direito"
    assert handedness_label("left") == "Canhoto - ataca: braco esquerdo"
    for text in (handedness_label("right"), handedness_label("left")):
        assert text.isascii()


def test_shows_handedness_label() -> None:
    """Sem lateralidade nada é desenhado na linha nova; com ela, cada modo tem o seu texto."""
    res = StepResult(t=0.0, fps=30.0, safety_state=SafetyState.ACTIVE, pose=None)
    row = (slice(52, 76), slice(0, SIZE[0]))  # abaixo do FPS (y = 20) e do estado (y = 42)
    plain = render(blank(), res)
    left = render(blank(), res, "left")
    right = render(blank(), res, "right")
    assert not plain[row].any()
    assert left[row].any()
    assert right[row].any()
    assert not np.array_equal(left[row], right[row])
    assert np.array_equal(plain[:52], left[:52])  # o resto do quadro não muda


def test_mirror_flips_image_and_pose_preserving_readable_hud() -> None:
    """Spec 009 (A4): ``mirror=True`` espelha frame, esqueleto e mantém o HUD legível."""
    from inmotion.pipeline import StepResult
    from inmotion.safety import SafetyState
    from inmotion.types import PoseFrame

    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, :40] = (200, 0, 0)  # faixa azul só na esquerda da câmera crua
    lm = np.full((33, 5), np.nan, dtype=np.float64)
    lm[0] = (0.25, 0.5, 0.0, 1.0, 1.0)  # x=40, y=60 na imagem crua
    pose = PoseFrame(t=0.0, landmarks=lm, image_size=(160, 120))
    res = StepResult(t=0.0, fps=30.0, safety_state=SafetyState.ACTIVE, pose=pose)

    out = render(img, res, mirror=True)
    # Faixa azul foi para a direita (160 - 40 .. 160); ponto em x=40 foi para 160 - 1 - 40 = 119.
    assert out[110, 150, 0] == 200
    assert out[60, 119, 1] == 255
    assert out[60, 119, 2] == 255  # ponto amarelo espelhado
    # Texto FPS no canto superior esquerdo (em área preta do espelho) permanece desenhado lá.
    assert out[8:26, 8:60].any()


def test_skeleton_bones_and_debug_panels_rendered() -> None:
    """Spec 009: ossos conectados, esqueleto normalizado M=S*R*T e telemetria no frame."""
    from inmotion.gestures.base import GestureKind, GestureState
    from inmotion.pipeline import GestureTelemetry, StepResult
    from inmotion.safety import SafetyState
    from inmotion.types import PoseFrame
    from inmotion.vision.geometry import NormalizedSkeleton
    from inmotion.vision.landmarks import PoseLandmark

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = np.full((33, 5), np.nan, dtype=np.float64)
    lm[PoseLandmark.LEFT_SHOULDER] = (0.4, 0.3, 0.0, 1.0, 1.0)
    lm[PoseLandmark.RIGHT_SHOULDER] = (0.6, 0.3, 0.0, 1.0, 1.0)
    pose = PoseFrame(t=0.0, landmarks=lm, image_size=(640, 480))

    norm_pts = np.zeros((33, 3), dtype=np.float64)
    norm_pts[PoseLandmark.LEFT_SHOULDER] = (-0.5, -1.0, 0.0)
    norm_pts[PoseLandmark.RIGHT_SHOULDER] = (0.5, -1.0, 0.0)
    skel = NormalizedSkeleton(
        points=norm_pts, matrix=np.eye(3, dtype=np.float64), shoulder_px=128.0
    )
    tele = (
        GestureTelemetry(
            name="crouch",
            kind=GestureKind.HOLD,
            state=GestureState.ACTIVE,
            raw_value=1.15,
            filtered_value=1.05,
            theta_on=0.90,
            theta_off=0.60,
        ),
    )
    res = StepResult(
        t=0.0,
        fps=60.0,
        safety_state=SafetyState.ACTIVE,
        pose=pose,
        look=(10, -5),
        pressed=frozenset({"shift"}),
        normalized=skel,
        gestures=tele,
    )

    out = render(img, res, handedness="right", mirror=False)
    # Ponto médio entre os ombros (320, 144) foi ligado por osso.
    assert out[144, 320].any()
    # Painel M=S*R*T no canto inferior direito e painel de telemetria no inferior esquerdo.
    assert out[320:472, 488:632].any()
    assert out[420:472, 8:378].any()


# ---------------------------------------------------------------------------
# Spec 009 (revisão de 18/11): níveis, gráfico e pureza (CA1, CA2)
# ---------------------------------------------------------------------------

FULL_SIZE = (640, 480)


def full_result() -> StepResult:
    """Pessoa inteira de frente, esqueleto normalizado e dois gestos na telemetria."""
    from inmotion.gestures.base import GestureKind, GestureState
    from inmotion.pipeline import GestureTelemetry
    from inmotion.vision.geometry import GeometryConfig, normalize_skeleton
    from tests.helpers import make_body, with_face

    pose = with_face(make_body(t=1.0))
    skel = normalize_skeleton(pose, GeometryConfig())
    tele = tuple(
        GestureTelemetry(name, GestureKind.TRIGGER, GestureState.IDLE, 0.5, 0.4, 0.8, 0.6)
        for name in ("attack", "jump")
    )
    return StepResult(
        t=1.0,
        fps=30.0,
        safety_state=SafetyState.ACTIVE,
        pose=pose,
        normalized=skel,
        gestures=tele,
        pressed=frozenset({"w"}),
    )


def series() -> SeriesView:
    t = np.linspace(0.0, 1.0, 31)
    return SeriesView(
        t=t,
        raw=np.sin(6 * t),
        filtered=np.sin(6 * t - 0.3),
        theta_on=0.8,
        theta_off=0.6,
        window_s=1.0,
    )


def camera_image() -> np.ndarray:
    rng = np.random.default_rng(9)
    return rng.integers(0, 256, (FULL_SIZE[1], FULL_SIZE[0], 3), dtype=np.uint8)


def test_render_twice_is_identical() -> None:
    """O1: mesmo resultado e mesma vista → mesma imagem; nada de entrada muda."""
    img = camera_image()
    before = img.copy()
    res, view = full_result(), series()
    raw_before = view.raw.copy()
    first = render(img, res, "right", mirror=True, plot=view, plot_name="attack")
    second = render(img, res, "right", mirror=True, plot=view, plot_name="attack")
    assert np.array_equal(first, second)
    assert np.array_equal(img, before)
    assert np.array_equal(view.raw, raw_before)


@pytest.mark.parametrize("mirror", [False, True])
def test_off_level_returns_the_camera_image(mirror: bool) -> None:
    from inmotion.overlay import OverlayLevel

    img = camera_image()
    out = render(img, full_result(), "right", mirror=mirror, level=OverlayLevel.OFF)
    assert out is not img
    assert np.array_equal(out, img[:, ::-1] if mirror else img)


def test_levels_change_output() -> None:
    """CA2: o ``minimal`` tira os painéis (miniquadro e tabela), mas mantém o esqueleto."""
    from inmotion.overlay import OverlayLevel

    img = np.zeros((FULL_SIZE[1], FULL_SIZE[0], 3), dtype=np.uint8)
    res = full_result()
    full = render(img, res, "right", level=OverlayLevel.FULL)
    minimal = render(img, res, "right", level=OverlayLevel.MINIMAL)
    inset = (slice(320, 472), slice(488, 632))
    table = (slice(420, 472), slice(8, 378))
    assert full[inset].any()
    assert full[table].any()
    assert not minimal[inset].any()
    assert not minimal[table].any()
    assert np.array_equal(minimal[150, 320], full[150, 320])  # osso entre os ombros
    assert minimal[150, 320].any()


def test_full_draws_elbow_angles_and_minimal_does_not() -> None:
    """RF3 (A7): arco e graus perto de cada cotovelo só no ``full``."""
    from inmotion.overlay import OverlayLevel
    from inmotion.vision.landmarks import PoseLandmark

    img = np.zeros((FULL_SIZE[1], FULL_SIZE[0], 3), dtype=np.uint8)
    res = full_result()
    assert res.pose is not None
    full = render(img, res, level=OverlayLevel.FULL)
    minimal = render(img, res, level=OverlayLevel.MINIMAL)
    w, h = FULL_SIZE
    for elbow in (PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW):
        x = round(float(res.pose.landmarks[elbow, 0]) * w)
        y = round(float(res.pose.landmarks[elbow, 1]) * h)
        near = (slice(y - 40, y + 40), slice(x - 40, x + 40))
        assert not np.array_equal(full[near], minimal[near])


PLOT_BOX = (slice(8, 130), slice(372, 632))  # canto superior direito em 640×480


def test_plot_drawn_only_with_view() -> None:
    """RF2: o gráfico aparece no ``full`` com uma vista; sem ela ou no ``minimal``, não."""
    from inmotion.overlay import OverlayLevel

    img = np.zeros((FULL_SIZE[1], FULL_SIZE[0], 3), dtype=np.uint8)
    res = full_result()
    with_plot = render(img, res, plot=series(), plot_name="attack")
    without = render(img, res)
    minimal = render(img, res, level=OverlayLevel.MINIMAL, plot=series(), plot_name="attack")
    assert with_plot[PLOT_BOX].any()
    assert not without[PLOT_BOX].any()
    assert not minimal[PLOT_BOX].any()


def test_plot_survives_gaps_and_short_series() -> None:
    """NaN (frames sem valor) interrompe a linha; série curta ou vazia não quebra."""
    from inmotion.history import SeriesView

    img = np.zeros((FULL_SIZE[1], FULL_SIZE[0], 3), dtype=np.uint8)
    t = np.linspace(0.0, 1.0, 10)
    raw = np.where(np.arange(10) % 3 == 0, np.nan, t)
    gappy = SeriesView(t=t, raw=raw, filtered=raw, theta_on=0.8, theta_off=0.6, window_s=3.0)
    empty = SeriesView(
        t=np.empty(0),
        raw=np.empty(0),
        filtered=np.empty(0),
        theta_on=0.8,
        theta_off=0.6,
        window_s=3.0,
    )
    for view in (gappy, empty):
        out = render(img, full_result(), plot=view, plot_name="attack")
        assert out.shape == img.shape


@pytest.mark.slow
def test_render_under_3ms() -> None:
    """CA7 (RNF1): ``full`` com espelho, ângulos, miniquadro, tabela e gráfico < 3 ms."""
    import time

    img = camera_image()
    res, view = full_result(), series()
    for _ in range(20):  # aquece caches
        render(img, res, "right", mirror=True, plot=view, plot_name="attack")
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        render(img, res, "right", mirror=True, plot=view, plot_name="attack")
    mean_ms = (time.perf_counter() - t0) / n * 1000.0
    print(f"overlay full: {mean_ms:.2f} ms/frame")
    assert mean_ms < 3.0


# --- Indicador de recepção de voz (27/11/2026) ---

VOICE_BAR = (slice(70, 90), slice(150, 260))  # à direita de "Voz: ouvindo", na linha y = 84
VOICE_HEARD = (slice(92, 112), slice(0, 320))  # linha de baixo: a última frase


def _voice_result(
    level: float,
    heard: str | None = None,
    heard_t: float | None = None,
    accepted: bool = False,
    t: float = 10.0,
) -> StepResult:
    from inmotion.voice import VoiceReception

    return StepResult(
        t=t,
        fps=30.0,
        safety_state=SafetyState.ACTIVE,
        pose=None,
        voice_status="ouvindo",
        voice=VoiceReception(level, heard, heard_t, accepted),
    )


def _black() -> np.ndarray:
    return np.zeros((FULL_SIZE[1], FULL_SIZE[0], 3), dtype=np.uint8)


def test_voice_meter_grows_with_the_level() -> None:
    quiet = render(_black(), _voice_result(0.0))
    half = render(_black(), _voice_result(0.5))
    full = render(_black(), _voice_result(1.0))
    assert quiet[VOICE_BAR].any()  # moldura da barra
    filled = [int((img[VOICE_BAR].sum(axis=2) > 0).sum()) for img in (quiet, half, full)]
    assert filled[0] < filled[1] < filled[2]


def test_voice_shows_recent_phrase_and_hides_old_one() -> None:
    from inmotion.overlay import OverlayLevel

    heard = render(_black(), _voice_result(0.2, "abrir inventário", 9.0, True))
    ignored = render(_black(), _voice_result(0.2, "abrir inventário", 9.0, False))
    old = render(_black(), _voice_result(0.2, "abrir inventário", 1.0, True))
    nothing = render(_black(), _voice_result(0.2))
    assert heard[VOICE_HEARD].any()
    assert not np.array_equal(heard[VOICE_HEARD], ignored[VOICE_HEARD])  # cor diferente
    assert not old[VOICE_HEARD].any()
    assert not nothing[VOICE_HEARD].any()
    minimal = render(
        _black(), _voice_result(0.2, "abrir menu", 9.0, True), level=OverlayLevel.MINIMAL
    )
    assert minimal[VOICE_HEARD].any()  # o indicador também aparece no nível minimal


def test_no_voice_means_no_indicator() -> None:
    res = StepResult(t=0.0, fps=30.0, safety_state=SafetyState.ACTIVE, pose=None)
    out = render(_black(), res)
    assert not out[VOICE_BAR].any()
    assert not out[VOICE_HEARD].any()


def test_voice_text_is_ascii() -> None:
    from inmotion.overlay import voice_heard_label

    assert voice_heard_label("abrir inventário", accepted=True) == 'ouvi: "abrir inventario"'
    assert voice_heard_label("abrir menu", accepted=False) == 'ouvi: "abrir menu" (ignorado)'
    # Com a confiança da palavra mais fraca (27/11): ajuda a escolher voice.min_confidence.
    ok = voice_heard_label("abrir menu", accepted=True, confidence=0.97, threshold=0.9)
    assert ok == 'ouvi: "abrir menu" (0,97)'
    low = voice_heard_label("voltar", accepted=False, confidence=0.824, threshold=0.9)
    assert low == 'ouvi: "voltar" (ignorado: 0,82 < 0,90)'
    # Palavra solta do vocabulário (dados de 27/11): o motivo é a frase incompleta, não a
    # confiança, mesmo com confiança alta.
    part = voice_heard_label(
        "pausar", accepted=False, confidence=0.95, threshold=0.9, in_vocabulary=False
    )
    assert part == 'ouvi: "pausar" (frase incompleta)'
    assert voice_heard_label("[unk]", accepted=False) == "ouvi: frase desconhecida"
