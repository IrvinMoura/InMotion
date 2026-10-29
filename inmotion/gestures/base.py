"""Base dos gestos: máquina de estados com debounce (R5) e histerese (R6).

Núcleo puro: o tempo entra sempre como parâmetro `t`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal


class GestureKind(Enum):
    HOLD = auto()
    TRIGGER = auto()
    ANALOG = auto()


class Direction(Enum):
    BELOW = auto()
    ABOVE = auto()


class GestureState(Enum):
    IDLE = auto()
    ARMING = auto()
    ACTIVE = auto()
    COOLDOWN = auto()


class EventKind(Enum):
    START = auto()
    END = auto()
    PULSE = auto()
    VALUE = auto()


@dataclass(frozen=True)
class GestureParams:
    """Parâmetros de um gesto de limiar (`hold` ou `trigger`).

    Sentido `BELOW` (ativa com f < θ_on): exige θ_on < θ_off (RF8, CA7).
    Sentido `ABOVE` (ativa com f > θ_on): exige θ_on > θ_off. A faixa [θ_on, θ_off] é a
    banda de histerese R6; o debounce R5 exige `debounce_frames` = N ≥ 1 frames seguidos.
    """

    theta_on: float
    theta_off: float
    direction: Direction = Direction.BELOW
    debounce_frames: int = 3
    cooldown_s: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.theta_on) and math.isfinite(self.theta_off)):
            raise ValueError("theta_on e theta_off devem ser finitos")
        if self.direction is Direction.BELOW and not self.theta_on < self.theta_off:
            raise ValueError("BELOW exige theta_on < theta_off")
        if self.direction is Direction.ABOVE and not self.theta_on > self.theta_off:
            raise ValueError("ABOVE exige theta_on > theta_off")
        if self.debounce_frames < 1:
            raise ValueError("debounce_frames deve ser >= 1")
        if not (math.isfinite(self.cooldown_s) and self.cooldown_s >= 0.0):
            raise ValueError("cooldown_s deve ser finito e >= 0")


@dataclass(frozen=True)
class AnalogParams:
    """Parâmetros do gesto analógico: zona morta (≥ 0) e ganho."""

    dead_zone: float = 0.0
    gain: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.dead_zone) and self.dead_zone >= 0.0):
            raise ValueError("dead_zone deve ser finito e >= 0")
        if not math.isfinite(self.gain):
            raise ValueError("gain deve ser finito")


@dataclass(frozen=True)
class GestureEvent:
    gesture: str
    kind: EventKind
    t: float
    value: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class GestureSnapshot:
    name: str
    kind: GestureKind
    state: GestureState
    counter: int
    last_value: float | None
    theta_on: float
    theta_off: float
    confidence: float


class Gesture:
    """Máquina de estados IDLE → ARMING → ACTIVE → COOLDOWN com R5 (debounce) e R6 (histerese).

    Modelo (sentido `BELOW`; o `ABOVE` usa f' = −f e θ' = −θ, uma única implementação):

    - (R6) a condição de ativação vale com f < θ_on; o gesto só sai de ACTIVE com f > θ_off;
      entre os dois limiares o estado anterior é mantido (histerese).
    - (R5) c_n = c_{n−1} + 1 se a condição vale; senão c_n = 0. Ativa quando c_n ≥ N.
    - Tipos: HOLD emite START ao entrar em ACTIVE e END ao sair (f > θ_off, f inválido ou
      `close()`); TRIGGER emite um PULSE ao entrar em ACTIVE e nunca END.
    - Ao sair de ACTIVE entra em COOLDOWN (t_end = t) e só volta a IDLE quando
      t − t_end ≥ T_cd, reavaliando f no mesmo frame.

    Propriedades: E1 (anti-flicker), E2 (START só após N frames), E3 (START/END alternados
    e balanceados após `close()`), E4 (pulsos distam ≥ T_cd). O tempo `t` é sempre parâmetro.
    """

    def __init__(
        self,
        name: str,
        kind: Literal[GestureKind.HOLD, GestureKind.TRIGGER],
        params: GestureParams,
    ) -> None:
        if kind not in (GestureKind.HOLD, GestureKind.TRIGGER):
            raise ValueError("Gesture só aceita HOLD ou TRIGGER (use AnalogGesture)")
        self._name = name
        self._kind = kind
        self._params = params
        # Espelho de sinal: no ABOVE, f' = −f e θ' = −θ; assim `BELOW` vale para os dois.
        self._sign = 1.0 if params.direction is Direction.BELOW else -1.0
        self._on = self._sign * params.theta_on
        self._off = self._sign * params.theta_off
        self._state = GestureState.IDLE
        self._count = 0
        self._t_end = 0.0
        self._last: float | None = None

    def update(self, f: float | None, t: float) -> GestureEvent | None:
        """Avança a máquina com a feature `f` (ou None) no instante `t`.

        Um f não finito (NaN/inf) conta como None: feature inválida (RF4).
        """
        if f is not None and not math.isfinite(f):
            f = None
        self._last = f
        s = None if f is None else self._sign * f

        if self._state is GestureState.COOLDOWN:
            if t - self._t_end < self._params.cooldown_s:
                return None
            self._state = GestureState.IDLE  # reavalia f neste mesmo frame
            self._count = 0

        if self._state is GestureState.ACTIVE:
            # R6: só sai além de θ_off (ou com feature inválida); na banda, mantém.
            if s is None or s > self._off:
                return self._deactivate(t)
            return None

        # IDLE ou ARMING (R5): conta frames seguidos com f < θ_on.
        if s is None or not s < self._on:
            self._state = GestureState.IDLE
            self._count = 0
            return None
        self._count += 1
        self._state = GestureState.ARMING
        if self._count >= self._params.debounce_frames:
            return self._activate(t)
        return None

    def _activate(self, t: float) -> GestureEvent:
        self._state = GestureState.ACTIVE
        # TRIGGER: um único PULSE; o período refratário é ACTIVE → COOLDOWN → IDLE (E4).
        kind = EventKind.START if self._kind is GestureKind.HOLD else EventKind.PULSE
        return GestureEvent(self._name, kind, t, confidence=self._confidence())

    def _deactivate(self, t: float) -> GestureEvent | None:
        self._state = GestureState.COOLDOWN
        self._t_end = t
        self._count = 0
        if self._kind is GestureKind.HOLD:
            return GestureEvent(self._name, EventKind.END, t)
        return None

    def _confidence(self) -> float:
        """Confiança c = clip(|f − θ_on| / |θ_off − θ_on|, 0, 1) no lado ativo; 0 fora dele."""
        if self._last is None:
            return 0.0
        margin = self._on - self._sign * self._last  # > 0 só no lado ativo (f < θ_on)
        if margin <= 0.0:
            return 0.0
        return min(margin / (self._off - self._on), 1.0)

    def reset(self) -> None:
        """RF5b: volta a IDLE com c = 0, SEM emitir evento.

        Usado quando o SafetyManager já soltou tudo (PAUSED/LOST): uma pose mantida
        re-arma e emite START de novo após N frames. Zera também o cooldown.
        """
        self._state = GestureState.IDLE
        self._count = 0
        self._t_end = 0.0
        self._last = None

    def close(self, t: float) -> GestureEvent | None:
        """RF5: encerra o HOLD ativo com END, para nunca sobrar input pressionado."""
        if self._kind is GestureKind.HOLD and self._state is GestureState.ACTIVE:
            return self._deactivate(t)
        return None

    def snapshot(self) -> GestureSnapshot:
        """Estado inspecionável para o overlay (RF6)."""
        return GestureSnapshot(
            name=self._name,
            kind=self._kind,
            state=self._state,
            counter=self._count,
            last_value=self._last,
            theta_on=self._params.theta_on,
            theta_off=self._params.theta_off,
            confidence=self._confidence(),
        )


class AnalogGesture:
    """Gesto analógico: um VALUE por frame, com zona morta e ganho (RF3, CA5).

    Saída y = 0 se x é None/não finito ou |x| ≤ zona morta; senão y = ganho · x.
    Não tem estados de histerese; o `state` do snapshot é ACTIVE fora da zona morta.
    """

    def __init__(self, name: str, params: AnalogParams) -> None:
        self._name = name
        self._params = params
        self._last: float | None = None
        self._out = 0.0

    def update(self, x: float | None, t: float) -> GestureEvent:
        """Emite sempre um VALUE com a saída analógica do frame."""
        if x is not None and not math.isfinite(x):
            x = None
        self._last = x
        if x is None or abs(x) <= self._params.dead_zone:
            self._out = 0.0
        else:
            self._out = self._params.gain * x
        return GestureEvent(self._name, EventKind.VALUE, t, value=self._out)

    def reset(self) -> None:
        """Volta ao estado inicial (saída 0), sem evento."""
        self._last = None
        self._out = 0.0

    def snapshot(self) -> GestureSnapshot:
        """Estado para o overlay; os limiares são a zona morta e a confiança é min(|y|, 1)."""
        return GestureSnapshot(
            name=self._name,
            kind=GestureKind.ANALOG,
            state=GestureState.ACTIVE if self._out != 0.0 else GestureState.IDLE,
            counter=0,
            last_value=self._last,
            theta_on=self._params.dead_zone,
            theta_off=self._params.dead_zone,
            confidence=min(abs(self._out), 1.0),
        )
