"""Medidor de FPS (núcleo puro: o tempo entra como parâmetro)."""

from collections import deque


class FpsMeter:
    """FPS por janela deslizante dos últimos ``window`` intervalos entre frames.

    Com ``m = min(window, n)``: ``fₙ = m / (tₙ − tₙ₋ₘ)``. Com ``Δt`` constante, ``fₙ = 1/Δt``
    exatamente; em geral é a média harmônica das taxas instantâneas da janela.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 1:
            raise ValueError(f"window deve ser >= 1, recebido {window}")
        self._times: deque[float] = deque(maxlen=window + 1)

    def update(self, t: float) -> float | None:
        """Registra o frame em ``t`` (segundos); ``None`` enquanto houver só um frame.

        Raises:
            ValueError: se ``t`` não for estritamente crescente.
        """
        if self._times and t <= self._times[-1]:
            raise ValueError(f"t deve ser estritamente crescente: {t} <= {self._times[-1]}")
        self._times.append(t)
        if len(self._times) < 2:
            return None
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])
