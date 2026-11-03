# Desenvolvimento e validação do InMotion

O projeto é um protótipo de controle por movimento. Uma aprovação dos testes valida os cenários automatizados; a validação de câmera, latência e compatibilidade com jogos é uma etapa adicional.

## Arquitetura

- `capture/`, `vision/pose.py`, `input/` e `overlay.py` adaptam webcam, MediaPipe, Windows e janela de debug.
- `pipeline.py`, `gestures/`, `filters/`, `mapping.py` e `safety.py` recebem tempo explicitamente e podem ser executados com poses sintéticas ou gravadas.
- `session.py` monta os componentes usados pelo app, replay e simulador. Mudanças nessa montagem devem valer para os três fluxos.
- `app.py` coordena captura, inferência, segurança, watchdog e liberação dos recursos.
- `tools.replay` e `tools.simulate` usam um backend que só registra comandos; não enviam input ao sistema operacional.

Há referências históricas a specs e ADRs em comentários do código. Esses documentos não estão incluídos no repositório; as orientações necessárias para contribuir estão neste arquivo, no README e nos testes.

## Decisões de segurança

1. A câmera mantém somente o frame mais recente para evitar acúmulo de latência.
2. O horário da captura acompanha a pose e os filtros. A segurança no app usa o horário após a inferência e descarta poses antigas, conforme `safety.lost_timeout_s`.
3. Apenas `TrackedInput` envia comandos ao backend. Ele registra a tecla antes da chamada, pois um backend pode falhar depois de já ter enviado o comando.
4. Pausa, perda e watchdog invalidam o estado do pipeline. Um contador registra essa invalidação mesmo que ela ocorra entre dois frames; na retomada, gestos, árbitro, mapper, filtros e bases recomeçam.
5. Ao entrar na zona de saída, comandos são liberados imediatamente. Ao sair dela antes de encerrar, o rastreamento e os gestos precisam armar novamente.
6. Vários gestos podem manter a mesma tecla. Ela só é liberada depois do último hold e do último tap pendente.
7. O backend físico desliga o failsafe de canto do PyDirectInput para permitir liberar botões mesmo com o cursor no canto. As saídas de emergência são F8, F9, perda de pose, watchdog e encerramento normal.

`atexit` ajuda no encerramento normal do interpretador; não é garantia em encerramento forçado do processo. Falhas permanentes do backend também impedem garantir que o sistema operacional recebeu a soltura.

## Verificação sem hardware

Use Python 3.10 e instale `requirements-dev.txt` na `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pytest -m "not hardware"
.\.venv\Scripts\python.exe -m tools.simulate --headless --output logs/simulacao.json
```

O CI omite testes `slow` e `hardware`. Localmente, `-m "not hardware"` inclui os testes demorados. Os testes com fakes de câmera não acessam uma webcam real.

Os relatórios JSON do replay/simulador contêm frames, duração, eventos reconhecidos após arbitragem e comandos com timestamps. Eles servem para inspeção e comparação; não são uma taxa de acerto do reconhecimento. Rótulos como `march_fast` descrevem uma fase com vários passos, não uma detecção individual.

## Validação que ainda exige hardware

1. Abrir a câmera correta e conferir resolução, iluminação e visibilidade do corpo.
2. Usar a sessão guiada, que não envia input real, e medir acertos e comandos extras.
3. Repetir com pessoas diferentes, distâncias e iluminação variadas. Ajustar limiares com gravações e manter recortes para regressão.
4. Medir FPS e latência de inferência com `tools.bench_pose`. Uma gravação de landmarks não substitui esse benchmark.
5. No app, verificar F8, F9, saída do quadro e encerramento da janela enquanto há comandos ativos.
6. Testar o perfil no jogo com a janela em foco, incluindo agachar, andar/correr, pular e conflito dos dois braços.

Antes de apresentar o projeto como concluído, os autores ainda precisam definir uma licença e registrar resultados de validação em hardware. Não há evidência suficiente para prometer funcionamento em todas as câmeras, computadores ou jogos.
