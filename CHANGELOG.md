# Changelog

Mudanças de cada versão, no formato [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).
As versões 0.1 e 0.2 foram reconstruídas a partir do histórico do git (não tinham tag).

## [0.3.0] - Entrega 3

### Adicionado
- Marcha no lugar: andar (`W`) pelos passos alternados dos joelhos e correr (`Ctrl`) pela
  cadência.
- Câmera pela cabeça: virar a cabeça move o mouse na horizontal e inclinar move na vertical,
  com zona morta, curva de resposta e movimento independente do FPS.
- Gestos do tronco: recuar (`S`) e andar para os lados (`A`/`D`) inclinando o tronco.
- Overlay: níveis `full`/`minimal`/`off` (tecla `o`), gráfico dos últimos 3 s do sinal bruto
  contra o filtrado com os limiares (`Tab` troca o gesto), ângulo dos cotovelos, fundo de
  Sobel (`s`) e latência no HUD. Janela redimensionável.
- `tools.evaluate`: precisão, recall e falsos positivos por minuto de cada gesto nas sessões
  guiadas, matriz de confusão e comparação dos filtros, com gráficos.
- `tools.check_env`: diagnóstico de Python, pacotes, modelo, input e câmera.
- Sessão guiada `look` (câmera) e fixtures reais de socos, agachar, marcha e câmera.
- Troca automática entre as câmeras `1` e `0`.

### Mudado
- O ruff impede o núcleo de importar o overlay.

### Corrigido
- Quem ligava o app perto da câmera e depois ia para o lugar ficava com a base do pulo e do
  agachar errada (agachar preso, pulo sem resposta). Agora, se a distância muda muito por 3 s,
  o app solta as teclas e mede as bases de novo (`baseline.scale_band`, `scale_hold_s`).

## [0.2.0] - 2026-11-01

### Adicionado
- Configuração em camadas (`default.yaml` ← jogo ← perfil ← `--set`), validada com mensagens
  que apontam o campo.
- Gestos concretos: soco com cada braço, pulo, agachar e inclinar, com base adaptativa (R7).
- Árbitro com prioridades e grupos exclusivos definidos no perfil do jogo.
- Perfil do Minecraft, modo canhoto e agachar pela queda dos ombros.
- Geometria homogênea (A1–A9) e normalização do esqueleto `M = S·R·T`.
- Filtros EMA e One Euro.
- Gravação `.npz`, replay determinístico e sessões guiadas (`tools.guided`).

## [0.1.0] - 2026-10-30

### Adicionado
- `SafetyManager`: solta todo o input ao perder a pose, ao pausar (`F8`), ao sair (`F9`), em
  exceções e pelo watchdog.
- Pipeline modular com câmera em thread e MediaPipe Tasks `PoseLandmarker`.
- Overlay mínimo com FPS e estado.
