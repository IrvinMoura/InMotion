# InMotion - Controle de Jogos por Movimento (Projeto A3)

Jogue com o corpo, usando apenas uma webcam: o InMotion reconhece gestos e os converte em teclas e cliques do mouse. O perfil incluído é o do **Minecraft**.

> **Status:** versão preliminar `0.3.0` (Entrega 3). Os limiares dos gestos foram escolhidos com gravações de sessões guiadas e são medidos com `tools.evaluate`.

### Núcleo do Sistema
| Módulo                     | Descrição                                                         |
|----------------------------|-------------------------------------------------------------------|
| **Captura**                | Lê frames da webcam via OpenCV (MJPG, 640x480, até 60 FPS)        |
| **Pose**                   | Estima os landmarks do corpo com o MediaPipe PoseLandmarker       |
| **Gestos e Árbitro**       | Reconhece socos, pulo, agachar e marcha; resolve conflitos        |
| **Segurança**              | Garante que nenhuma tecla ou botão fique preso                    |
| **Overlay**                | Janela de debug: esqueleto, ângulos, sinais e estado dos gestos   |
| **Ferramentas**            | Gravação, replay, sessões guiadas, métricas e benchmark           |

### Gestos e Controles (Minecraft)

| Ação no jogo       | Gesto                                   | Entrada                 | Tipo   |
|--------------------|-----------------------------------------|-------------------------|--------|
| Atacar / quebrar   | Braço dominante esticado para a câmera  | Botão esquerdo do mouse | segura |
| Usar / colocar     | Braço não dominante esticado            | Botão direito do mouse  | segura |
| Pular              | Subida do quadril                       | `Espaço`                | toque  |
| Agachar            | Ombros abaixo da base calibrada         | `Shift`                 | segura |
| Andar              | Marcha no lugar (2 passos alternados)   | `W`                     | segura |
| Correr             | Marcha rápida (soma ao andar)           | `Ctrl`                  | segura |
| Recuar             | Inclinar o tronco para trás             | `S`                     | segura |
| Andar p/ esquerda  | Inclinar o tronco para a esquerda       | `A`                     | segura |
| Andar p/ direita   | Inclinar o tronco para a direita        | `D`                     | segura |
| Olhar (câmera)     | Virar e inclinar a cabeça               | Movimento do mouse      | contínuo |

- Com **os dois braços esticados juntos**, os cliques são suprimidos (combo `both_arms`)
- **Agachar** vence pular, andar e correr
- **Modo canhoto:** `--set handedness=left` troca os braços
- **Câmera:** a posição neutra da cabeça é medida no primeiro segundo de cada retomada; ao agachar a câmera para, e durante a marcha e a inclinação do tronco só o olhar para cima/baixo para
- O gesto `lean` (inclinar o tronco) existe no código, mas não tem binding no Minecraft

### Como Funciona

```
Câmera → Pré-processamento → Pose (MediaPipe) → Filtro (One Euro / EMA)
      → Gestos → Árbitro → Mapeador de ações → Teclado/mouse
                    ↑
        Segurança (perda de pose, pausa, watchdog)
```

- Cada gesto é uma feature escalar com histerese (`theta_on`/`theta_off`), debounce e cooldown
- O esqueleto é normalizado pela largura dos ombros, e pulo e agachar usam uma base adaptativa
- O núcleo é puro: recebe o tempo como parâmetro e não importa `cv2`, `mediapipe` nem `pydirectinput`, então tudo é testável por replay, sem hardware

### Bibliotecas Principais

| Categoria       | Tecnologias                                                                 | Versão     | Uso Específico                                   |
|-----------------|-----------------------------------------------------------------------------|------------|--------------------------------------------------|
| **Visão**       | <img src="https://img.shields.io/badge/MediaPipe-000000?logo=google">       | 0.10.31    | Estimativa de pose (PoseLandmarker)              |
|                 | <img src="https://img.shields.io/badge/OpenCV-000000?logo=opencv">          | 4.12.0.88  | Captura da webcam e janela de debug              |
| **Dados**       | <img src="https://img.shields.io/badge/NumPy-000000?logo=numpy">            | 2.2.6      | Geometria, filtros e gravações `.npz`            |
| **Configuração**| <img src="https://img.shields.io/badge/PyYAML-000000">                      | 6.0.3      | Leitura segura (`safe_load`) dos perfis          |
| **Input**       | <img src="https://img.shields.io/badge/PyDirectInput-000000">               | 1.0.4      | Injeção de teclado e mouse (somente Windows)     |
| **Qualidade**   | <img src="https://img.shields.io/badge/Pytest-000000?logo=pytest">          | 9.2.0      | Testes (com Hypothesis 6.168.5)                  |
|                 | <img src="https://img.shields.io/badge/Ruff-000000?logo=ruff">              | 0.17.3     | Lint e formatação                                |
|                 | <img src="https://img.shields.io/badge/mypy-000000?logo=python">            | 2.4.1      | Tipos (estrito no núcleo)                        |

### Infraestrutura

```bash
Python 3.10      # Versão fixa (>=3.10,<3.11)
Windows          # Injeção de input e teclas globais
GitHub Actions   # CI em windows-latest (ruff, mypy, pytest)
```

## Configuração e Execução

### Pré-requisitos
- Windows
- Python 3.10
- Webcam, com boa iluminação e ombros e quadris visíveis (testado a 1,5 m e 2 m)
- Git (para clonagem)

### Instalação Passo-a-Passo

```bash
# Clone o repositório
git clone https://github.com/IrvinMoura/InMotion.git
cd InMotion

# Crie um ambiente virtual com o Python 3.10
py -3.10 -m venv .venv
.venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Baixe o modelo de pose (lite | full | heavy)
python tools/download_model.py --variant full

# Verifique se o ambiente e a câmera estão prontos (opcional: --skip-camera)
python -m tools.check_env
```

O arquivo `.task` é salvo em `models/` e não é versionado.

## Execução
```bash
python -m inmotion --game minecraft
```

| Tecla | Efeito                                         |
|-------|------------------------------------------------|
| `F8`  | Pausa/retoma (solta todo o input ao pausar)    |
| `F9`  | Encerra o app                                  |
| `o`   | Nível do overlay: `full` → `minimal` → `off`   |
| `Tab` | Troca o gesto do gráfico bruto × filtrado      |
| `s`   | Liga/desliga o fundo de Sobel (gradiente, A3)  |

`F8` e `F9` funcionam com qualquer janela em foco; `o`, `Tab` e `s` só com a janela do overlay em foco.

Há também uma zona de saída provisória: manter a mão no retângulo do canto superior esquerdo da imagem por 4 s encerra o app.

> O índice de câmera padrão é `1` (`config/default.yaml`), com fallback automático entre os índices `1` e `0`. Se precisar escolher outro dispositivo, use `--camera N`.

O overlay de debug exibe o vídeo espelhado horizontalmente (`overlay.mirror: true`), esqueleto conectado, ângulo dos cotovelos, miniquadro do esqueleto normalizado ($M = S \cdot R \cdot T$), gráfico dos últimos 3 s do sinal bruto contra o filtrado com os limiares (`theta_off`/`theta_on`), estado de cada gesto, inputs ativos, FPS e latência (da captura até o input). A janela é redimensionável, para dividir a tela com o jogo. O nível inicial vem de `overlay.level` (`full`, `minimal` ou `off`).

### Opções da Linha de Comando

| Opção                  | Descrição                                                                  |
|------------------------|----------------------------------------------------------------------------|
| `--config ARQUIVO`     | YAML base, no lugar de `config/default.yaml`                               |
| `--game NOME`          | Perfil em `config/games/NOME.yaml` (ex.: `minecraft`)                      |
| `--profile ARQUIVO`    | YAML do usuário (ex.: lateralidade e limiares), aplicado depois do jogo    |
| `--set CAMINHO=VALOR`  | Sobrescreve um campo (repetível). Ex.: `--set filters.ema.alpha=0.4`       |
| `--camera N`           | Índice da câmera                                                           |
| `--camera-backend`     | `dshow`, `msmf` ou `any`                                                   |
| `--model-dir PASTA`    | Pasta do modelo `pose_landmarker_<variante>.task`                          |
| `--model-variant`      | `lite`, `full` ou `heavy`                                                  |
| `--log-level`          | `DEBUG`, `INFO`, `WARNING` ou `ERROR`                                      |
| `--log-file NOME`      | Grava o log também em `logs/NOME`                                          |
| `--no-overlay`         | Não abre a janela de debug (mais FPS)                                      |

Código de saída `2` indica configuração ou ambiente inválido (perfil inválido, modelo ausente, câmera indisponível).

### Configuração em Camadas

```
config/default.yaml  ←  config/games/<jogo>.yaml  ←  perfil do usuário  ←  --set / CLI
```

```bash
# Modo canhoto
python -m inmotion --game minecraft --set handedness=left

# Filtro EMA, sem janela de debug
python -m inmotion --set filters.kind=ema --set filters.ema.alpha=0.4 --no-overlay

# Pulo mais sensível
python -m inmotion --set gestures.jump.theta_on=0.20
```

Para outro jogo, copie `config/games/minecraft.yaml`, ajuste `bindings` e `arbiter` e use `--game <nome>`. Gestos disponíveis: `attack`, `use`, `jump`, `crouch`, `walk`, `sprint`, `walk_back`, `strafe_left`, `strafe_right` e `lean`.

### Segurança

| Mecanismo            | Comportamento                                                                  |
|----------------------|--------------------------------------------------------------------------------|
| **Perda de pose**    | Sem pose por 0,2 s, solta todo o input                                         |
| **Pose fantasma**    | Só vale com ombros e quadris visíveis e 1,0 s de rastreamento contínuo         |
| **Watchdog**         | Loop travado por mais de 0,5 s: uma thread separada solta tudo                 |
| **Teclas globais**   | `F8` e `F9` funcionam com qualquer janela em foco                              |
| **Saída do processo**| `release_all()` roda também no `atexit`; botões do mouse são sempre soltos     |

### Ferramentas

| Ferramenta       | Para quê                                                                 | Exemplo                                                                   |
|------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `download_model` | Baixa o modelo do MediaPipe                                              | `python tools/download_model.py --variant full`                           |
| `check_env`      | Valida Python 3.10, dependências, modelo `.task`, input e câmera         | `python -m tools.check_env`                                               |
| `bench_pose`     | Mede FPS e latência por variante                                         | `python -m tools.bench_pose --seconds 10`                                 |
| `guided`         | Sessão guiada em tela grande (**não envia input**); grava `.npz` e `.json` | `python -m tools.guided --person p1 --distance 2.0 --script full`       |
| `record`         | Grava landmarks rotulados por teclas (`1` soco D, `2` soco E, `3` pulo, `4` agachar, `5` inclinar) | `python -m tools.record --person p1 --lighting clara --out data/p1.npz` |
| `replay`         | Reproduz uma gravação pelo mesmo pipeline do app                         | `python -m tools.replay --recording sessao.npz`                           |
| `traceability`   | Tabela fórmula (A1–A9, R1–R7) → arquivo e função, para o relatório | `python -m tools.traceability --out reports/rastreabilidade.md` |
| `evaluate`       | Métricas por gesto (precisão, recall, FP/min), matriz de confusão e filtros | `python -m tools.evaluate --recordings "recordings/guided_*.npz"`      |

Roteiros do `guided`: `full`, `crouch`, `march` e `look`. O `evaluate` reconhece o roteiro de cada gravação e grava `metrics.md`, CSVs e gráficos em `reports/evaluate/`. Em `record`, `--lighting` aceita `clara`, `escura` ou `outra`. `F9` encerra e salva o que já foi gravado.

Estrutura do Código
```
InMotion/
├── config/
│   ├── default.yaml        → Configuração padrão
│   └── games/minecraft.yaml→ Perfil de jogo (gesto → entrada)
├── inmotion/
│   ├── __main__.py         → CLI (python -m inmotion)
│   ├── app.py              → Composição: loop, watchdog, janela de debug
│   ├── pipeline.py         → Passo puro por frame
│   ├── capture/            → Câmera (OpenCV)
│   ├── vision/             → Pose (MediaPipe), geometria, pré-processamento
│   ├── filters/            → EMA e One Euro
│   ├── gestures/           → Gestos, marcha, árbitro, fábrica
│   ├── input/              → Teclado/mouse e teclas globais
│   ├── config/             → Schema, loader e validação
│   ├── safety.py           → Segurança de input
│   ├── mapping.py          → Gesto → entrada
│   ├── overlay.py / history.py → Janela de debug e histórico do gráfico
│   ├── metrics.py          → Métricas por gesto (sessões guiadas)
│   └── recording.py / recorder.py / replay.py → Gravação e replay
├── tools/                  → check_env, download_model, bench_pose, guided, record, replay, evaluate, traceability, simulate
├── tests/                  → Testes e fixtures de sessões .npz
├── pyproject.toml          → Ruff, pytest e mypy
├── requirements.txt        → Dependências de runtime (versões fixas)
└── requirements-dev.txt    → Dependências de desenvolvimento
```

## Problemas Comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| Câmera não abre ou mostra outra imagem | Índice errado (câmeras virtuais, como Snap Camera, ocupam o `0`) ou outro app usando a webcam | Feche o outro app; teste `--camera 0`, `--camera 1`...; troque o backend com `--camera-backend msmf` |
| "SEM POSE" o tempo todo | Ombros e quadris fora do quadro ou pouca luz | Afaste-se até 1,5–2 m com o corpo inteiro visível; acenda uma luz à sua frente, não atrás |
| Imagem escura ou granulada | Exposição automática da webcam | Mais luz frontal; ou fixe a exposição: `--set camera.exposure=-6` (ajuste o valor) |
| FPS baixo (< 25) | Modelo pesado ou overlay completo | `--model-variant lite` (baixe antes com `python tools/download_model.py --variant lite`), tecla `o` para `minimal` ou `--no-overlay`; meça com `python -m tools.bench_pose` |
| Gesto reconhecido no overlay, mas nada acontece no jogo | Jogo sem foco, app pausado (`F8`) ou jogo rodando como administrador | Clique na janela do jogo; confira se não está PAUSADO; se o jogo roda como administrador, abra o terminal como administrador também |
| Câmera do jogo gira rápido demais ou dá trancos | Aceleração do mouse do Windows ou do jogo | No Minecraft, ligue *Raw Input* (Controles → Configurações do mouse); ajuste a velocidade com `--set head_look.yaw.gain=800` |
| Tecla parece presa | Tracking perdido no meio de um gesto | `F8` solta tudo; o app também solta sozinho ao perder a pose por 0,2 s |
| Agachar preso ou pulo sem resposta depois de mudar de lugar | Base medida em outra distância | Fique parado uns 4 s no lugar novo (o app mede de novo se a distância mudou muito) ou aperte `F8` duas vezes |
| Canhoto: os braços estão trocados | Lateralidade padrão é destro | `--set handedness=left` |

## Desenvolvimento

```bash
pip install -r requirements-dev.txt

ruff check .                           # lint
ruff format --check .                  # formatação
mypy                                   # tipos
pytest -m "not hardware and not slow"  # testes, como no CI
```

- `hardware`: precisa de webcam e/ou input real no Windows; fica de fora por padrão (`-m hardware`)
- `slow`: testes demorados; rodam por padrão (pule com `-m "not slow and not hardware"`; só `-m "not slow"` substitui o filtro padrão e roda os testes de hardware)
- Núcleo puro: sem `cv2`, `mediapipe`, `pydirectinput` ou `time.*` (o tempo entra como `t`)
- Sem `print` no app: use `logging`
- Limiares só mudam com base em gravações, nunca "no olho"

## Autores

- [IrvinMoura](https://github.com/IrvinMoura)
- [João Vítor](https://github.com/VihSousa)
- [Gabriel Moreira](https://github.com/Dhekki) 

## Licença

[MIT](LICENSE). As mudanças de cada versão estão no [CHANGELOG](CHANGELOG.md).
