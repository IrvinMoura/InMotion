# InMotion - Controle de Jogos por Movimento (Projeto A3)

Jogue com o corpo, usando apenas uma webcam: o InMotion reconhece gestos e os converte em teclas e cliques do mouse. O perfil incluído é o do **Minecraft**.

> **Status:** em desenvolvimento (versão `0.0.0`). Os limiares dos gestos são valores iniciais, calibrados com gravações de sessões guiadas.

### Núcleo do Sistema
| Módulo                     | Descrição                                                         |
|----------------------------|-------------------------------------------------------------------|
| **Captura**                | Lê frames da webcam via OpenCV (MJPG, 640x480, até 60 FPS)        |
| **Pose**                   | Estima os landmarks do corpo com o MediaPipe PoseLandmarker       |
| **Gestos e Árbitro**       | Reconhece socos, pulo, agachar e marcha; resolve conflitos        |
| **Segurança**              | Garante que nenhuma tecla ou botão fique preso                    |
| **Ferramentas**            | Gravação, replay, sessões guiadas e benchmark para calibração     |

### Gestos e Controles (Minecraft)

| Ação no jogo       | Gesto                                   | Entrada                 | Tipo   |
|--------------------|-----------------------------------------|-------------------------|--------|
| Atacar / quebrar   | Braço dominante esticado para a câmera  | Botão esquerdo do mouse | segura |
| Usar / colocar     | Braço não dominante esticado            | Botão direito do mouse  | segura |
| Pular              | Subida do quadril                       | `Espaço`                | toque  |
| Agachar            | Ombros abaixo da base calibrada         | `Shift`                 | segura |
| Andar              | Marcha no lugar (2 passos alternados)   | `W`                     | segura |
| Correr             | Marcha rápida (soma ao andar)           | `Ctrl`                  | segura |

- Com **os dois braços esticados juntos**, os cliques são suprimidos (combo `both_arms`)
- **Agachar** vence pular, andar e correr
- **Modo canhoto:** `--set handedness=left` troca os braços
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

# Crie um ambiente virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Baixe o modelo de pose (lite | full | heavy)
python tools/download_model.py --variant full
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

Há também uma zona de saída provisória: manter a mão no retângulo do canto superior esquerdo da imagem por 4 s encerra o app.

> O índice de câmera padrão é `1` (provisório, no `config/default.yaml`). Se a câmera não abrir, use `--camera 0`.

### Opções da Linha de Comando

| Opção                  | Descrição                                                                  |
|------------------------|----------------------------------------------------------------------------|
| `--config ARQUIVO`     | YAML base, no lugar de `config/default.yaml`                               |
| `--game NOME`          | Perfil em `config/games/NOME.yaml` (ex.: `minecraft`)                      |
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

Para outro jogo, copie `config/games/minecraft.yaml`, ajuste `bindings` e `arbiter` e use `--game <nome>`. Gestos disponíveis: `attack`, `use`, `jump`, `crouch`, `walk`, `sprint` e `lean`.

### Segurança

| Mecanismo            | Comportamento                                                                  |
|----------------------|--------------------------------------------------------------------------------|
| **Perda de pose**    | Sem pose por 0,2 s, solta todo o input                                         |
| **Pose fantasma**    | Só vale com ombros e quadris visíveis e 1,0 s de rastreamento contínuo         |
| **Watchdog**         | Loop travado por mais de 0,5 s: uma thread separada solta tudo                 |
| **Teclas globais**   | `F8` e `F9` funcionam com qualquer janela em foco                              |
| **Saída do processo**| `release_all()` roda também no `atexit`; botões do mouse são sempre soltos     |

### Ferramentas de Calibração

| Ferramenta       | Para quê                                                                 | Exemplo                                                                   |
|------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `download_model` | Baixa o modelo do MediaPipe                                              | `python tools/download_model.py --variant full`                           |
| `bench_pose`     | Mede FPS e latência por variante                                         | `python -m tools.bench_pose --seconds 10`                                 |
| `guided`         | Sessão guiada em tela grande (**não envia input**); grava `.npz` e `.json` | `python -m tools.guided --person p1 --distance 2.0 --script full`       |
| `record`         | Grava landmarks rotulados por teclas (`1` soco D, `2` soco E, `3` pulo, `4` agachar, `5` inclinar) | `python -m tools.record --person p1 --lighting clara --out data/p1.npz` |
| `replay`         | Reproduz uma gravação pelo mesmo pipeline do app                         | `python tools/replay.py --recording sessao.npz`                           |

Roteiros do `guided`: `full`, `crouch` e `march`. Em `record`, `--lighting` aceita `clara`, `escura` ou `outra`. `F9` encerra e salva o que já foi gravado.

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
│   └── recording.py / recorder.py / replay.py → Gravação e replay
├── tools/                  → download_model, bench_pose, guided, record, replay
├── tests/                  → Testes e fixtures de sessões .npz
├── pyproject.toml          → Ruff, pytest e mypy
├── requirements.txt        → Dependências de runtime (versões fixas)
└── requirements-dev.txt    → Dependências de desenvolvimento
```

## Desenvolvimento

```bash
pip install -r requirements-dev.txt

ruff check .                           # lint
ruff format --check .                  # formatação
mypy                                   # tipos
pytest -m "not hardware and not slow"  # testes, como no CI
```

- `hardware`: precisa de webcam e/ou input real no Windows; fica de fora por padrão (`-m hardware`)
- `slow`: testes demorados; rodam por padrão (pule com `-m "not slow"`)
- Núcleo puro: sem `cv2`, `mediapipe`, `pydirectinput` ou `time.*` (o tempo entra como `t`)
- Sem `print` no app: use `logging`
- Limiares só mudam com base em gravações, nunca "no olho"

## Autores

- [IrvinMoura](https://github.com/IrvinMoura)
- [João Vítor](https://github.com/VihSousa)
- [Gabriel Moreira](https://github.com/Dhekki) 

## Licença

Defina a licença do projeto (nenhum arquivo `LICENSE` foi encontrado no repositório).
