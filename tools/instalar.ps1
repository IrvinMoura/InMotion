# Instalador do InMotion (Release da Entrega 3).

# Chamado pelo Instalar.bat. Prepara tudo na pasta do projeto, sem mexer no resto do PC:

# Python 3.10 (se faltar, oferece instalar pelo winget), ambiente .venv, dependências com a

# voz, modelo de pose, modelo de voz em português e um atalho opcional na área de trabalho.

# Pode rodar de novo a qualquer momento: o que já está pronto é pulado.

#

#   -Sim        responde "sim" às perguntas (instalação sem interação)

#   -SemAtalho  não cria o atalho na área de trabalho

param([switch]$Sim, [switch]$SemAtalho)



$ErrorActionPreference = "Stop"

$ProgressPreference = "SilentlyContinue"  # a barra do Invoke-WebRequest deixa o download lento

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}



$Raiz = Split-Path -Parent $PSScriptRoot

Set-Location $Raiz

New-Item -ItemType Directory -Force "logs", "models" | Out-Null

Start-Transcript -Path (Join-Path $Raiz "logs\instalar.log") -Force | Out-Null



$VoskNome = "vosk-model-small-pt-0.3"

$VoskUrl = "https://alphacephei.com/vosk/models/$VoskNome.zip"

$Total = 6



function Passo([int]$n, [string]$texto) {

    Write-Host ""

    Write-Host "[$n/$Total] $texto" -ForegroundColor Cyan

}



function Falha([string]$mensagem) {

    Write-Host ""

    Write-Host "ERRO: $mensagem" -ForegroundColor Red

    Write-Host "Detalhes em logs\instalar.log" -ForegroundColor Red

    Stop-Transcript | Out-Null

    exit 1

}



function Pergunta([string]$texto) {

    if ($Sim) { return $true }

    return ((Read-Host "$texto (S/N)") -match "^[sS]")

}



function Achar-Python310 {

    # Devolve o comando (lista) que roda o Python 3.10, ou $null.
    # "Continue": no PowerShell 5.1, o aviso do py sem o 3.10 (stderr com 2>$null) viraria erro fatal.
    $ErrorActionPreference = "Continue"

    if (Get-Command py -ErrorAction SilentlyContinue) {

        & py -3.10 -c "import sys" 2>$null

        if ($LASTEXITCODE -eq 0) { return @("py", "-3.10") }

    }

    $candidatos = @(

        "python3.10",

        "python",

        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe")

    )

    foreach ($c in $candidatos) {

        if (Get-Command $c -ErrorAction SilentlyContinue) {

            $v = & $c -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null

            if ($v -eq "3.10") { return @($c) }

        }

    }

    return $null

}



Write-Host "Instalador do InMotion" -ForegroundColor Green

Write-Host "Pasta: $Raiz"



# 1 ---------------------------------------------------------------- Python 3.10

Passo 1 "Procurando o Python 3.10"

$Py = Achar-Python310

if (-not $Py) {

    Write-Host "O InMotion precisa do Python 3.10, que não foi encontrado neste PC."

    if ((Get-Command winget -ErrorAction SilentlyContinue) -and (Pergunta "Instalar o Python 3.10 agora pelo winget?")) {

        winget install -e --id Python.Python.3.10 --scope user

        $Py = Achar-Python310

    }

    if (-not $Py) {

        Falha "Instale o Python 3.10 (https://www.python.org/downloads/) e rode o Instalar.bat de novo."

    }

}

Write-Host ("Usando: " + ($Py -join " "))



# 2 ---------------------------------------------------------------- ambiente .venv

Passo 2 "Criando o ambiente do InMotion (.venv)"

$VPy = Join-Path $Raiz ".venv\Scripts\python.exe"

if (Test-Path $VPy) {

    Write-Host "Já existe; mantido."

} else {

    $cmd = $Py[0]

    $args0 = @()

    if ($Py.Count -gt 1) { $args0 = $Py[1..($Py.Count - 1)] }

    & $cmd @args0 -m venv .venv

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VPy)) { Falha "Não consegui criar o ambiente .venv." }

}



# 3 ---------------------------------------------------------------- dependências

Passo 3 "Instalando as dependências, com a voz (pode levar alguns minutos)"

& $VPy -m pip install --disable-pip-version-check --no-input -r requirements-voice.txt

if ($LASTEXITCODE -ne 0) { Falha "A instalação das dependências falhou (internet?)." }



# 4 ---------------------------------------------------------------- modelo de pose

Passo 4 "Baixando o modelo de pose (MediaPipe)"

if (Test-Path "models\pose_landmarker_full.task") {

    Write-Host "Já existe; mantido."

} else {

    & $VPy tools\download_model.py --variant full

    if ($LASTEXITCODE -ne 0) { Falha "Não consegui baixar o modelo de pose." }

}



# 5 ---------------------------------------------------------------- modelo de voz

Passo 5 "Baixando o modelo de voz em português (Vosk, ~31 MB)"

$VoskDir = Join-Path $Raiz "models\$VoskNome"

if (Test-Path $VoskDir) {

    Write-Host "Já existe; mantido."

} else {

    $zip = Join-Path $Raiz "models\$VoskNome.zip"

    try {

        Invoke-WebRequest -Uri $VoskUrl -OutFile $zip -UseBasicParsing

        Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $Raiz "models") -Force

        Remove-Item $zip -Force

    } catch {

        Falha "Não consegui baixar o modelo de voz: $($_.Exception.Message)"

    }

    if (-not (Test-Path $VoskDir)) { Falha "O modelo de voz não ficou em models\$VoskNome." }

}



# 6 ---------------------------------------------------------------- conferência

Passo 6 "Conferindo a instalação"

& $VPy -m tools.check_env --skip-camera

if ($LASTEXITCODE -ne 0) { Falha "A conferência encontrou um problema (veja a tabela acima)." }

& $VPy -c "import vosk, sounddevice, tkinter; print('Voz e menu: OK')"

if ($LASTEXITCODE -ne 0) { Falha "A voz ou o menu não carregaram." }



if (-not $SemAtalho -and (Pergunta "Criar um atalho do InMotion na área de trabalho?")) {

    $desktop = [Environment]::GetFolderPath("Desktop")

    $shell = New-Object -ComObject WScript.Shell

    $atalho = $shell.CreateShortcut((Join-Path $desktop "InMotion.lnk"))

    $atalho.TargetPath = Join-Path $Raiz "InMotion.bat"

    $atalho.WorkingDirectory = $Raiz

    $atalho.WindowStyle = 7  # minimizado: o .bat só abre o menu e fecha

    $atalho.Description = "InMotion: jogue com o corpo usando a câmera"

    $atalho.Save()

    Write-Host "Atalho criado na área de trabalho."

}



Write-Host ""

Write-Host "Pronto! Para jogar, dê duplo clique em InMotion.bat (ou no atalho)." -ForegroundColor Green

Stop-Transcript | Out-Null

exit 0

