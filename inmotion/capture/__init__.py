"""Captura de vídeo (adaptador): câmera em thread com o frame mais recente."""

# Padrão provisório: no PC de desenvolvimento o índice 0 é uma câmera virtual (Snap Camera)
# e a webcam real é a 1. A escolha da câmera vai para o YAML na spec 004.
DEFAULT_CAMERA_INDEX = 1
