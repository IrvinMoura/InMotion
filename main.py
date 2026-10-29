import cv2
import time
import PoseModule as pm
from actions.pular import Pular
from actions.atacar import Atacar
from actions.usarItem import UsarItem

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
USUARIO_DESTRO = True

# Lista de ações ativas no jogo
acoes = [
    Pular(),
    Atacar(destro=USUARIO_DESTRO),
    UsarItem(destro=USUARIO_DESTRO),
]

print("Fique parado em frente a câmera para calibração inicial...")
time.sleep(2)

while True:
    success, img = cap.read()
    if not success: break

    # 1. Detectar Corpo
    img = detector.findPose(img, altura_desejada = 720)
    lmList = detector.findPosition(img, draw=False)

    # 2. Processar Ações
    if len(lmList) != 0:
        for acao in acoes:
            acao.process(lmList, img)

    # 3. FPS e Display
    cTime = time.time()
    try:
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    except:
        fps = 0
    pTime = cTime

    # Mostra se está no modo destro ou canhoto na tela
    modo_str = "DESTRO" if USUARIO_DESTRO else "CANHOTO"
    cv2.putText(img, f'Modo: {modo_str}', (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 2)

    cv2.imshow("KinectProject", img)

    # Tecla de segurança para fechar (q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()