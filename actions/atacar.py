import cv2
import math
import pydirectinput
from .base import Action


class Atacar(Action):
    def __init__(self, destro = True):
        super().__init__('left_click')
        self.braco_direito = destro

    def process(self, lmList, img):
        # 1. Definir pontos
        if self.braco_direito:
            p_ombro, p_cotovelo, p_pulso, p_quadril = 12, 14, 16, 24
        else:
            p_ombro, p_cotovelo, p_pulso, p_quadril = 11, 13, 15, 23

        # 2. Pegar coordenadas
        x_ombro, y_ombro = lmList[p_ombro][1], lmList[p_ombro][2]
        x_cotovelo, y_cotovelo = lmList[p_cotovelo][1], lmList[p_cotovelo][2]
        x_pulso, y_pulso = lmList[p_pulso][1], lmList[p_pulso][2]
        x_quadril, y_quadril = lmList[p_quadril][1], lmList[p_quadril][2]

        # 3. Desenho Visual
        cv2.line(img, (x_ombro, y_ombro), (x_cotovelo, y_cotovelo), (255, 255, 255), 3)
        cv2.line(img, (x_pulso, y_pulso), (x_cotovelo, y_cotovelo), (255, 255, 255), 3)
        cv2.circle(img, (x_cotovelo, y_cotovelo), 8, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x_pulso, y_pulso), 8, (0, 255, 255), cv2.FILLED)

        # Linha da Cintura
        h, w, _ = img.shape
        cv2.line(img, (0, y_quadril), (w, y_quadril), (0, 0, 255), 1)

        # 4. NOVA LÓGICA: COMPRESSÃO VISUAL (DISTÂNCIA TOTAL)

        # Calcula a distância real em pixels entre Cotovelo e Punho
        # Se o braço estiver esticado pra frente, essa distância será PEQUENA
        comprimento_antebraço = math.hypot(x_cotovelo - x_pulso, y_cotovelo - y_pulso)

        # Calcula o tamanho do braço (ombro até cotovelo) para usar como referência
        # Isso ajuda se você chegar perto ou longe da câmera
        comprimento_braço = math.hypot(x_ombro - x_cotovelo, y_ombro - y_cotovelo)

        # Evita divisão por zero
        if comprimento_braço == 0: comprimento_braço = 1

        # RELAÇÃO: Quanto do antebraço estamos vendo?
        # Se ratio < 0.6, o antebraço está "escondido" pela perspectiva (apontando pra frente)
        ratio = comprimento_antebraço / comprimento_braço

        # Debug na tela
        cv2.putText(img, f'Ratio: {ratio:.2f}', (x_cotovelo - 60, y_cotovelo + 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        # --- REGRAS DO SOCO ---
        # 1. Punho acima da linha da cintura
        punho_alto = y_pulso < y_quadril

        # 2. O antebraço deve parecer curto (ratio baixo)
        # Ajuste esse 0.60 se necessário. Menor = precisa esticar mais perfeito. Maior = mais sensível.
        soco_detectado = ratio < 0.60

        if punho_alto and soco_detectado:
            cv2.putText(img, "SOCO FRONTAL!", (x_pulso - 50, y_pulso - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.circle(img, (x_pulso, y_pulso), 15, (0, 255, 0), cv2.FILLED)

            if not self.is_active:
                pydirectinput.mouseDown()
                self.is_active = True
        else:
            if self.is_active:
                pydirectinput.mouseUp()
                self.is_active = False