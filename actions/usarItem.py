import cv2
import math
import pydirectinput
from .base import Action


class UsarItem(Action):
    def __init__(self, destro=True):
        # A ação é 'right_click' (botão direito)
        # O pydirectinput lida com botões do mouse de forma específica
        super().__init__('right_click')

        # Lógica de Espelhamento:
        # Se o usuário é DESTRO (True), ele ataca com a Direita e usa item com a ESQUERDA.
        # Se o usuário é CANHOTO (False), ele ataca com a Esquerda e usa item com a DIREITA.
        self.usar_mao_direita = not destro

    def process(self, lmList, img):
        # 1. Definir pontos baseado na mão configurada (mão oposta à dominante)
        if self.usar_mao_direita:
            # Usa braço DIREITO
            p_ombro, p_cotovelo, p_pulso, p_quadril = 12, 14, 16, 24
            cor = (255, 100, 0)  # Azulado para diferenciar
        else:
            # Usa braço ESQUERDO (Padrão para destros)
            p_ombro, p_cotovelo, p_pulso, p_quadril = 11, 13, 15, 23
            cor = (255, 100, 0)

        # 2. Pegar coordenadas
        x_ombro, y_ombro = lmList[p_ombro][1], lmList[p_ombro][2]
        x_cotovelo, y_cotovelo = lmList[p_cotovelo][1], lmList[p_cotovelo][2]
        x_pulso, y_pulso = lmList[p_pulso][1], lmList[p_pulso][2]
        x_quadril, y_quadril = lmList[p_quadril][1], lmList[p_quadril][2]

        # 3. Desenho Visual (Cor diferente para facilitar: Azul/Laranja)
        cv2.line(img, (x_ombro, y_ombro), (x_cotovelo, y_cotovelo), (255, 255, 255), 3)
        cv2.line(img, (x_pulso, y_pulso), (x_cotovelo, y_cotovelo), (255, 255, 255), 3)
        cv2.circle(img, (x_cotovelo, y_cotovelo), 8, cor, cv2.FILLED)
        cv2.circle(img, (x_pulso, y_pulso), 8, cor, cv2.FILLED)

        # 4. LÓGICA DE ATIVAÇÃO (Mesma do soco: Extensão visual)

        # Distância Euclidiana (Pitágoras)
        comprimento_antebraço = math.hypot(x_cotovelo - x_pulso, y_cotovelo - y_pulso)
        comprimento_braço = math.hypot(x_ombro - x_cotovelo, y_ombro - y_cotovelo)

        if comprimento_braço == 0: comprimento_braço = 1

        ratio = comprimento_antebraço / comprimento_braço

        # Regras:
        punho_alto = y_pulso < y_quadril
        ativado = ratio < 0.60

        # Debug na tela
        cv2.putText(img, f'Item Ratio: {ratio:.2f}', (x_cotovelo - 60, y_cotovelo + 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, cor, 2)

        if punho_alto and ativado:
            cv2.putText(img, "USANDO ITEM!", (x_pulso - 50, y_pulso - 50), cv2.FONT_HERSHEY_PLAIN, 2, cor, 3)

            if not self.is_active:
                # Botão DIREITO do mouse
                pydirectinput.mouseDown(button='right')
                self.is_active = True
        else:
            if self.is_active:
                pydirectinput.mouseUp(button='right')
                self.is_active = False