import cv2
import pydirectinput
from .base import Action

class Pular(Action):
    def __init__(self):
        # Tecla pular é "space"
        super().__init__("space")
        self.y_quadril_base = None # Vai guardar a altura "chão"
        self.calibrado = False
        self.sensibilidade = 50 # Quanto pixels subir para considerar pulo

    def process(self, lmList, img):
        # Quadril: 23 (esq) 24(dir)
        x1, y1 = lmList[23][1], lmList[23][2]
        x2, y2 = lmList[24][1], lmList[24][2]
        centro_quadril_y = (y1 + y2) // 2
        centro_quadril_x = (x1 + x2) // 2

        # Desenhar bolinha para visualização
        cv2.circle(img, (centro_quadril_x, centro_quadril_y), 10, (0, 0, 255), cv2.FILLED)

        # --- CALIBRAÇÃO ---
        # Se ainda não calibrou, define a altura atual como o "chão" e PARA AQUI.
        if not self.calibrado:
            self.y_quadril_base = centro_quadril_y
            self.calibrado = True
            print(f"Altura base calibrada em: {self.y_quadril_base}")
            return  # <--- Este return é essencial para não dar erro na linha de baixo na primeira vez

        # --- SEGURANÇA EXTRA ---
        # Se por algum motivo a base ainda for None, não faz nada
        if self.y_quadril_base is None:
            return

        # Lógica do Pulo
        # Se a altura atual for MENOR que ( Base - Sensibilidade ) considera pulo
        isJumping = centro_quadril_y < (self.y_quadril_base - self.sensibilidade)

        if isJumping:
            cv2.putText(img, "PULO DETECTADO", (50,200), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
            if not self.is_active:
                pydirectinput.keyDown(self.key_code)
                self.is_active = True
        else:
            # Se voltou pro chão e a tecla tava apertada, solta
            if self.is_active:
                pydirectinput.keyUp(self.key_code)
                self.is_active = False

                # Recalibragem pra evitar drift
                if abs(centro_quadril_y - self.y_quadril_base) < 20:
                    self.y_quadril_base = centro_quadril_y
