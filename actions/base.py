import abc


class Action(abc.ABC):
    def __init__(self, key_code):
        self.key_code = key_code
        self.is_active = False

    @abc.abstractmethod
    def process(self, lmList, img):
        """
        Recebe a lista de landmarks e a imagem.
        Deve conter  a lógica matemática do movimento.
        Retorna True se o movimento foi detectado, False caso contrário.

        """
        pass
