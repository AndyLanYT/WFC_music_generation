from abc import ABC, abstractmethod


class InterfaceClick(ABC):
    @abstractmethod
    def check_click(self):
        pass
