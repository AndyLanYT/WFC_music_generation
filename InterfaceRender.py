from abc import ABC, abstractmethod


class InterfaceRender(ABC):
    @abstractmethod
    def render(self, screen, *args, **kwargs):
        pass
