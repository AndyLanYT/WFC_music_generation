from abc import ABC, abstractmethod


class IRenderable(ABC):
    @abstractmethod
    def render(self, screen, *args, **kwargs):
        pass
