from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, question: str, context: str = "", stream: bool = False) -> str:
        pass