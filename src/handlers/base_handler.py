# handlers/base_handler.py
from abc import ABC, abstractmethod
from telegram import Update
from telegram.ext import CallbackContext

class BaseHandler(ABC):
    """Abstract base class for all handlers."""

    @abstractmethod
    def handle(self, update: Update, context: CallbackContext):
        pass