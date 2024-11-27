# handlers/start_handler.py
from telegram import Update
from telegram.ext import CallbackContext

from src.handlers.base_handler import BaseHandler

class StartHandler(BaseHandler):
    """Handler for the /start command."""

    def __init__(self, logger):
        self.logger = logger

    def handle(self, update: Update, context: CallbackContext):
        self.logger.debug("Start command received")
        update.message.reply_text(
            "Hi! I can help you manage your Google Calendar events.\n\n"
            "You can:\n"
            "- Add a new event by sending event details.\n"
            "- Delete an event by sending a command like 'Delete [event name]'.\n"
            "- Reschedule an event by sending a command like 'Reschedule [event name]'.\n"
            "- Send audio messages with event details or commands."
        )
        return 'PARSE_INPUT'