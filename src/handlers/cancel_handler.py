# handlers/cancel_handler.py
from telegram import Update, ReplyKeyboardRemove
from telegram.ext import CallbackContext, ConversationHandler

from src.handlers.base_handler import BaseHandler

class CancelHandler(BaseHandler):
    """Handler for the /cancel command."""

    def __init__(self, logger):
        self.logger = logger

    def handle(self, update: Update, context: CallbackContext):
        self.logger.debug("Cancel handler triggered")
        update.message.reply_text("Operation cancelled.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END