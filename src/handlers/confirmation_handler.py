# handlers/confirmation_handler.py
from telegram import Update
from telegram.ext import CallbackContext, ConversationHandler

from src.handlers.base_handler import BaseHandler
from src.models import CalendarEvent
from src.services.google_calendar_service import GoogleCalendarService

class ConfirmationHandler(BaseHandler):
    """Handler for confirming event details."""

    def __init__(self, logger, google_calendar_service: GoogleCalendarService):
        self.logger = logger
        self.google_calendar_service = google_calendar_service

    def handle(self, update: Update, context: CallbackContext):
        self.logger.debug("Confirmation handler triggered")
        response = update.message.text.lower()
        user_id = update.effective_user.id

        if response in ['yes', 'y']:
            # Proceed to add event to Google Calendar
            try:
                event = context.user_data.get('event', {})
                calendar_event = self.google_calendar_service.create_event(
                    CalendarEvent(**event), user_id)
                event_link = calendar_event.get('htmlLink')

                update.message.reply_text(f"Event added successfully! You can view it here: {event_link}")
            except Exception as e:
                self.logger.error(f"Error adding event to Google Calendar: {e}")
                update.message.reply_text("Sorry, there was an error adding the event to your Google Calendar.")
            return ConversationHandler.END
        elif response in ['no', 'n']:
            update.message.reply_text("Okay, let's start over. Please send me the event details.")
            return 'PARSE_INPUT'
        else:
            update.message.reply_text("Please respond with 'Yes' or 'No'. Is the event information correct?")
            return 'CONFIRMATION'