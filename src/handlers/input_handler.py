import json
import os
import base64
from datetime import datetime

from telegram import Update
from telegram.ext import CallbackContext, ConversationHandler

from src.handlers.base_handler import BaseHandler
from src.models import CalendarEvent, EventIdentifier, RescheduleDetails
from src.services.litellm_service import LiteLLMService
from src.services.google_calendar_service import GoogleCalendarService
from src.utils.helpers import download_audio

class InputHandler(BaseHandler):
    """Handler for parsing user input."""

    def __init__(self, logger, litellm_service: LiteLLMService, google_calendar_service: GoogleCalendarService):
        self.logger = logger
        self.litellm_service = litellm_service
        self.google_calendar_service = google_calendar_service

    def handle(self, update: Update, context: CallbackContext):
        self.logger.debug("Input handler triggered")
        message = update.message
        user_id = update.effective_user.id

        # Extract text from message.text or message.caption
        user_message = message.text if message.text else message.caption
        self.logger.debug(f"Extracted user message: {user_message}")

        if not user_message and not message.voice and not message.audio:
            update.message.reply_text("Unsupported message type. Please send text, audio, or voice messages.")
            return ConversationHandler.END

        # Prepare the system prompt
        now = datetime.now().astimezone()
        system_prompt = (
            "You are a smart calendar assistant. Your primary task is to help users manage their events efficiently. "
            "You can add new events, delete existing events, or reschedule events in the user's calendar. "
            "Users may provide details such as the event name, date, time, timezone, and optional descriptions. "
            "Users can also send commands in natural language, such as 'Meeting with John tomorrow at 5 PM.' "
            "When the user provides event details, analyze their message to determine the appropriate action, "
            "such as adding, deleting, or rescheduling an event. Always validate the information to ensure it is complete "
            "and meets the necessary requirements for the action. "
            "When ready, use the appropriate function (`add_event`, `delete_event`, or `reschedule_event`) and include the necessary parameters. "
            "Be accurate and concise in your communication, ensuring user satisfaction and clarity."
            "\n\n"
            f"Today's context:"
            f"\nToday's date and time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
            f"User's Timezone: {now.tzinfo}. "
            f"Day of the week: {now.strftime('%A')}."
        )

        # Build the message payload
        if message.voice or message.audio:
            # Handle voice or audio messages: download and transcribe
            audio_path = download_audio(message, user_id, self.logger)
            if not audio_path:
                update.message.reply_text("Failed to download audio.")
                return ConversationHandler.END

            # Read and encode the audio file
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
            self.logger.debug("Encoded audio to base64")

            # Remove the audio file after encoding
            os.remove(audio_path)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please process the following audio message and perform commands."},
                {
                    "role": "user",
                    "content": {
                        "type": "audio",
                        "content": encoded_audio,
                        "format": "ogg",
                    },
                },
            ]
        else:
            # Handle text or caption messages
            self.logger.debug(f"Received text/caption message: {user_message}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        try:
            # Make the completion request with function calling
            response = self.litellm_service.get_completion(
                model="openai/gpt-4o",
                messages=messages,
                tool_choice="auto",
            )

            response_message = response['choices'][0]['message']
            tool_calls = self.litellm_service.parse_function_calls(response_message)

            if tool_calls:
                # A function needs to be called
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    function_args = json.loads(tool_call['function']['arguments'])

                    self.logger.debug(f"Function call detected: {function_name} with args: {function_args}")

                    if function_name == "add_event":
                        event = CalendarEvent(**function_args)
                        result = self.google_calendar_service.create_event(event, user_id)
                    elif function_name == "delete_event":
                        identifier = EventIdentifier(**function_args)
                        result = self.google_calendar_service.delete_event(identifier.identifier, user_id)
                    elif function_name == "reschedule_event":
                        details = RescheduleDetails(**function_args)
                        result = self.google_calendar_service.reschedule_event(details.model_dump(), user_id)
                    else:
                        result = {"error": "Unknown function called."}

                    # Send the result back to the user
                    if 'error' in result:
                        update.message.reply_text(f"Error: {result['error']}")
                    else:
                        if result.get("status") == "deleted":
                            update.message.reply_text("Event deleted successfully.")
                        elif result.get("status") == "rescheduled":
                            update.message.reply_text("Event rescheduled successfully.")
                        else:
                            # For add_event, provide the event link
                            event_link = result.get("htmlLink")
                            if event_link:
                                update.message.reply_text(f"Event added successfully! You can view it here: {event_link}")
                            else:
                                update.message.reply_text("Action completed successfully.")

                return ConversationHandler.END
            else:
                update.message.reply_text("Sorry, I couldn't understand the request. Please try again.")
                return ConversationHandler.END

        except Exception as e:
            self.logger.error(f"Error with LiteLLM completion: {e}")
            update.message.reply_text("An error occurred while processing your request.")
            return ConversationHandler.END
