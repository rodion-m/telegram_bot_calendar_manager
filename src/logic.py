import base64
import datetime
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

import litellm
import pytz
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from tzlocal import get_localzone

SEARCH_MODEL = "gemini/gemini-1.5-pro-002"
COMMANDS_MODEL_VOICE = "gemini/gemini-1.5-pro-002"
COMMANDS_MODEL_TEXT = "gemini/gemini-1.5-pro-002"
#COMMANDS_MODEL_TEXT = "openai/gpt-4o"

import os
from dotenv import load_dotenv


class Config:
    """Configuration management using environment variables."""

    def __init__(self, env_file: str = ".env"):
        load_dotenv(env_file)
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.LITELLM_LOG = os.getenv("LITELLM_LOG", "INFO")
        self.SCOPES = ['https://www.googleapis.com/auth/calendar.events']
        self.CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "../google_credentials.json")


from abc import ABC, abstractmethod
from telegram import Update
from telegram.ext import CallbackContext


class BaseHandler(ABC):
    """Abstract base class for all handlers."""

    @abstractmethod
    def handle(self, update: Update, context: CallbackContext):
        pass

from telegram import Message
from typing import Optional


def download_audio(message: Message, user_id: int, logger) -> Optional[str]:
    """Downloads audio from a Telegram message."""
    try:
        if message.voice:
            file = message.voice.get_file()
            audio_path = f"audio_{user_id}.ogg"
        elif message.audio:
            file = message.audio.get_file()
            audio_path = f"audio_{user_id}.{message.audio.mime_type.split('/')[-1]}"
        else:
            return None
        file.download(audio_path)
        logger.debug(f"Downloaded audio to {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        return None


class CalendarEvent(BaseModel):
    name: str = Field(..., description="Name of the event")
    date: str = Field(..., description="Date of the event in YYYY-MM-DD format")
    time: str = Field(..., description="Time of the event in HH:MM (24-hour) format")
    timezone: str = Field(..., description="IANA timezone string")
    description: Optional[str] = Field(None, description="Description of the event")
    meeting_link: Optional[str] = Field(None, description="Link to the meeting if applicable")


class EventIdentifier(BaseModel):
    identifier: str = Field(..., description="Identifier for the event, can be name or description")


class RescheduleDetails(BaseModel):
    identifier: str = Field(..., description="Identifier for the event to reschedule")
    new_date: Optional[str] = Field(None, description="New date in YYYY-MM-DD format")
    new_time: Optional[str] = Field(None, description="New time in HH:MM (24-hour) format")
    new_timezone: Optional[str] = Field(None, description="New IANA timezone string")

class GoogleCalendarService:
    """Service to interact with Google Calendar API."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.SCOPES = self.config.SCOPES

    def get_credentials(self, user_id: int) -> Credentials:
        creds = None
        # Create tokens directory if it doesn't exist
        if not os.path.exists('../google_tokens'):
            os.makedirs('../google_tokens')
        token_path = f'../google_tokens/token_{user_id}.json'  # Unique token file per user
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.config.CREDENTIALS_FILE, self.SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        return creds

    def create_event(self, event: CalendarEvent, user_id: int) -> Dict:
        """Creates an event in Google Calendar."""
        creds = self.get_credentials(user_id)
        service = build('calendar', 'v3', credentials=creds)

        # Combine date and time
        start_datetime_str = f"{event.date} {event.time}"

        timezone = event.timezone
        try:
            start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
        except ValueError as e:
            self.logger.error(f"Date/time format error: {e}")
            raise e

        start_datetime = pytz.timezone(timezone).localize(start_datetime)
        end_datetime = start_datetime + timedelta(hours=1)  # Default duration 1 hour

        # Prepare event body
        event_body = {
            'summary': event.name,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': timezone,
            },
        }

        # Add optional description
        if event.description:
            event_body['description'] = event.description

        # Add meeting link to description
        if event.meeting_link:
            if 'description' in event_body:
                event_body['description'] += f"\n\nMeeting Link: {event.meeting_link}"
            else:
                event_body['description'] = f"Meeting Link: {event.meeting_link}"

        created_event = service.events().insert(calendarId='primary', body=event_body).execute()
        self.logger.debug(f"Created event: {created_event}")
        return created_event

    def delete_event(self, identifier: str, user_id: int) -> Dict:
        """Deletes an event from Google Calendar based on identifier using LLM for relevance."""
        creds = self.get_credentials(user_id)
        service = build('calendar', 'v3', credentials=creds)

        # Fetch events to find the most relevant one
        now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        events_result = service.events().list(
            calendarId='primary', timeMin=now,
            maxResults=50, singleEvents=True,
            orderBy='startTime').execute()
        events = events_result.get('items', [])

        # Utilize LLM to find the most relevant event based on identifier
        relevant_event = self.find_relevant_event_with_llm(identifier, events)
        if relevant_event:
            event_id = relevant_event['id']
            service.events().delete(calendarId='primary', eventId=event_id).execute()
            self.logger.debug(f"Deleted event: {relevant_event}")
            return {"status": "deleted", "event": relevant_event}
        self.logger.debug("No matching event found to delete.")
        return {"status": "not_found"}

    def reschedule_event(self, details: Dict, user_id: int) -> Dict:
        """Reschedules an existing event based on identifier and new details using LLM for relevance."""
        identifier = details.get('identifier')
        new_date = details.get('new_date')
        new_time = details.get('new_time')
        new_timezone = details.get('new_timezone')

        creds = self.get_credentials(user_id)
        service = build('calendar', 'v3', credentials=creds)

        # Fetch events to find the most relevant one
        now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        events_result = service.events().list(
            calendarId='primary', timeMin=now,
            maxResults=50, singleEvents=True,
            orderBy='startTime').execute()
        events = events_result.get('items', [])

        # Utilize LLM to find the most relevant event based on identifier
        relevant_event = self.find_relevant_event_with_llm(identifier, events)
        if relevant_event:
            event_id = relevant_event['id']
            # Update event details
            if new_date and new_time and new_timezone:
                start_datetime_str = f"{new_date} {new_time}"
                try:
                    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError as e:
                    self.logger.error(f"Date/time format error: {e}")
                    return {"status": "error", "message": "Invalid date/time format."}

                start_datetime = pytz.timezone(new_timezone).localize(start_datetime)
                end_datetime = start_datetime + timedelta(hours=1)  # Default duration

                relevant_event['start']['dateTime'] = start_datetime.isoformat()
                relevant_event['start']['timeZone'] = new_timezone
                relevant_event['end']['dateTime'] = end_datetime.isoformat()
                relevant_event['end']['timeZone'] = new_timezone

                updated_event = service.events().update(
                    calendarId='primary', eventId=event_id, body=relevant_event).execute()
                self.logger.debug(f"Rescheduled event: {updated_event}")
                return {"status": "rescheduled", "event": updated_event}
        self.logger.debug("No matching event found to reschedule.")
        return {"status": "not_found"}

    def find_relevant_event_with_llm(self, identifier: str, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Use LLM to determine the most relevant event based on the identifier."""
        if not events:
            self.logger.debug("No events found in the calendar.")
            return None

        # Prepare messages for LLM
        system_prompt = (
            "You are an assistant that helps identify the most relevant event based on a given identifier. "
            "Given a list of events with their summaries and descriptions, determine which event best matches the identifier."
        )
        user_prompt = f"Identifier: {identifier}\n\nEvents:\n" + "\n".join(
            [f"- {event.get('summary', 'No Title')}: {event.get('description', 'No Description')}" for event in events]
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Make the completion request
        try:
            response = litellm.completion(
                model=SEARCH_MODEL,
                messages=messages,
                tools=[],  # No function calling needed here
            )
            chosen_event_summary = response.choices[0].message['content'].strip().lower()
            self.logger.debug(f"LLM chose event summary: {chosen_event_summary}")

            # Find the event that matches the chosen summary
            for event in events:
                if event.get('summary', '').lower() == chosen_event_summary:
                    return event
        except Exception as e:
            self.logger.error(f"Error in LLM for finding relevant event: {e}")
        return None


class LiteLLMService:
    """Service to interact with LiteLLM for function calling."""

    def __init__(self, config: Config, logger: logging.Logger, google_calendar_service: GoogleCalendarService):
        self.config = config
        self.logger = logger
        self.google_calendar_service = google_calendar_service
        # Define function schemas for function calling
        self.functions = [
            {
                "type": "function",
                "function": {
                    "name": "add_event",
                    "description": "Add a new event to Google Calendar.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the event"},
                            "date": {"type": "string", "description": "Date of the event in YYYY-MM-DD format"},
                            "time": {"type": "string", "description": "Time of the event in HH:MM (24-hour) format"},
                            "timezone": {"type": "string", "description": "IANA timezone string if applicable"},
                            "description": {"type": "string", "description": "Description of the event if applicable"},
                            "meeting_link": {"type": "string",
                                             "description": "Link to the meeting and its passcode if applicable"},
                        },
                        "required": ["name", "date", "time", "timezone"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_event",
                    "description": "Delete an existing event from Google Calendar.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "identifier": {"type": "string",
                                           "description": "Identifier for the event, can be name or description"}
                        },
                        "required": ["identifier"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reschedule_event",
                    "description": "Reschedule an existing event in Google Calendar.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "identifier": {"type": "string", "description": "Identifier for the event to reschedule"},
                            "new_date": {"type": "string", "description": "New date in YYYY-MM-DD format"},
                            "new_time": {"type": "string", "description": "New time in HH:MM (24-hour) format"},
                            "new_timezone": {"type": "string", "description": "New IANA timezone string"},
                        },
                        "required": ["identifier"]
                    }
                }
            }
        ]

    def get_completion(self, model: str, messages: List[Dict[str, Any]], tool_choice: str = "auto") -> Any:
        """Make a completion request to LiteLLM with function calling."""
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=self.functions,  # Pass the function schemas
            tool_choice=tool_choice,
            temperature=0
        )
        return response

    def parse_function_calls(self, response_message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract function calls from the model response."""
        return response_message.get('tool_calls', [])

    def execute_function(self, function_name: str, function_args: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """Map function calls to GoogleCalendarService methods."""
        self.logger.debug(f"Executing function '{function_name}' with args: {function_args}")
        if function_name == "add_event":
            try:
                event = CalendarEvent(**function_args)
                return self.google_calendar_service.create_event(event, user_id)
            except Exception as e:
                self.logger.error(f"Error in add_event: {e}")
                return {"error": str(e)}
        elif function_name == "delete_event":
            try:
                identifier = EventIdentifier(**function_args)
                return self.google_calendar_service.delete_event(identifier.identifier, user_id)
            except Exception as e:
                self.logger.error(f"Error in delete_event: {e}")
                return {"error": str(e)}
        elif function_name == "reschedule_event":
            try:
                details = RescheduleDetails(**function_args)
                return self.google_calendar_service.reschedule_event(details.model_dump(), user_id)
            except Exception as e:
                self.logger.error(f"Error in reschedule_event: {e}")
                return {"error": str(e)}
        else:
            self.logger.error(f"Unknown function called: {function_name}")
            return {"error": "Unknown function called."}


class InputHandler(BaseHandler):
    """Handler for parsing user input."""

    def __init__(self, logger: logging.Logger, litellm_service: LiteLLMService,
                 google_calendar_service: GoogleCalendarService):
        self.logger = logger
        self.litellm_service = litellm_service
        self.google_calendar_service = google_calendar_service

    def handle(self, update: Update, context: CallbackContext):
        self.logger.debug("Input handler triggered")
        message = update.message
        user_id = update.effective_user.id

        # Extract text from message.text or message.caption
        user_message = message.text_markdown_v2_urled if message.text_markdown_urled else message.caption_markdown_v2_urled
        self.logger.debug(f"Extracted user message: {user_message}")

        if not user_message and not message.voice and not message.audio:
            update.message.reply_text("Unsupported message type. Please send text, audio, or voice messages.")
            return ConversationHandler.END

        # Prepare the system prompt
        now = datetime.now().astimezone()
        system_prompt = f"""
            You are a smart calendar assistant. Your primary task is to help users manage their events efficiently by adding new events, deleting existing events, or rescheduling events in the user's calendar.
            
            When a user sends you a message, analyze it carefully to determine the appropriate action (adding, deleting, or rescheduling an event). Users may provide details such as the event name, date, time, timezone, and optional descriptions. They may also send commands in natural language, such as "Meeting with John tomorrow at 5 PM."
            
            If any event details are unclear, try to infer them from the user's message. If the timezone is not specified, use the user's timezone provided in the context.
            
            To perform an action, use the appropriate function (`add_event`, `delete_event`, or `reschedule_event`) with the necessary parameters. Be sure to use the functions exactly as they are defined, without modifying or extending them.
            
            Here's the current context:
            <context>
            Today's date and time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}
            User's Timezone: {get_localzone()}.
            Day of the week: {now.strftime('%A')}.
            </context>
            
            Here's the user's message:
        """

        # Build the message payload
        is_voice = message.voice or message.audio
        if is_voice:
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
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please process the following audio message and perform commands. Audio language: Russian"
                        },
                        {
                            "type": "image_url",
                            "image_url": "data:audio/ogg;base64,{}".format(encoded_audio),
                        }
                    ],
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
            model = COMMANDS_MODEL_VOICE if is_voice else COMMANDS_MODEL_TEXT
            response = self.litellm_service.get_completion(
                model=model,
                messages=messages,
                tool_choice="auto",
            )

            self.logger.debug(f"LLM Response:\n{response}")
            response_message = response['choices'][0]['message']
            tool_calls = self.litellm_service.parse_function_calls(response_message)

            if tool_calls:
                # A function needs to be called
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    function_args = json.loads(tool_call['function']['arguments'])

                    self.logger.debug(f"Function call detected: {function_name} with args: {function_args}")

                    # Execute the function using LiteLLMService
                    result = self.litellm_service.execute_function(function_name, function_args, user_id)

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
                                update.message.reply_text(
                                    f"Event added successfully! You can view it here: {event_link}")
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


class ConfirmationHandler(BaseHandler):
    """Handler for confirming event details."""

    def __init__(self, logger: logging.Logger, google_calendar_service: GoogleCalendarService):
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
                if not event:
                    update.message.reply_text("No event details found to confirm.")
                    return ConversationHandler.END
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


from telegram import Update, ReplyKeyboardRemove
from telegram.ext import CallbackContext, ConversationHandler


class CancelHandler(BaseHandler):
    """Handler for the /cancel command."""

    def __init__(self, logger):
        self.logger = logger

    def handle(self, update: Update, context: CallbackContext):
        self.logger.debug("Cancel handler triggered")
        update.message.reply_text("Operation cancelled.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END


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


import logging


class LoggerSetup:
    """Sets up the logging configuration."""

    @staticmethod
    def setup_logging(level: str = "DEBUG"):
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=getattr(logging, level.upper(), logging.DEBUG)
        )
        logger = logging.getLogger(__name__)
        return logger
