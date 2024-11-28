import base64
import datetime
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

import litellm
import pytz
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from tzlocal import get_localzone

from abc import ABC, abstractmethod
from telegram import Update, Message, ReplyKeyboardRemove
from telegram.ext import CallbackContext, ConversationHandler
import os
from dotenv import load_dotenv

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# Constants
SEARCH_MODEL = "gemini/gemini-1.5-flash-002"
COMMANDS_MODEL_VOICE = "gemini/gemini-1.5-pro-002"
COMMANDS_MODEL_TEXT = "gemini/gemini-1.5-pro-002"
# COMMANDS_MODEL_TEXT = "openai/gpt-4o"

# TODO: Reimplement reschedule_event feature, cause it's complex

class FallbacksModels:
    """Fallback models for LLM completion requests."""
    SearchFallbacks = "openai/gpt-4o-mini"
    CommandsFallbacks = "gemini/gemini-1.5-flash-002"


class BotStates:
    """States for the conversation handler."""
    PARSE_INPUT = 1
    CONFIRMATION = 2


class Config:
    """Configuration management using environment variables."""

    def __init__(self, env_file: str = ".env"):
        load_dotenv(env_file)
        self.GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.TELEGRAM_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        self.LITELLM_LOG: str = os.getenv("LITELLM_LOG", "INFO")
        self.SCOPES: List[str] = ['https://www.googleapis.com/auth/calendar.events']
        self.CREDENTIALS_FILE: str = os.getenv("GOOGLE_CREDENTIALS_FILE", "../google_credentials.json")
        self.SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")

        # Initialize Sentry SDK
        if self.SENTRY_DSN:
            sentry_logging = LoggingIntegration(
                level=logging.INFO,        # Capture info and above as breadcrumbs
                event_level=logging.ERROR  # Send errors as events
            )
            sentry_sdk.init(
                dsn=self.SENTRY_DSN,
                integrations=[sentry_logging],
                traces_sample_rate=1.0,   # Capture 100% of transactions for performance monitoring
                profiles_sample_rate=1.0  # Profile 100% of sampled transactions
            )
            logging.getLogger(__name__).info("Sentry SDK initialized.")
        else:
            logging.getLogger(__name__).warning("SENTRY_DSN not found. Sentry SDK not initialized.")


class BaseHandler(ABC):
    """Abstract base class for all handlers."""

    @abstractmethod
    def handle(self, update: Update, context: CallbackContext) -> Union[int, None]:
        pass


def download_audio(message: Message, user_id: int, logger: logging.Logger) -> Optional[str]:
    """Downloads audio from a Telegram message."""
    try:
        if message.voice:
            file = message.voice.get_file()
            audio_path = f"audio_{user_id}.ogg"
        elif message.audio:
            file = message.audio.get_file()
            audio_extension = message.audio.mime_type.split('/')[-1]
            audio_path = f"audio_{user_id}.{audio_extension}"
        else:
            return None
        file.download(audio_path)
        logger.debug(f"Downloaded audio to {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        sentry_sdk.capture_exception(e)
        return None


class CalendarEvent(BaseModel):
    name: str = Field(..., description="Name of the event")
    date: str = Field(..., description="Date of the event in YYYY-MM-DD format")
    time: str = Field(..., description="Time of the event in HH:MM (24-hour) format")
    timezone: str = Field(..., description="IANA timezone string")
    description: Optional[str] = Field(None, description="Description of the event")
    connection_info: Optional[str] = Field(None,
                                           description="Connection information for the event, such as links and passwords")


class EventIdentifier(BaseModel):
    event_text: str = Field(...,
                            description="Info to identify the event to delete. All info that helps to identify the event in one string.")


class RescheduleDetails(BaseModel):
    event_text: str = Field(...,
                            description="Info to identify the event to reschedule. All info that helps to identify the event in one string.")
    new_date: Optional[str] = Field(None, description="New date in YYYY-MM-DD format")
    new_time: Optional[str] = Field(None, description="New time in HH:MM (24-hour) format")
    new_timezone: Optional[str] = Field(None, description="New IANA timezone string")


class RelevantEventResponse(BaseModel):
    found_something: bool = Field(..., description="True if the model found something possibly relevant")
    event_id: str = Field(..., description="The id of the most relevant event. Empty string if no match found.")
    event_name: str = Field(..., description="The name (summary) of the most relevant event. Empty string if no match found.")
    uncertain_match: bool = Field(..., description="True if uncertain or match is ambiguous, false if confident")


class GoogleCalendarService:
    """Service to interact with Google Calendar API."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.SCOPES = self.config.SCOPES

    def get_credentials(self, user_id: int) -> Credentials:
        creds: Optional[Credentials] = None
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

    def create_event(self, event: CalendarEvent, user_id: int) -> Dict[str, Any]:
        """Creates an event in Google Calendar."""
        with sentry_sdk.start_span(op="google_calendar", description="Creating event") as span:
            try:
                creds = self.get_credentials(user_id)
                service = build('calendar', 'v3', credentials=creds)

                # Combine date and time
                start_datetime_str = f"{event.date} {event.time}"

                timezone = event.timezone
                try:
                    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError as e:
                    self.logger.error(f"Date/time format error: {e}")
                    sentry_sdk.capture_exception(e)
                    raise e

                start_datetime = pytz.timezone(timezone).localize(start_datetime)
                end_datetime = start_datetime + timedelta(hours=1)  # Default duration 1 hour

                # Prepare event body
                event_body: Dict[str, Any] = {
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
                    event_body['description'] = event.description.replace("\\n", "\n")

                # Add connection info to description
                if event.connection_info:
                    if 'description' in event_body:
                        event_body['description'] += f"\n\nConnection Info: " + event.connection_info.replace("\\n", " ")
                    else:
                        event_body['description'] = f"Connection Info: " + event.connection_info.replace("\\n", " ")

                created_event = service.events().insert(calendarId='primary', body=event_body).execute()
                self.logger.debug(f"Created event: {created_event}")
                return created_event
            except Exception as e:
                self.logger.error(f"Failed to create event: {e}")
                sentry_sdk.capture_exception(e)
                raise e

    def delete_event(self, event_text: str, user_id: int, update: Update) -> Dict[str, Any]:
        """Deletes an event from Google Calendar based on identifier using LLM for relevance."""
        with sentry_sdk.start_span(op="google_calendar", description="Deleting event") as span:
            try:
                creds = self.get_credentials(user_id)
                service = build('calendar', 'v3', credentials=creds)

                # Fetch events to find the most relevant one
                now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
                events_result = service.events().list(
                    calendarId='primary', timeMin=now,
                    maxResults=20, singleEvents=True,
                    orderBy='startTime').execute()
                events = events_result.get('items', [])

                if len(events) == 0:
                    self.logger.debug("No events found in the calendar.")
                    return {"status": "not_found"}

                update.message.reply_text(f"Searching in {len(events)} events for the most relevant one...")

                # Utilize LLM to find the most relevant event based on identifier
                relevant_event: Optional[RelevantEventResponse] = self.find_relevant_event_with_llm(event_text, events)

                if not relevant_event or not relevant_event.found_something:
                    # TODO: Retry with more maxResults value
                    self.logger.debug("LLM did not find anything relevant.")
                    return {"status": "not_found"}

                if relevant_event.uncertain_match:
                    self.logger.debug("LLM is not sure about the event. Awaiting user confirmation.")
                    return {"status": "requires_confirmation", "event": relevant_event}

                event_id = relevant_event.event_id
                service.events().delete(calendarId='primary', eventId=event_id).execute()
                self.logger.debug(f"Deleted event: {relevant_event}")
                return {"status": "deleted", "event": relevant_event}
            except Exception as e:
                self.logger.error(f"Failed to delete event: {e}")
                sentry_sdk.capture_exception(e)
                return {"status": "error", "error": str(e)}

    def reschedule_event(self, details: Dict[str, Any], user_id: int, update: Update) -> Dict[str, Any]:
        """Reschedules an existing event based on identifier and new details using LLM for relevance."""
        with sentry_sdk.start_span(op="google_calendar", description="Rescheduling event") as span:
            try:
                event_text = details.get('event_text')
                new_date = details.get('new_date')
                new_time = details.get('new_time')
                new_timezone = details.get('new_timezone', str(get_localzone()))

                creds = self.get_credentials(user_id)
                service = build('calendar', 'v3', credentials=creds)

                # Fetch events to find the most relevant one
                now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
                events_result = service.events().list(
                    calendarId='primary', timeMin=now,
                    maxResults=20, singleEvents=True,
                    orderBy='startTime').execute()
                events = events_result.get('items', [])

                if len(events) == 0:
                    self.logger.debug("No events found in the calendar.")
                    return {"status": "not_found"}

                update.message.reply_text(f"Searching in {len(events)} events for the most relevant one...")

                # Utilize LLM to find the most relevant event based on identifier
                relevant_event: Optional[RelevantEventResponse] = self.find_relevant_event_with_llm(event_text, events)

                if not relevant_event or not relevant_event.found_something:
                    # TODO: Retry with more maxResults value
                    self.logger.debug("LLM did not find anything relevant.")
                    return {"status": "not_found"}

                if relevant_event.uncertain_match:
                    self.logger.debug("LLM is not sure about the event. Awaiting user confirmation.")
                    return {"status": "requires_confirmation", "event": relevant_event}

                event_id = relevant_event.event_id
                # Update event details
                start_datetime_str = f"{new_date} {new_time}"
                start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
                start_datetime = pytz.timezone(new_timezone).localize(start_datetime)
                end_datetime = start_datetime + timedelta(hours=1)  # Default duration
                if not new_timezone:
                    new_timezone = str(get_localzone())

                new_event: Dict[str, Any] = {
                    'start': {
                        'dateTime': start_datetime.isoformat(),
                        'timeZone': new_timezone,
                    },
                    'end': {
                        'dateTime': end_datetime.isoformat(),
                        'timeZone': new_timezone,
                    },
                }

                updated_event = service.events().update(
                    calendarId='primary', eventId=event_id, body=new_event).execute()
                self.logger.debug(f"Rescheduled event: {updated_event}")
                return {"status": "rescheduled", "event": updated_event}
            except Exception as e:
                self.logger.error(f"Failed to reschedule event: {e}")
                sentry_sdk.capture_exception(e)
                return {"status": "error", "error": str(e)}

    def find_relevant_event_with_llm(self, event_text: str, events: List[Dict[str, Any]]) -> Optional[
        RelevantEventResponse]:
        """Use LLM to determine the most relevant event based on the identifier."""
        if not events:
            self.logger.debug("No events found in the calendar.")
            return None

        events_list_json = {
            "events": [
                {
                    "id": event.get("id"),
                    "name": event.get("summary", "No Title"),
                    "kind": event.get("kind", "Unknown"),
                    "description": event.get("description", "No Description"),
                    "start": event.get("start", "Unknown"),
                    "end": event.get("end", "Unknown"),
                    "created": event.get("created", "Unknown"),
                    "updated": event.get("updated", "Unknown"),
                    # "creator": event.get("creator", "Unknown"),
                }
                for event in events
            ]
        }
        system_prompt = f"""
        You are an assistant that helps identify the most relevant event based on a given identifier. Your task is to determine which event from a provided list best matches the given event text.

        First, you will be given the event text to analyze:
        <event_text>
        {event_text}
        </event_text>

        Next, you will be provided with a list of events, each containing an id, name (summary), and description:
        <events_list>
        {json.dumps(events_list_json)}
        </events_list>

        To complete this task, follow these steps:

        1. Carefully read and analyze the event text provided.

        2. Review each event in the events list, paying close attention to the id, summary (name), and description.

        3. Compare the event text with each event in the list, looking for similarities in keywords, themes, or context.

        4. Determine which event, if any, best matches the event text. Consider the following:
           - Exact or close matches in wording
           - Thematic similarities
           - Contextual relevance

        5. If you are confident about a match, prepare to output the event's id and name.

        6. If you are not sure which event is being referred to, or if there are multiple possible matches with no clear best option, set the 'uncertain_match' flag to true.
        """

        system_prompt += """

        Provide your answer in JSON format with the following structure:
        <answer>
        {
          "found_something": boolean (true if found a possible match, false if no match found),
          "event_id": "The id of the most relevant event",
          "event_name": "The name (summary) of the most relevant event",
          "uncertain_match": boolean (true if uncertain, false if confident)
        }
        </answer>

        Remember to base your decision solely on the information provided in the event text and events list. Do not include any external information or assumptions in your analysis.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Process the following event text and find the most relevant event."},
        ]

        try:
            response = litellm.completion(
                model=SEARCH_MODEL,
                messages=messages,
                temperature=0,  # IT'S VERY IMPORTANT TO SET TEMPERATURE TO 0.
                response_format=RelevantEventResponse,
                retries=3,
                fallbacks=[FallbacksModels.SearchFallbacks]
            )
            response_content: str = response['choices'][0]['message']['content']
            response_data: Dict[str, Any] = json.loads(response_content)
            matched_event: RelevantEventResponse = RelevantEventResponse(**response_data)

            self.logger.debug(f"LLM chose event: {matched_event}")
            return matched_event
        except Exception as e:
            self.logger.error(f"LLM failed to find a relevant event: {e}")
            sentry_sdk.capture_exception(e)
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
                            "connection_info": {"type": "string",
                                                "description": "Connection information for the event, such as links and passwords"},
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
                            "event_text": {"type": "string",
                                           "description": "Info to identify the event to delete. All info that helps to identify the event in one string."},
                        },
                        "required": ["event_text"]
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
                            "event_text": {"type": "string",
                                           "description": "Info to identify the event to reschedule. All info that helps to identify the event in one string."},
                            "new_date": {"type": "string", "description": "New date in YYYY-MM-DD format"},
                            "new_time": {"type": "string", "description": "New time in HH:MM (24-hour) format"},
                            "new_timezone": {"type": "string", "description": "New IANA timezone string"},
                            "im_not_sure": {"type": "boolean",
                                            "description": "Set to true if the assistant is not sure about the exact NEW event data."}
                        },
                        "required": ["event_text", "new_date", "new_time"]
                    }
                }
            }
        ]

    def get_completion(self, model: str, messages: List[Dict[str, Any]], tool_choice: str = "auto") -> Dict[str, Any]:
        """Make a completion request to LiteLLM with function calling."""
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=self.functions,  # Pass the function schemas
                tool_choice=tool_choice,
                temperature=0,  # IT'S VERY IMPORTANT TO SET TEMPERATURE TO 0.
                retries=3,
                fallbacks=[FallbacksModels.CommandsFallbacks]
            )
            return response
        except Exception as e:
            self.logger.error(f"LiteLLM completion failed: {e}")
            sentry_sdk.capture_exception(e)
            return {}

    def parse_function_calls(self, response_message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract function calls from the model response."""
        return response_message.get('tool_calls', [])

    def execute_function(self, function_name: str, function_args: Dict[str, Any], user_id: int, update: Update) -> Dict[
        str, Any]:
        """Map function calls to GoogleCalendarService methods."""
        self.logger.debug(f"Executing function '{function_name}' with args: {function_args}")
        im_not_sure: bool = function_args.get('im_not_sure', False)

        # Define action_info based on the function
        if function_name == "add_event":
            action_info = f"Adding a new event: {function_args.get('name')} on {function_args.get('date')} at {function_args.get('time')} ({function_args.get('timezone')})"
        elif function_name == "delete_event":
            action_info = f"Deleting an event identified by: {function_args.get('event_text')}"
        elif function_name == "reschedule_event":
            action_info = f"Rescheduling an event identified by: {function_args.get('event_text')} to {function_args.get('new_date')} at {function_args.get('new_time')} ({function_args.get('new_timezone')})"
        else:
            action_info = "Performing an unknown action."

        if im_not_sure:
            return {"status": "requires_confirmation", "action_info": action_info}
        else:
            update.message.reply_text(f"{action_info}")

        # Proceed to execute the function
        try:
            if function_name == "add_event":
                event = CalendarEvent(**function_args)
                result = self.google_calendar_service.create_event(event, user_id)
                self.logger.debug(f"Event added: {result}")
                return {"status": "added", "event": result}
            elif function_name == "delete_event":
                identifier = EventIdentifier(**function_args)
                result = self.google_calendar_service.delete_event(identifier.event_text, user_id, update)
                return result
            elif function_name == "reschedule_event":
                details = RescheduleDetails(**function_args)
                result = self.google_calendar_service.reschedule_event(details.model_dump(), user_id, update)
                return result
            else:
                self.logger.error(f"Unknown function called: {function_name}")
                return {"error": "Unknown function called."}
        except Exception as e:
            self.logger.error(f"Error executing function '{function_name}': {e}")
            sentry_sdk.capture_exception(e)
            return {"status": "error", "error": str(e)}


class InputHandler(BaseHandler):
    """Handler for parsing user input."""

    def __init__(self, logger: logging.Logger, litellm_service: LiteLLMService,
                 google_calendar_service: GoogleCalendarService):
        self.logger = logger
        self.litellm_service = litellm_service
        self.google_calendar_service = google_calendar_service

    def handle(self, update: Update, context: CallbackContext) -> int:
        with sentry_sdk.start_transaction(op="handler", name="InputHandler.handle") as transaction:
            try:
                self.logger.debug("Input handler triggered")
                user = update.effective_user
                user_id: int = user.id

                # Set Sentry user context
                sentry_sdk.set_user({"id": user_id, "username": user.username, "first_name": user.first_name})

                # Inform the user that the message is being processed
                update.message.reply_text("Processing your request...")

                # Extract user message
                user_message: Optional[str] = update.message.text_markdown_v2_urled or update.message.caption_markdown_v2_urled
                self.logger.debug(f"Extracted user message: {user_message}")

                if not user_message and not update.message.voice and not update.message.audio:
                    update.message.reply_text("Unsupported message type. Please send text, audio, or voice messages.")
                    return ConversationHandler.END

                # Prepare the system prompt
                now = datetime.now().astimezone()
                system_prompt = f"""
                    You are a smart calendar assistant. Your primary task is to help users manage their events efficiently by adding new events, deleting existing events, or rescheduling events in the user's calendar.

                    When a user sends you a message, analyze it carefully to determine the appropriate action (adding, deleting, or rescheduling an event). Users may provide details such as the event name, date, time, timezone, and optional descriptions. They may also send commands in natural language, such as "Meeting with John tomorrow at 5 PM."

                    Always extract data in the user's request language.

                    If any event details are unclear, try to infer them circumstantial from the user's message.

                    To perform an action, use the appropriate function (`add_event`, `delete_event`, or `reschedule_event`) with the necessary parameters. Be sure to use the functions exactly as they are defined, without modifying or extending them.

                    Here's the current context:
                    <context>
                    Today's date and time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}
                    User's Timezone: {get_localzone()} (can be different from the event timezone)
                    Day of the week: {now.strftime('%A')}
                    </context>
                """

                # Build the message payload
                is_voice = bool(update.message.voice or update.message.audio)
                if is_voice:
                    # Handle voice or audio messages: download and transcribe
                    audio_path = download_audio(update.message, user_id, self.logger)
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
                                    "type": "image_url",  # IT'S A HACK FROM LiteLLM's DOCS. IT'S 100% WORKING.
                                    "image_url": f"data:audio/ogg;base64,{encoded_audio}",
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
                    response: Dict[str, Any] = self.litellm_service.get_completion(
                        model=model,
                        messages=messages,
                        tool_choice="auto",
                    )

                    self.logger.debug(f"LLM Response:\n{response}")
                    response_message: Dict[str, Any] = response.get('choices', [{}])[0].get('message', {})
                    tool_calls: List[Dict[str, Any]] = self.litellm_service.parse_function_calls(response_message)

                    if tool_calls:
                        # A function needs to be called
                        for tool_call in tool_calls:
                            function_name: str = tool_call['function']['name']
                            function_args: Dict[str, Any] = json.loads(tool_call['function']['arguments'])

                            self.logger.debug(f"Function call detected: {function_name} with args: {function_args}")

                            # Execute the function using LiteLLMService
                            result: Dict[str, Any] = self.litellm_service.execute_function(function_name, function_args,
                                                                                           user_id, update)

                            # Send action info to the user
                            action_info: Optional[str] = result.get("action_info")
                            if action_info:
                                update.message.reply_text(f"About to perform: {action_info}")

                            # Check if confirmation is required
                            if result.get("status") == "requires_confirmation":
                                event: Optional[RelevantEventResponse] = result.get("event")
                                if event:
                                    update.message.reply_text(f"Found event: {event.event_name}")
                                update.message.reply_text("Do you want to proceed with this action? (Yes/No)")
                                context.user_data['pending_action'] = {
                                    "function_name": function_name,
                                    "function_args": function_args
                                }
                                return BotStates.CONFIRMATION
                            elif result.get("status") == "not_found":
                                update.message.reply_text("Sorry, I couldn't find the event. Please try again.")
                            elif result.get("status") == "error":
                                update.message.reply_text(f"Error: {result.get('error')}")
                            else:
                                # Action does not require confirmation; execute and inform the user
                                if result.get("status") == "deleted":
                                    event_name: str = result.get("event", {}).get('summary', 'the event')
                                    event_time: str = result.get("event", {}).get('start', {}).get('dateTime',
                                                                                                   'the specified time')
                                    update.message.reply_text(
                                        f"I have deleted the event '{event_name}' scheduled at '{event_time}'."
                                    )
                                elif result.get("status") == "rescheduled":
                                    event: Dict[str, Any] = result.get("event", {})
                                    event_name: str = event.get('summary', 'the event')
                                    new_time: str = event.get('start', {}).get('dateTime', 'the new specified time')
                                    update.message.reply_text(
                                        f"The event '{event_name}' has been rescheduled to '{new_time}'."
                                    )
                                elif result.get("status") == "added":
                                    event: Dict[str, Any] = result.get("event", {})
                                    event_name: str = event.get('summary', 'the event')
                                    event_time: str = event.get('start', {}).get('dateTime', 'the specified time')
                                    event_link: Optional[str] = event.get("htmlLink")
                                    context.user_data['last_added_event'] = event
                                    reply_text: str = f"Event '{event_name}' has been added on '{event_time}'."
                                    if event_link:
                                        reply_text += f" You can view it here: {event_link}"
                                    update.message.reply_text(reply_text)
                        return BotStates.PARSE_INPUT
                    else:
                        update.message.reply_text("Sorry, I couldn't understand the request. Please try again.")
                        return ConversationHandler.END

                except Exception as e:
                    self.logger.error(f"Error with LiteLLM completion: {e}")
                    sentry_sdk.capture_exception(e)
                    update.message.reply_text(f"An error occurred while processing your request.\n{e}")
                    return ConversationHandler.END

            except Exception as e:
                self.logger.error(f"Unexpected error in InputHandler: {e}")
                sentry_sdk.capture_exception(e)
                update.message.reply_text(f"An unexpected error occurred.\n{e}")
                return ConversationHandler.END


class ConfirmationHandler(BaseHandler):
    """Handler for confirming event actions when LLM is unsure."""

    def __init__(self, logger: logging.Logger, litellm_service: LiteLLMService,
                 google_calendar_service: GoogleCalendarService):
        self.logger = logger
        self.litellm_service = litellm_service
        self.google_calendar_service = google_calendar_service

    def handle(self, update: Update, context: CallbackContext) -> int:
        with sentry_sdk.start_transaction(op="handler", name="ConfirmationHandler.handle") as transaction:
            try:
                self.logger.debug("Confirmation handler triggered")
                response: str = update.message.text.lower()
                user_id: int = update.effective_user.id

                # Set Sentry user context
                sentry_sdk.set_user({"id": user_id})

                pending_action: Optional[Dict[str, Any]] = context.user_data.get('pending_action')
                if not pending_action:
                    update.message.reply_text("No pending actions to confirm.")
                    return ConversationHandler.END

                if response in ['yes', 'y']:
                    function_name: str = pending_action['function_name']
                    function_args: Dict[str, Any] = pending_action['function_args']

                    # Inform the user that the action is being executed
                    update.message.reply_text(f"Proceeding with '{function_name}' action.")

                    # Execute the function now that user has confirmed
                    result: Dict[str, Any] = self.litellm_service.execute_function(function_name, function_args, user_id,
                                                                                   update)

                    if result.get("status") == "error":
                        update.message.reply_text(f"Error: {result.get('error')}")
                    else:
                        # Send appropriate success messages
                        if result.get("status") == "deleted":
                            event_name: str = result.get("event", {}).get('summary', 'the event')
                            event_time: str = result.get("event", {}).get('start', {}).get('dateTime',
                                                                                           'the specified time')
                            update.message.reply_text(
                                f"I have deleted the event '{event_name}' scheduled at '{event_time}'."
                            )
                        elif result.get("status") == "rescheduled":
                            event: Dict[str, Any] = result.get("event", {})
                            event_name: str = event.get('summary', 'the event')
                            new_time: str = event.get('start', {}).get('dateTime', 'the new specified time')
                            update.message.reply_text(
                                f"The event '{event_name}' has been rescheduled to '{new_time}'."
                            )
                        elif result.get("status") == "added":
                            event: Dict[str, Any] = result.get("event", {})
                            event_name: str = event.get('summary', 'the event')
                            event_time: str = event.get('start', {}).get('dateTime', 'the specified time')
                            event_link: Optional[str] = event.get("htmlLink")
                            context.user_data['last_added_event'] = event
                            reply_text: str = f"Event '{event_name}' has been added on '{event_time}'."
                            if event_link:
                                reply_text += f" You can view it here: {event_link}"
                            update.message.reply_text(reply_text)

                    # Clear the pending action
                    context.user_data.pop('pending_action', None)
                    return ConversationHandler.END

                elif response in ['no', 'n']:
                    update.message.reply_text("Okay, action has been cancelled.")
                    # Clear the pending action
                    context.user_data.pop('pending_action', None)
                    return ConversationHandler.END
                else:
                    update.message.reply_text("Please respond with 'Yes' or 'No'. Do you want to proceed with this action?")
                    return BotStates.CONFIRMATION

            except Exception as e:
                self.logger.error(f"Unexpected error in ConfirmationHandler: {e}")
                sentry_sdk.capture_exception(e)
                update.message.reply_text(f"An unexpected error occurred.\n{e}")
                return ConversationHandler.END


class CancelHandler(BaseHandler):
    """Handler for the /cancel command."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def handle(self, update: Update, context: CallbackContext) -> int:
        try:
            self.logger.debug("Cancel handler triggered")
            update.message.reply_text("Operation cancelled.", reply_markup=ReplyKeyboardRemove())
            return ConversationHandler.END
        except Exception as e:
            self.logger.error(f"Error in CancelHandler: {e}")
            sentry_sdk.capture_exception(e)
            update.message.reply_text(f"An error occurred while cancelling the operation.\n{e}")
            return ConversationHandler.END


class StartHandler(BaseHandler):
    """Handler for the /start command."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def handle(self, update: Update, context: CallbackContext) -> int:
        try:
            self.logger.debug("Start command received")
            update.message.reply_text(
                "Hi! I can help you manage your Google Calendar events.\n\n"
                "You can:\n"
                "- Add a new event by sending event details.\n"
                "- Delete an event by sending a command like 'Delete [event name]'.\n"
                "- Reschedule an event by sending a command like 'Reschedule [event name]'.\n"
                "- Send audio messages with event details or commands."
            )
            return BotStates.PARSE_INPUT
        except Exception as e:
            self.logger.error(f"Error in StartHandler: {e}")
            sentry_sdk.capture_exception(e)
            update.message.reply_text(f"An error occurred while starting the bot.\n{e}")
            return ConversationHandler.END


class LoggerSetup:
    """Sets up the logging configuration."""

    @staticmethod
    def setup_logging(level: str = "DEBUG") -> logging.Logger:
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=getattr(logging, level.upper(), logging.DEBUG)
        )
        logger = logging.getLogger(__name__)
        return logger