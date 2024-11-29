# logic.py

import base64
import datetime
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

import litellm
import pytz
import telegram
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from tzlocal import get_localzone

from telegram import Update, Message, ReplyKeyboardRemove
from telegram.ext import CallbackContext, ConversationHandler

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from google.cloud import firestore

# Constants
SEARCH_MODEL = "gemini/gemini-1.5-flash-002"
COMMANDS_MODEL_VOICE = "gemini/gemini-1.5-flash-002"
COMMANDS_MODEL_TEXT = "gemini/gemini-1.5-flash-002"
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


class IRepository(ABC):
    """Repository interface for managing tokens and user state."""

    @abstractmethod
    def save_tokens(self, user_id: int, tokens: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_tokens(self, user_id: int) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_tokens(self, user_id: int) -> None:
        pass

    @abstractmethod
    def save_user_state(self, user_id: int, state: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_user_state(self, user_id: int) -> None:
        pass


class FirestoreRepository(IRepository):
    """Repository implementation using Google Firestore."""

    def __init__(self, config: 'Config'):
        self.client = firestore.Client()
        self.collection = config.FIRESTORE_COLLECTION

    def get_user_document(self, user_id: int) -> firestore.DocumentReference:
        """Get a reference to the user's document."""
        return self.client.collection(self.collection).document(str(user_id))

    def save_tokens(self, user_id: int, tokens: Dict[str, Any]) -> None:
        """Save OAuth tokens to Firestore."""
        user_doc = self.get_user_document(user_id)
        user_doc.set({"tokens": tokens}, merge=True)

    def get_tokens(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve OAuth tokens from Firestore."""
        user_doc = self.get_user_document(user_id)
        doc = user_doc.get()
        if doc.exists:
            return doc.to_dict().get("tokens")
        return None

    def delete_tokens(self, user_id: int) -> None:
        """Delete OAuth tokens from Firestore."""
        user_doc = self.get_user_document(user_id)
        user_doc.update({"tokens": firestore.DELETE_FIELD})

    def save_user_state(self, user_id: int, state: Dict[str, Any]) -> None:
        """Save user state to Firestore."""
        user_doc = self.get_user_document(user_id)
        user_doc.set({"state": state}, merge=True)

    def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve user state from Firestore."""
        user_doc = self.get_user_document(user_id)
        doc = user_doc.get()
        if doc.exists:
            return doc.to_dict().get("state")
        return None

    def delete_user_state(self, user_id: int) -> None:
        """Delete user state from Firestore."""
        user_doc = self.get_user_document(user_id)
        user_doc.update({"state": firestore.DELETE_FIELD})


class FileSystemRepository(IRepository):
    """Repository implementation using the local filesystem."""

    def __init__(self, config: 'Config'):
        self.token_dir = os.path.abspath("../google_tokens")
        os.makedirs(self.token_dir, exist_ok=True)

    def _get_token_path(self, user_id: int) -> str:
        return os.path.join(self.token_dir, f"token_{user_id}.json")

    def _get_state_path(self, user_id: int) -> str:
        return os.path.join(self.token_dir, f"state_{user_id}.json")

    def save_tokens(self, user_id: int, tokens: Dict[str, Any]) -> None:
        path = self._get_token_path(user_id)
        with open(path, 'w') as f:
            json.dump(tokens, f)
        logging.info(f"Saved tokens for user {user_id} to {path}")

    def get_tokens(self, user_id: int) -> Optional[Dict[str, Any]]:
        path = self._get_token_path(user_id)
        if os.path.exists(path):
            with open(path, 'r') as f:
                tokens = json.load(f)
            logging.info(f"Retrieved tokens for user {user_id} from {path}")
            return tokens
        return None

    def delete_tokens(self, user_id: int) -> None:
        path = self._get_token_path(user_id)
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"Deleted tokens for user {user_id} from {path}")

    def save_user_state(self, user_id: int, state: Dict[str, Any]) -> None:
        path = self._get_state_path(user_id)
        with open(path, 'w') as f:
            json.dump(state, f)
        logging.info(f"Saved user state for user {user_id} to {path}")

    def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        path = self._get_state_path(user_id)
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
            logging.info(f"Retrieved user state for user {user_id} from {path}")
            return state
        return None

    def delete_user_state(self, user_id: int) -> None:
        path = self._get_state_path(user_id)
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"Deleted user state for user {user_id} from {path}")


class Config:
    """Configuration management using environment variables."""

    def __init__(self, env_file: str = ".env"):
        # Load environment variables from .env in local development
        if os.getenv("ENV", "local") == "local":
            from dotenv import load_dotenv
            load_dotenv(env_file)

        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
        self.GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
        self.GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
        self.SCOPES = ['https://www.googleapis.com/auth/calendar.events']
        self.SENTRY_DSN = os.getenv("SENTRY_DSN")
        self.ENV = os.getenv("ENV", "local")

        # Firestore configuration
        self.FIRESTORE_PROJECT = os.getenv("FIRESTORE_PROJECT")
        self.FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "google_access_tokens")

        # Initialize Sentry SDK for production
        sentry_logging = LoggingIntegration(
            level=logging.INFO,        # Capture info and above as breadcrumbs
            event_level=logging.ERROR  # Send errors as events
        )
        sentry_sdk.init(
            dsn=self.SENTRY_DSN,
            integrations=[sentry_logging],
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0  # Profile 100% of sampled transactions
        )
        logging.getLogger(__name__).info("Sentry SDK initialized.")

    def get_repository(self) -> IRepository:
        """Returns the appropriate repository based on the environment."""
        if self.ENV == "local":
            logging.info("Using FileSystemRepository for local environment.")
            return FileSystemRepository(self)
        else:
            logging.info("Using FirestoreRepository for production environment.")
            return FirestoreRepository(self)


class BaseHandler(ABC):
    """Abstract base class for all handlers."""

    @abstractmethod
    def handle(self, update: Update, context: CallbackContext) -> Union[int, None]:
        pass


def download_audio_in_memory(message: Message, user_id: int, logger: logging.Logger) -> bytes:
    """Downloads audio from a Telegram message into RAM."""
    if message.voice:
        file = message.voice.get_file()
        mime_type = message.voice.mime_type
    elif message.audio:
        file = message.audio.get_file()
        mime_type = message.audio.mime_type
    else:
        raise ValueError("Message does not contain audio or voice data.")

    # Download the file as bytes
    audio_bytes = file.download_as_bytearray()
    logger.info(f"Downloaded audio for user {user_id}, MIME type: {mime_type}")

    return bytes(audio_bytes)


# Pydantic Models
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

    def __init__(self, config: Config, logger: logging.Logger, repository: IRepository):
        self.config = config
        self.logger = logger
        self.SCOPES = self.config.SCOPES
        self.repository = repository

    def generate_auth_url(self, user_id: int) -> str:
        """Generates the OAuth 2.0 authorization URL for the user."""
        client_config = {
            "web": {
                "client_id": self.config.GOOGLE_CLIENT_ID,
                "client_secret": self.config.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.config.GOOGLE_REDIRECT_URI],
            }
        }

        flow = Flow.from_client_config(
            client_config=client_config,
            scopes=self.SCOPES,
            redirect_uri=self.config.GOOGLE_REDIRECT_URI
        )

        # Embed user_id in state for mapping after callback
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=str(user_id),  # TODO: Encrypt with JWT for security
            prompt='consent'  # Forces consent screen to ensure refresh_token is received
        )

        return authorization_url

    def get_credentials(self, user_id: int, authorization_code: Optional[str] = None) -> Credentials | None:
        """Gets or refreshes credentials for a user."""
        creds = None

        if authorization_code:
            # Initialize Flow with client_config and scopes
            client_config = {
                "web": {
                    "client_id": self.config.GOOGLE_CLIENT_ID,
                    "client_secret": self.config.GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.config.GOOGLE_REDIRECT_URI],
                }
            }

            flow = Flow.from_client_config(
                client_config=client_config,
                scopes=self.SCOPES,
                redirect_uri=self.config.GOOGLE_REDIRECT_URI
            )

            # Fetch token using authorization_code
            flow.fetch_token(code=authorization_code)

            creds = flow.credentials

            # Save credentials using the repository
            self.repository.save_tokens(user_id, json.loads(creds.to_json()))

        else:
            tokens = self.repository.get_tokens(user_id)
            if tokens:
                creds = Credentials.from_authorized_user_info(info=tokens, scopes=self.SCOPES)
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    # Save the refreshed credentials

                    # Convert JSON string to dictionary before saving
                    self.repository.save_tokens(user_id, json.loads(creds.to_json()))
            else:
                return None

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
                self.logger.info(f"Created event: {created_event}")
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
                    self.logger.info("No events found in the calendar.")
                    return {"status": "not_found"}

                update.message.reply_text(f"Searching in {len(events)} events for the most relevant one...")

                # Utilize LLM to find the most relevant event based on identifier
                relevant_event: Optional[RelevantEventResponse] = self.find_relevant_event_with_llm(event_text, events)

                if not relevant_event or not relevant_event.found_something:
                    # TODO: Retry with more maxResults value
                    self.logger.info("LLM did not find anything relevant.")
                    return {"status": "not_found"}

                if relevant_event.uncertain_match:
                    self.logger.info("LLM is not sure about the event. Awaiting user confirmation.")
                    return {"status": "requires_confirmation", "event": relevant_event}

                event_id = relevant_event.event_id
                service.events().delete(calendarId='primary', eventId=event_id).execute()
                self.logger.info(f"Deleted event: {relevant_event}")
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
                    self.logger.info("No events found in the calendar.")
                    return {"status": "not_found"}

                update.message.reply_text(f"Searching in {len(events)} events for the most relevant one...")

                # Utilize LLM to find the most relevant event based on identifier
                relevant_event: Optional[RelevantEventResponse] = self.find_relevant_event_with_llm(event_text, events)

                if not relevant_event or not relevant_event.found_something:
                    # TODO: Retry with more maxResults value
                    self.logger.info("LLM did not find anything relevant.")
                    return {"status": "not_found"}

                if relevant_event.uncertain_match:
                    self.logger.info("LLM is not sure about the event. Awaiting user confirmation.")
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
                self.logger.info(f"Rescheduled event: {updated_event}")
                return {"status": "rescheduled", "event": updated_event}
            except Exception as e:
                self.logger.error(f"Failed to reschedule event: {e}")
                sentry_sdk.capture_exception(e)
                return {"status": "error", "error": str(e)}

    def find_relevant_event_with_llm(self, event_text: str, events: List[Dict[str, Any]]) -> Optional[
        RelevantEventResponse]:
        """Use LLM to determine the most relevant event based on the identifier."""
        if not events:
            self.logger.info("No events found in the calendar.")
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

            self.logger.info(f"LLM chose event: {matched_event}")
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
        self.logger.info(f"Executing function '{function_name}' with args: {function_args}")
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
                self.logger.info(f"Event added: {result}")
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
                 google_calendar_service: GoogleCalendarService, repository: IRepository):
        self.logger = logger
        self.litellm_service = litellm_service
        self.google_calendar_service = google_calendar_service
        self.repository = repository

    def handle(self, update: Update, context: CallbackContext) -> int:
        with sentry_sdk.start_transaction(op="handler", name="InputHandler.handle") as transaction:

            self.logger.info("Input handler triggered")
            user = update.effective_user
            user_id: int = user.id

            # Set Sentry user context
            sentry_sdk.set_user({"id": user_id, "username": user.username, "first_name": user.first_name})

            # Check if user has valid credentials
            try:
                creds = self.google_calendar_service.get_credentials(user_id)
                if not creds or not creds.valid:
                    # Generate auth URL and prompt user
                    auth_url = self.google_calendar_service.generate_auth_url(user_id)
                    update.message.reply_text(
                        "You need to authorize access to your Google Calendar. Please click the link below to authorize:",
                        reply_markup=telegram.InlineKeyboardMarkup([
                            [telegram.InlineKeyboardButton("Authorize", url=auth_url)]
                        ])
                    )
                    return ConversationHandler.END

                # Inform the user that the message is being processed
                update.message.reply_text("Processing your request...")

                # IT'S 100% RIGHT TO USE update.message.text_markdown_v2_urled INSTEAD OF update.message.text_markdown_v2.
                user_message: Optional[str] = update.message.text_markdown_v2_urled or update.message.caption_markdown_v2_urled
                self.logger.info(f"Extracted user message: {user_message}")

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
                    audio_bytes = download_audio_in_memory(update.message, user_id, self.logger)
                    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
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
                    self.logger.info(f"Received text/caption message: {user_message}")
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

                    self.logger.info(f"LLM Response:\n{response}")
                    response_message: Dict[str, Any] = response.get('choices', [{}])[0].get('message', {})
                    tool_calls: List[Dict[str, Any]] = self.litellm_service.parse_function_calls(response_message)

                    if tool_calls:
                        # A function needs to be called
                        for tool_call in tool_calls:
                            function_name: str = tool_call['function']['name']
                            function_args: Dict[str, Any] = json.loads(tool_call['function']['arguments'])

                            self.logger.info(f"Function call detected: {function_name} with args: {function_args}")

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

                                # Save pending action to the repository
                                pending_action = {
                                    "function_name": function_name,
                                    "function_args": function_args
                                }
                                self.repository.save_user_state(user_id, {"pending_action": pending_action})

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
                 google_calendar_service: GoogleCalendarService, repository: IRepository):
        self.logger = logger
        self.litellm_service = litellm_service
        self.google_calendar_service = google_calendar_service
        self.repository = repository

    def handle(self, update: Update, context: CallbackContext) -> int:
        with sentry_sdk.start_transaction(op="handler", name="ConfirmationHandler.handle") as transaction:
            try:
                self.logger.info("Confirmation handler triggered")
                response: str = update.message.text.lower()
                user_id: int = update.effective_user.id

                # Set Sentry user context
                sentry_sdk.set_user({"id": user_id})

                # Retrieve pending action from the repository
                user_state = self.repository.get_user_state(user_id)
                if not user_state or "pending_action" not in user_state:
                    update.message.reply_text("No pending actions to confirm.")
                    return ConversationHandler.END

                pending_action = user_state["pending_action"]

                if response in ['yes', 'y', 'да', 'д']:
                    function_name: str = pending_action['function_name']
                    function_args: Dict[str, Any] = pending_action['function_args']

                    # Inform the user that the action is being executed
                    update.message.reply_text(f"Proceeding with '{function_name}' action.")

                    # Execute the function using LiteLLMService
                    result: Dict[str, Any] = self.litellm_service.execute_function(function_name, function_args, user_id,
                                                                                   update)

                    # Handle the result
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

                    # Clear the pending action from the repository
                    self.repository.delete_user_state(user_id)
                    return ConversationHandler.END

                elif response in ['no', 'n', 'нет', 'н']:
                    update.message.reply_text("Okay, action has been cancelled.")
                    # Clear the pending action from the repository
                    self.repository.delete_user_state(user_id)
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
            self.logger.info("Cancel handler triggered")
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
            self.logger.info("Start command received")
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
    def setup_logging(level: str = "INFO") -> logging.Logger:
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=getattr(logging, level.upper(), logging.INFO)
        )
        logger = logging.getLogger(__name__)
        return logger