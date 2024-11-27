import os
import logging
import datetime
from typing import Optional

import pytz
import json
import base64

from pydantic import BaseModel
import litellm

from telegram import Update, ReplyKeyboardRemove
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging with DEBUG level for detailed output
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Changed to DEBUG for more detailed logs
)
logger = logging.getLogger(__name__)

# Retrieve API keys and tokens from environment variables
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")  # Gemini API Key
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Set LiteLLM verbosity if needed
os.environ['LITELLM_LOG'] = 'INFO'

# Define the scope for Google Calendar API
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

# Define conversation states
PARSE_INPUT, ASK_TIMEZONE, CONFIRMATION = range(3)

# Define the CalendarEvent Pydantic model with optional fields
class CalendarEvent(BaseModel):
    name: str
    date: str  # Expected format: YYYY-MM-DD
    time: str  # Expected format: HH:MM (24-hour)
    timezone: str  # IANA timezone string
    description: Optional[str] = None
    meeting_link: Optional[str] = None

# Define the structure for delete and reschedule functions
class EventIdentifier(BaseModel):
    identifier: str  # Could be name or description

class RescheduleDetails(BaseModel):
    identifier: str  # Event identifier
    new_date: Optional[str] = None
    new_time: Optional[str] = None
    new_timezone: Optional[str] = None

# Define function schemas for LiteLLM
function_schemas = [
    {
        "name": "add_event",
        "description": "Add a new event to the calendar.",
        "parameters": CalendarEvent.model_json_schema(),
    },
    {
        "name": "delete_event",
        "description": "Delete an event by name or description.",
        "parameters": EventIdentifier.model_json_schema(),
    },
    {
        "name": "reschedule_event",
        "description": "Reschedule an existing event by identifier.",
        "parameters": RescheduleDetails.model_json_schema(),
    },
]

# Register functions with LiteLLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_event",
            "description": "Add a new event to the calendar.",
            "parameters": CalendarEvent.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": "Delete an event by name or description.",
            "parameters": EventIdentifier.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_event",
            "description": "Reschedule an existing event by identifier.",
            "parameters": RescheduleDetails.model_json_schema(),
        },
    },
]

def add_event(event: CalendarEvent, user_id: int) -> dict:
    """Adds a new event to Google Calendar."""
    return create_calendar_event(event.model_dump(), user_id)

def delete_event(identifier: EventIdentifier, user_id: int) -> dict:
    """Deletes an event by name or description."""
    return perform_delete_event(identifier.identifier, user_id)

def reschedule_event(details: RescheduleDetails, user_id: int) -> dict:
    """Reschedules an existing event."""
    return perform_reschedule_event(details.model_dump(), user_id)

def start(update: Update, context: CallbackContext) -> int:
    """
    Handles the /start command. Greets the user and provides instructions.
    """
    logger.debug("Start command received")
    update.message.reply_text(
        "Hi! I can help you manage your Google Calendar events.\n\n"
        "You can:\n"
        "- Add a new event by sending event details.\n"
        "- Delete an event by sending a command like 'Delete [event name]'.\n"
        "- Reschedule an event by sending a command like 'Reschedule [event name]'.\n"
        "- Send audio messages with event details or commands."
    )
    return PARSE_INPUT

def parse_input(update: Update, context: CallbackContext) -> int:
    """
    Parses the user's input (text, caption, or audio) to determine the intended action.
    Utilizes LiteLLM with function calling to handle different commands.
    """
    logger.debug("parse_input handler triggered")
    message = update.message
    user_id = update.effective_user.id

    # Extract text from message.text or message.caption
    user_message = message.text if message.text else message.caption
    logger.debug(f"Extracted user message: {user_message}")

    if not user_message and not message.voice and not message.audio:
        update.message.reply_text("Unsupported message type. Please send text, audio, or voice messages.")
        return ConversationHandler.END

    if message.voice or message.audio:
        # Handle voice or audio messages: download and transcribe
        if message.voice:
            file = message.voice.get_file()
        else:
            file = message.audio.get_file()
        audio_path = f"audio_{user_id}.ogg"
        file.download(audio_path)
        logger.debug(f"Downloaded audio to {audio_path}")

        # Read and encode the audio file
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
        logger.debug("Encoded audio to base64")

        # Remove the audio file after encoding
        os.remove(audio_path)

        # Create messages payload with audio
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please process the following audio message."},
                    {
                        "type": "audio",
                        "content": encoded_audio,
                        "format": "ogg",
                    },
                ],
            }
        ]
    elif user_message:
        # Handle text or caption messages
        logger.debug(f"Received text/caption message: {user_message}")
        messages = [
            {
                "role": "user",
                "content": user_message,
            }
        ]
    else:
        update.message.reply_text("Unsupported message type. Please send text, audio, or voice messages.")
        return ConversationHandler.END

    try:
        # Make the completion request with function calling
        response = litellm.completion(
            model="gemini/gemini-1.5-pro",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # Let LiteLLM decide which tool to use
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            # A function needs to be called
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.debug(f"Function call detected: {function_name} with args: {function_args}")

                if function_name == "add_event":
                    event = CalendarEvent(**function_args)
                    result = add_event(event, user_id)
                elif function_name == "delete_event":
                    identifier = EventIdentifier(**function_args)
                    result = delete_event(identifier, user_id)
                elif function_name == "reschedule_event":
                    details = RescheduleDetails(**function_args)
                    result = reschedule_event(details, user_id)
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
            # No function calls, possibly parsing event details
            # Use LiteLLM to parse event details
            event_response = litellm.completion(
                model="gemini/gemini-1.5-pro",
                messages=messages,
                response_format={"type": "json_object", "response_schema": CalendarEvent.model_json_schema()},
            )

            event_content = event_response.choices[0].message.content  # Assuming JSON response
            logger.debug(f"Parsed CalendarEvent: {event_content}")

            try:
                event_data = CalendarEvent.model_validate_json(event_content)
                context.user_data['event'] = event_data.model_dump()
            except Exception as e:
                logger.error(f"Error parsing event: {e}")
                update.message.reply_text("Sorry, I couldn't understand the event details.")
                return ConversationHandler.END

            # Check for missing mandatory fields
            missing_fields = [field for field in ['name', 'date', 'time', 'timezone'] if not context.user_data['event'].get(field)]
            logger.debug(f"Missing fields: {missing_fields}")

            if missing_fields:
                context.user_data['missing'] = missing_fields
                return ask_missing_info(update, context)
            else:
                return confirm_event(update, context)

    except Exception as e:
        logger.error(f"Error with LiteLLM completion: {e}")
        update.message.reply_text("An error occurred while processing your request.")
        return ConversationHandler.END

def ask_missing_info(update: Update, context: CallbackContext) -> int:
    """
    Asks the user for any missing event details.
    """
    logger.debug("ask_missing_info handler triggered")
    missing = context.user_data.get('missing', [])
    if 'timezone' in missing:
        update.message.reply_text("I noticed the timezone is missing. Please provide the timezone (e.g., 'UTC', 'America/New_York').")
        return ASK_TIMEZONE
    # Add handling for other missing fields if necessary
    update.message.reply_text("Some event details are missing. Please provide the necessary information.")
    return PARSE_INPUT

def receive_timezone(update: Update, context: CallbackContext) -> int:
    """
    Receives and validates the timezone provided by the user.
    """
    logger.debug("receive_timezone handler triggered")
    timezone = update.message.text.strip()
    # Validate timezone
    if timezone not in pytz.all_timezones:
        update.message.reply_text("Invalid timezone. Please provide a valid timezone (e.g., 'UTC', 'America/New_York').")
        return ASK_TIMEZONE
    # Save timezone
    context.user_data['event']['timezone'] = timezone
    del context.user_data['missing']
    return confirm_event(update, context)

from telegram.utils.helpers import escape_markdown

def confirm_event(update: Update, context: CallbackContext) -> int:
    """
    Sends a confirmation message to the user with the extracted event details.
    """
    logger.debug("confirm_event handler triggered")
    event = context.user_data.get('event', {})
    name = event.get('name', 'N/A')
    date = event.get('date', 'N/A')
    time = event.get('time', 'N/A')
    timezone = event.get('timezone', 'N/A')
    description = event.get('description', 'No description provided.')
    meeting_link = event.get('meeting_link', 'No meeting link provided.')

    # Escape special characters for MarkdownV2
    name = escape_markdown(name, version=2)
    date = escape_markdown(date, version=2)
    time = escape_markdown(time, version=2)
    timezone = escape_markdown(timezone, version=2)
    description = escape_markdown(description, version=2)
    meeting_link = escape_markdown(meeting_link, version=2)

    # Create a human-readable summary
    confirmation_message = (
        f"*Please confirm the event details:*\n\n"
        f"*Name:* {name}\n"
        f"*Date:* {date}\n"
        f"*Time:* {time} {timezone}\n"
        f"*Description:* {description}\n"
        f"*Meeting Link:* {meeting_link}\n\n"
        "Is this correct? (Yes/No)"
    )

    update.message.reply_text(confirmation_message, parse_mode='MarkdownV2')

    return CONFIRMATION

def handle_confirmation(update: Update, context: CallbackContext) -> int:
    """
    Handles the user's confirmation response.
    If confirmed, adds the event to Google Calendar.
    """
    logger.debug("handle_confirmation handler triggered")
    response = update.message.text.lower()
    if response in ['yes', 'y']:
        # Proceed to add event to Google Calendar
        try:
            event = context.user_data.get('event', {})
            calendar_event = create_calendar_event(event, update.effective_user.id)
            event_link = calendar_event.get('htmlLink')

            update.message.reply_text(f"Event added successfully! You can view it here: {event_link}")
        except Exception as e:
            logger.error(f"Error adding event to Google Calendar: {e}")
            update.message.reply_text("Sorry, there was an error adding the event to your Google Calendar.")
        return ConversationHandler.END
    elif response in ['no', 'n']:
        update.message.reply_text("Okay, let's start over. Please send me the event details.")
        return PARSE_INPUT
    else:
        update.message.reply_text("Please respond with 'Yes' or 'No'. Is the event information correct?")
        return CONFIRMATION

def cancel(update: Update, context: CallbackContext) -> int:
    """
    Handles the /cancel command to terminate the conversation.
    """
    logger.debug("cancel handler triggered")
    update.message.reply_text("Operation cancelled.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

def create_calendar_event(event: dict, user_id: int) -> dict:
    """
    Creates an event in the user's Google Calendar.
    """
    creds = get_credentials(user_id)
    service = build('calendar', 'v3', credentials=creds)

    # Combine date and time
    start_datetime_str = f"{event['date']} {event['time']}"
    timezone = event['timezone']
    try:
        start_datetime = datetime.datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
    except ValueError as e:
        logger.error(f"Date/time format error: {e}")
        raise e

    start_datetime = pytz.timezone(timezone).localize(start_datetime)

    end_datetime = start_datetime + datetime.timedelta(hours=1)  # Default duration 1 hour

    # Prepare event body
    event_body = {
        'summary': event['name'],
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
    if event.get('description'):
        event_body['description'] = event['description']

    # Add meeting link to description or as location
    if event.get('meeting_link'):
        if 'description' in event_body:
            event_body['description'] += f"\n\nMeeting Link: {event['meeting_link']}"
        else:
            event_body['description'] = f"Meeting Link: {event['meeting_link']}"

    created_event = service.events().insert(calendarId='primary', body=event_body).execute()
    logger.debug(f"Created event: {created_event}")
    return created_event

def perform_delete_event(identifier: str, user_id: int) -> dict:
    """
    Deletes an event from Google Calendar based on identifier (name or description).
    """
    creds = get_credentials(user_id)
    service = build('calendar', 'v3', credentials=creds)

    # Fetch events to find the most relevant one
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=50, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])

    # Iterate and find a matching event
    for event in events:
        summary = event.get('summary', '').lower()
        description = event.get('description', '').lower()
        if identifier.lower() in summary or identifier.lower() in description:
            event_id = event['id']
            service.events().delete(calendarId='primary', eventId=event_id).execute()
            logger.debug(f"Deleted event: {event}")
            return {"status": "deleted", "event": event}
    logger.debug("No matching event found to delete.")
    return {"status": "not_found"}

def perform_reschedule_event(details: dict, user_id: int) -> dict:
    """
    Reschedules an existing event based on identifier and new details.
    """
    identifier = details.get('identifier')
    new_date = details.get('new_date')
    new_time = details.get('new_time')
    new_timezone = details.get('new_timezone')

    creds = get_credentials(user_id)
    service = build('calendar', 'v3', credentials=creds)

    # Fetch events to find the most relevant one
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=50, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])

    # Find the event to reschedule
    for event in events:
        summary = event.get('summary', '').lower()
        description = event.get('description', '').lower()
        if identifier.lower() in summary or identifier.lower() in description:
            event_id = event['id']
            # Update event details
            if new_date and new_time and new_timezone:
                start_datetime_str = f"{new_date} {new_time}"
                try:
                    start_datetime = datetime.datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError as e:
                    logger.error(f"Date/time format error: {e}")
                    return {"status": "error", "message": "Invalid date/time format."}

                start_datetime = pytz.timezone(new_timezone).localize(start_datetime)
                end_datetime = start_datetime + datetime.timedelta(hours=1)  # Default duration

                event['start']['dateTime'] = start_datetime.isoformat()
                event['start']['timeZone'] = new_timezone
                event['end']['dateTime'] = end_datetime.isoformat()
                event['end']['timeZone'] = new_timezone

                updated_event = service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
                logger.debug(f"Rescheduled event: {updated_event}")
                return {"status": "rescheduled", "event": updated_event}
    logger.debug("No matching event found to reschedule.")
    return {"status": "not_found"}

def get_credentials(user_id: int) -> Credentials:
    """
    Obtains user credentials, handling OAuth 2.0 flow if necessary.
    Stores tokens uniquely per user to support multiple users.
    """
    creds = None
    token_path = f'token_{user_id}.json'  # Unique token file per user
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return creds

def main():
    """Start the Telegram bot."""
    updater = Updater(TELEGRAM_TOKEN, use_context=True)

    dispatcher = updater.dispatcher

    # Define the conversation handler with the states PARSE_INPUT, ASK_TIMEZONE, CONFIRMATION
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start),
            MessageHandler(
                (Filters.text | Filters.voice | Filters.audio | Filters.caption) & ~Filters.command,
                parse_input
            )
        ],
        states={
            PARSE_INPUT: [
                MessageHandler(
                    (Filters.text | Filters.voice | Filters.audio | Filters.caption) & ~Filters.command,
                    parse_input
                ),
            ],
            ASK_TIMEZONE: [
                MessageHandler(
                    Filters.text & ~Filters.command,
                    receive_timezone
                ),
            ],
            CONFIRMATION: [
                MessageHandler(
                    Filters.text & ~Filters.command,
                    handle_confirmation
                ),
            ],
            # Additional states for delete and reschedule can be added here
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    # **For Testing Only**: Uncomment the echo handler to verify message reception
    # def echo(update: Update, context: CallbackContext) -> None:
    #     logger.debug("echo handler triggered")
    #     user_message = update.message.text
    #     logger.debug(f"Echoing message: {user_message}")
    #     update.message.reply_text(f"You said: {user_message}")

    # echo_handler = MessageHandler(Filters.text & ~Filters.command, echo)
    # dispatcher.add_handler(echo_handler)

    # Start polling for updates from Telegram
    updater.start_polling()

    logger.info("Bot started. Listening for messages...")

    # Run the bot until Ctrl-C is pressed or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()