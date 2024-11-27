# services/google_calendar_service.py
import os
import datetime
import pytz
from typing import Dict

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from src.models import CalendarEvent
from src.config import Config

class GoogleCalendarService:
    """Service to interact with Google Calendar API."""

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.SCOPES = self.config.SCOPES

    def get_credentials(self, user_id: int) -> Credentials:
        creds = None
        # create tokens directory if not exists
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
            start_datetime = datetime.datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
        except ValueError as e:
            self.logger.error(f"Date/time format error: {e}")
            raise e

        start_datetime = pytz.timezone(timezone).localize(start_datetime)
        end_datetime = start_datetime + datetime.timedelta(hours=1)  # Default duration 1 hour

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
        """Deletes an event from Google Calendar based on identifier."""
        creds = self.get_credentials(user_id)
        service = build('calendar', 'v3', credentials=creds)

        # Fetch events to find the most relevant one
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        events_result = service.events().list(
            calendarId='primary', timeMin=now,
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
                self.logger.debug(f"Deleted event: {event}")
                return {"status": "deleted", "event": event}
        self.logger.debug("No matching event found to delete.")
        return {"status": "not_found"}

    def reschedule_event(self, details: Dict, user_id: int) -> Dict:
        """Reschedules an existing event based on identifier and new details."""
        identifier = details.get('identifier')
        new_date = details.get('new_date')
        new_time = details.get('new_time')
        new_timezone = details.get('new_timezone')

        creds = self.get_credentials(user_id)
        service = build('calendar', 'v3', credentials=creds)

        # Fetch events to find the most relevant one
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        events_result = service.events().list(
            calendarId='primary', timeMin=now,
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
                        self.logger.error(f"Date/time format error: {e}")
                        return {"status": "error", "message": "Invalid date/time format."}

                    start_datetime = pytz.timezone(new_timezone).localize(start_datetime)
                    end_datetime = start_datetime + datetime.timedelta(hours=1)  # Default duration

                    event['start']['dateTime'] = start_datetime.isoformat()
                    event['start']['timeZone'] = new_timezone
                    event['end']['dateTime'] = end_datetime.isoformat()
                    event['end']['timeZone'] = new_timezone

                    updated_event = service.events().update(
                        calendarId='primary', eventId=event_id, body=event).execute()
                    self.logger.debug(f"Rescheduled event: {updated_event}")
                    return {"status": "rescheduled", "event": updated_event}
        self.logger.debug("No matching event found to reschedule.")
        return {"status": "not_found"}