# google_genai_service.py
import base64

import google.generativeai as genai
from typing import List, Dict, Any

from google.ai.generativelanguage_v1beta import FunctionCall

from src.models import CalendarEvent, EventIdentifier, RescheduleDetails

class GoogleGenAIService:
    """Service to interact with Google GenAI for function calling."""

    def __init__(self, api_key: str, logger):
        """
        Initialize the service.
        :param api_key: API key for Google GenAI.
        :param logger: Logger instance for debugging.
        """
        self.logger = logger
        genai.configure(api_key=api_key)

        # Initialize the GenerativeModel with the function declarations
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=[add_event, delete_event, reschedule_event]
        )

    def get_completion(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Makes a completion request using the provided messages.

        :param messages: List of message dictionaries with 'role' and 'content'.
        :return: GenAI response object.
        """
        try:
            # Construct history from messages
            history = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")

                if role == "system":
                    history.append({"role": "system", "content": content})
                elif role == "user":
                    if isinstance(content, dict) and content.get("type") == "audio":
                        # Handle audio content
                        audio_data = base64.b64decode(content.get("content"))
                        mime_type = content.get("format", "audio/ogg")  # Default to 'audio/ogg' if not specified
                        history.append({
                            "role": "user",
                            "parts": [
                                {"mime_type": mime_type, "data": audio_data}
                            ]
                        })
                    else:
                        history.append({"role": "user", "content": content})
                elif role == "assistant":
                    history.append({"role": "assistant", "content": content})
                else:
                    self.logger.warning(f"Unknown role '{role}' in message. Skipping.")

            self.logger.debug(f"Constructed history for GenAI: {history}")

            # Make the completion request
            response = self.model.generate_content(history)
            self.logger.debug(f"Received response from GenAI: {response}")

            return response
        except Exception as e:
            self.logger.error(f"Error with GenAI completion: {e}")
            raise e

    def parse_function_calls(self, response: Any) -> List[FunctionCall]:
        """
        Extracts function calls from the GenAI response.

        :param response: GenAI response object.
        :return: List of FunctionCall objects.
        """
        function_calls = []
        try:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
        except AttributeError as e:
            self.logger.error(f"Unexpected response structure: {e}")

        return function_calls

def add_event(name: str, date: str, time: str, timezone: str, description: str = None, meeting_link: str = None) -> str:
    """Add a new event to the calendar."""
    # This function is a placeholder for schema definition.
    # Actual implementation is handled in InputHandler.
    return "add_event called"

def delete_event(identifier: str) -> str:
    """Delete an event by name or description."""
    # Placeholder for schema definition.
    return "delete_event called"

def reschedule_event(identifier: str, new_date: str = None, new_time: str = None, new_timezone: str = None) -> str:
    """Reschedule an existing event by identifier."""
    # Placeholder for schema definition.
    return "reschedule_event called"