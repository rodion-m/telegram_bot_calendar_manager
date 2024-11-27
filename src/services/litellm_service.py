from typing import List, Dict
import litellm
from src.models import CalendarEvent, EventIdentifier, RescheduleDetails

class LiteLLMService:
    """Service to interact with LLMs for function calling."""

    def __init__(self, logger, provider: str = "openai"):
        """
        Initialize the service.
        :param logger: Logger instance for debugging.
        :param provider: LLM provider, e.g., "openai" or "gemini".
        """
        self.logger = logger
        self.provider = provider
        self._tools = self.define_tools()

    def define_tools(self) -> List[Dict]:
        """Defines the function schemas for LLM."""
        function_declarations = [
            self.create_function_schema(
                name="add_event",
                description="Add a new event to the calendar.",
                model=CalendarEvent,
                optional_fields=["description", "meeting_link"],
            ),
            self.create_function_schema(
                name="delete_event",
                description="Delete an event by name or description.",
                model=EventIdentifier,
            ),
            self.create_function_schema(
                name="reschedule_event",
                description="Reschedule an existing event by identifier.",
                model=RescheduleDetails,
                optional_fields=["new_date", "new_time", "new_timezone"],
            ),
        ]
        config = {
            "function_declarations": function_declarations,
        }

        # For Gemini API, add function calling configuration
        if self.provider == "gemini":
            config["function_calling_config"] = {"mode": "AUTO"}  # Options: AUTO, ANY, NONE

        return [config]

    # GEMINI IS NOT WORKING!!!
    def create_function_schema(self, name: str, description: str, model, optional_fields: List[str] = None) -> Dict:
        """Creates function schema compatible with OpenAI and Gemini."""
        schema = model.model_json_schema()
        if self.provider == "gemini":
            # Add `optionalProperties` for Gemini
            schema["optionalProperties"] = optional_fields or []
            schema = self._remove_default_fields(schema) # Remove 'default' fields for Gemini

        return {
            "name": name,
            "description": description,
            "parameters": schema,
        }

    def _remove_default_fields(self, schema):
        """
        Recursively remove 'default' fields from the schema dictionary.
        """
        if isinstance(schema, dict):
            new_dict = {}
            for key, value in schema.items():
                if key == 'default':
                    self.logger.debug(f"Removing 'default' key with value: {value}")
                    continue  # Skip the 'default' key
                # Recursively process the value
                new_value = self._remove_default_fields(value)
                new_dict[key] = new_value
            return new_dict
        elif isinstance(schema, list):
            return [self._remove_default_fields(item) for item in schema]
        else:
            return schema

    def get_completion(self, model: str, messages: List[Dict], tool_choice: str = "auto") -> Dict:
        """Makes a completion request with function calling."""
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=self._tools,
            tool_choice=tool_choice,
        )
        return response

    def parse_function_calls(self, response_message: Dict) -> List[Dict]:
        """Extracts function calls from the LLM response."""
        return response_message.get("tool_calls", [])