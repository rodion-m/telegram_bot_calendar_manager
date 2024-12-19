# logic.py

import asyncio
import base64
import datetime
import json
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Generator

import aioboto3
import jwt
import litellm
import pytz
import sentry_sdk
import telegram
from cryptography.hazmat.backends import default_backend
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from loguru import logger
from pydantic import BaseModel, Field
from requests import HTTPError
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.quart import QuartIntegration
from telegram import Update, Message, ReplyKeyboardRemove
from telegram.ext import ContextTypes, ConversationHandler
from tinydb import Query, TinyDB
from tzlocal import get_localzone

# Constants
SEARCH_MODEL = "gemini/gemini-2.0-flash-exp"
COMMANDS_MODEL_VOICE = "gemini/gemini-2.0-flash-exp"
COMMANDS_MODEL_TEXT = "gemini/gemini-2.0-flash-exp"

# TODO: Reimplement reschedule_event feature, cause it's complex

class FallbacksModels:
    """Fallback models for LLM completion requests."""
    SearchFallbacks = "openai/gpt-4o-mini"
    CommandsFallbacks = "gemini/gemini-1.5-flash-002"

class BotStates:
    """States for the conversation handler."""
    AUTHENTICATE = 1
    PARSE_INPUT = 2
    CONFIRMATION = 3


from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_data(plaintext: str, encryption_key: bytes) -> str:
    """
    Encrypts plaintext using AES-GCM.

    Args:
        plaintext (str): The data to encrypt.
        encryption_key (bytes): The AES key (32 bytes for AES-256).

    Returns:
        str: The encrypted data encoded in base64.
    """
    aesgcm = AESGCM(encryption_key)
    nonce = os.urandom(12)  # 96-bit nonce for AES-GCM
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
    encrypted = base64.b64encode(nonce + ciphertext).decode('utf-8')
    return encrypted


def decrypt_data(encrypted_data: str, encryption_key: bytes) -> str:
    """
    Decrypts data encrypted with AES-GCM.

    Args:
        encrypted_data (str): The encrypted data in base64.
        encryption_key (bytes): The AES key (32 bytes for AES-256).

    Returns:
        str: The decrypted plaintext.
    """
    aesgcm = AESGCM(encryption_key)
    data = base64.b64decode(encrypted_data.encode('utf-8'))
    nonce = data[:12]
    ciphertext = data[12:]
    plaintext = aesgcm.decrypt(nonce, ciphertext, None).decode('utf-8')
    return plaintext

@contextmanager
def start_span_smart(op: str, description: str) -> Generator[Any, None, None]:
    """
    Smart context manager that profiles code execution either locally or with Sentry
    based on the environment.

    In debug/local environment, it logs execution time.
    In production, it uses Sentry's performance monitoring.

    Args:
        op: Operation name/category (e.g. "db", "http", "cache")
        description: Detailed description of the operation

    Yields:
        In debug mode: None
        In production: Sentry span object

    Usage:
        with start_span_smart("db", "Fetch user profile") as span:
            # Your code here
            pass
    """
    try:
        # Check if we're in debug/local environment
        is_debug = __debug__ or os.getenv('ENVIRONMENT', '').lower() in ('local', 'development')

        if is_debug:
            start_time = time.perf_counter()
            try:
                yield None
            finally:
                elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
                logger.debug(
                    f"Operation '{op}' - {description} took {elapsed_time:.2f}ms"
                )
        else:
            # Use actual Sentry span in production
            with sentry_sdk.start_span(op=op, description=description) as span:
                yield span

    except Exception as e:
        logger.exception(f"Error in span '{op}' - {description}: {str(e)}")
        raise

class IRepository(ABC):
    """Repository interface for managing tokens and user state."""

    @abstractmethod
    async def save_tokens(self, user_id: int, tokens: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def get_tokens(self, user_id: int) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_tokens(self, user_id: int) -> None:
        pass

    @abstractmethod
    async def save_user_state(self, user_id: int, state: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_user_state(self, user_id: int) -> None:
        pass

    @abstractmethod
    async def get_daily_request_count(self, user_id: int) -> (int, str):
        """Retrieve the daily request count and last request date for a user."""
        pass

    @abstractmethod
    async def increment_daily_request_count(self, user_id: int) -> None:
        """Increment the daily request count for a user."""
        pass

    @abstractmethod
    async def reset_daily_request_count(self, user_id: int, new_date: str) -> None:
        """Reset the daily request count for a user with the new date."""
        pass

class DynamoDbRepository(IRepository):
    """Repository implementation using Amazon DynamoDB."""

    def __init__(self, encryption_key: bytes, table_name: str = 'google-telegram-planner', region_name: str = 'eu-north-1'):
        self.table_name = table_name
        self.region_name = region_name
        self.encryption_key = encryption_key

    async def get_user_item(self, user_id: int) -> Dict[str, Any]:
        """Construct the primary key for the user's item."""
        return {'user_id': {'N': str(user_id)}}

    async def save_tokens(self, user_id: int, tokens: Dict[str, Any]) -> None:
        """Save OAuth tokens to DynamoDB, encrypted."""
        async with aioboto3.Session().client('dynamodb', region_name=self.region_name) as client:
            item = await self.get_user_item(user_id)

            # Convert tokens dict to JSON string
            tokens_json = json.dumps(tokens)

            # Encrypt the tokens
            encrypted_tokens = encrypt_data(tokens_json, self.encryption_key)

            # Store as a single encrypted string
            item['tokens'] = {'S': encrypted_tokens}

            await client.put_item(TableName=self.table_name, Item=item)
            logger.info(f"Saved encrypted tokens for user {user_id} to DynamoDB.")

    async def get_tokens(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve OAuth tokens from DynamoDB, decrypted."""
        async with aioboto3.Session().client('dynamodb', region_name=self.region_name) as client:
            key = await self.get_user_item(user_id)
            response = await client.get_item(TableName=self.table_name, Key=key)
            item = response.get('Item')
            if item and 'tokens' in item:
                encrypted_tokens = item['tokens']['S']
                try:
                    tokens_json = decrypt_data(encrypted_tokens, self.encryption_key)
                    tokens = json.loads(tokens_json)
                    logger.info(f"Retrieved and decrypted tokens for user {user_id} from DynamoDB.")
                    return tokens
                except Exception as e:
                    logger.error(f"Failed to decrypt tokens for user {user_id}: {e}")
                    sentry_sdk.capture_exception(e)
                    return None
            return None

    async def delete_tokens(self, user_id: int) -> None:
        """Delete OAuth tokens from DynamoDB."""
        async with aioboto3.Session().client('dynamodb', region_name=self.region_name) as client:
            key = await self.get_user_item(user_id)
            update_expression = "REMOVE tokens"
            await client.update_item(TableName=self.table_name, Key=key, UpdateExpression=update_expression)
            logger.info(f"Deleted tokens for user {user_id} from DynamoDB.")

    async def save_user_state(self, user_id: int, state: Dict[str, Any]) -> None:
        """Save user state to DynamoDB."""
        async with aioboto3.Session().client('dynamodb', region_name=self.region_name) as client:
            key = await self.get_user_item(user_id)
            # Retrieve existing state
            response = await client.get_item(TableName=self.table_name, Key=key)
            existing_item = response.get('Item', {})
            existing_state = existing_item.get('state', {}).get('M', {})
            # Merge existing state with new_state
            merged_state = {k: v['S'] for k, v in existing_state.items()}
            merged_state.update(state)
            # Prepare the merged state for DynamoDB
            merged_state_dynamodb = {'M': {k: {'S': str(v)} for k, v in merged_state.items()}}
            # Update the item
            await client.update_item(
                TableName=self.table_name,
                Key=key,
                UpdateExpression="SET state = :s",
                ExpressionAttributeValues={":s": merged_state_dynamodb['M']}
            )
            logger.info(f"Updated user state for user {user_id} in DynamoDB.")

    async def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve user state from DynamoDB."""
        async with aioboto3.Session().client('dynamodb', region_name=self.region_name) as client:
            key = await self.get_user_item(user_id)
            response = await client.get_item(TableName=self.table_name, Key=key)
            item = response.get('Item')
            if item and 'state' in item:
                state = {k: v['S'] for k, v in item['state']['M'].items()}
                logger.info(f"Retrieved user state for user {user_id} from DynamoDB.")
                return state
            return None

    async def delete_user_state(self, user_id: int) -> None:
        """Delete user state from DynamoDB."""
        async with aioboto3.Session().client('dynamodb', region_name=self.region_name) as client:
            key = await self.get_user_item(user_id)
            update_expression = "REMOVE state"
            await client.update_item(TableName=self.table_name, Key=key, UpdateExpression=update_expression)
            logger.info(f"Deleted user state for user {user_id} from DynamoDB.")

    async def get_daily_request_count(self, user_id: int) -> (int, str):
        """Retrieve the daily request count and last request date for a user."""
        async with aioboto3.Session().client('dynamodb', region_name=self.region_name) as client:
            key = await self.get_user_item(user_id)
            response = await client.get_item(TableName=self.table_name, Key=key)
            item = response.get('Item')
            if item and 'daily_requests' in item:
                daily_requests = item['daily_requests'].get('N', '0')
                last_request_date = item['last_request_date'].get('S', '')
                return int(daily_requests), last_request_date
            return 0, ''

    async def increment_daily_request_count(self, user_id: int) -> None:
        """Increment the daily request count for a user."""
        async with aioboto3.Session().client('dynamodb', region_name=self.region_name) as client:
            key = await self.get_user_item(user_id)
            update_expression = "ADD daily_requests :inc"
            expression_attribute_values = {":inc": {"N": "1"}}
            await client.update_item(
                TableName=self.table_name,
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values
            )
            logger.info(f"Incremented daily request count for user {user_id}.")

    async def reset_daily_request_count(self, user_id: int, new_date: str) -> None:
        """Reset the daily request count for a user with the new date."""
        async with aioboto3.Session().client('dynamodb', region_name=self.region_name) as client:
            key = await self.get_user_item(user_id)
            update_expression = "SET daily_requests = :zero, last_request_date = :date"
            expression_attribute_values = {
                ":zero": {"N": "0"},
                ":date": {"S": new_date}
            }
            await client.update_item(
                TableName=self.table_name,
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values
            )
            logger.info(f"Reset daily request count for user {user_id} on {new_date}.")


# noinspection PyTypeChecker
class TinyDbRepository(IRepository):
    """Repository implementation using TinyDB."""

    def __init__(self, db_path: str = "tinydb.json"):
        """
        Initializes the TinyDbRepository.

        Args:
            db_path (str): Path to the TinyDB JSON file. Defaults to "tinydb.json".
        """
        # Define the path to the TinyDB JSON file
        self.db_path = os.path.abspath(db_path)
        self.db = TinyDB(self.db_path)

        # Create separate tables for tokens and user states
        self.tokens_table = self.db.table('tokens')
        self.state_table = self.db.table('state')

    async def save_tokens(self, user_id: int, tokens: Dict[str, Any]) -> None:
        """
        Saves the authentication tokens for a user.

        Args:
            user_id (int): The unique identifier of the user.
            tokens (Dict[str, Any]): A dictionary of token data.
        """
        user = Query()
        # Upsert ensures that the record is updated if it exists, or inserted if it doesn't
        self.tokens_table.upsert(
            {'user_id': user_id, 'tokens': tokens},
            user.user_id == user_id
        )

    async def get_tokens(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the authentication tokens for a user.

        Args:
            user_id (int): The unique identifier of the user.

        Returns:
            Optional[Dict[str, Any]]: The tokens dictionary if found, else None.
        """
        User = Query()
        result = self.tokens_table.get(User.user_id == user_id)
        if result:
            return result.get('tokens')
        return None

    async def delete_tokens(self, user_id: int) -> None:
        """
        Deletes the authentication tokens for a user.

        Args:
            user_id (int): The unique identifier of the user.
        """
        User = Query()
        self.tokens_table.remove(User.user_id == user_id)

    async def save_user_state(self, user_id: int, state: Dict[str, Any]) -> None:
        """
        Saves the state information for a user.

        Args:
            user_id (int): The unique identifier of the user.
            state (Dict[str, Any]): A dictionary representing the user's state.
        """
        User = Query()
        result = self.state_table.get(User.user_id == user_id)
        if result:
            existing_state = result.get('state', {})
            existing_state.update(state)
            self.state_table.update({'state': existing_state}, User.user_id == user_id)
            logger.info(f"Updated user state for user {user_id} in TinyDB.")
        else:
            # If no existing state, create a new one
            self.state_table.insert({'user_id': user_id, 'state': state})
            logger.info(f"Created new user state for user {user_id} in TinyDB.")

    async def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the state information for a user.

        Args:
            user_id (int): The unique identifier of the user.

        Returns:
            Optional[Dict[str, Any]]: The state dictionary if found, else None.
        """
        User = Query()
        result = self.state_table.get(User.user_id == user_id)
        if result:
            return result.get('state')
        return None

    async def delete_user_state(self, user_id: int) -> None:
        """
        Deletes the state information for a user.

        Args:
            user_id (int): The unique identifier of the user.
        """
        User = Query()
        self.state_table.remove(User.user_id == user_id)

    async def get_daily_request_count(self, user_id: int) -> (int, str):
        """Retrieve the daily request count and last request date for a user."""
        User = Query()
        result = self.state_table.get(User.user_id == user_id)
        if result:
            daily_requests = result.get('daily_requests', 0)
            last_request_date = result.get('last_request_date', '')
            return daily_requests, last_request_date
        return 0, ''

    async def increment_daily_request_count(self, user_id: int) -> None:
        """Increment the daily request count for a user."""
        User = Query()
        result = self.state_table.get(User.user_id == user_id)
        if result:
            new_count = result.get('daily_requests', 0) + 1
            self.state_table.update({'daily_requests': new_count}, User.user_id == user_id)
        else:
            # If no state exists, create one
            self.state_table.insert({'user_id': user_id, 'daily_requests': 1, 'last_request_date': ''})
        logger.info(f"Incremented daily request count for user {user_id}.")

    async def reset_daily_request_count(self, user_id: int, new_date: str) -> None:
        """Reset the daily request count for a user with the new date."""
        User = Query()
        result = self.state_table.get(User.user_id == user_id)
        if result:
            self.state_table.update({'daily_requests': 0, 'last_request_date': new_date}, User.user_id == user_id)
        else:
            # If no state exists, create one
            self.state_table.insert({'user_id': user_id, 'daily_requests': 0, 'last_request_date': new_date})
        logger.info(f"Reset daily request count for user {user_id} on {new_date}.")

class Config:
    """Configuration management using environment variables."""

    def __init__(self, env_file: str = ".env"):
        # Load environment variables from .env in local development
        if os.getenv("ENV", "local").lower() == "local":
            from dotenv import load_dotenv
            load_dotenv(env_file)

        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
        self.GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
        self.GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
        self.SCOPES = [
            'https://www.googleapis.com/auth/calendar.events',
            'https://www.googleapis.com/auth/calendar.settings.readonly'
        ]
        self.SENTRY_DSN = os.getenv("SENTRY_DSN")
        self.ENV = os.getenv("ENV", "local")

        # Firestore configuration
        self.FIRESTORE_PROJECT = os.getenv("FIRESTORE_PROJECT")
        self.FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "google_access_tokens")

        encryption_key_hex = os.getenv("ENCRYPTION_KEY")
        if not encryption_key_hex:
            raise ValueError("ENCRYPTION_KEY environment variable not set.")

        self.ENCRYPTION_KEY = bytes.fromhex(encryption_key_hex)
        if len(self.ENCRYPTION_KEY) != 32:
            raise ValueError("ENCRYPTION_KEY must be 32 bytes (64 hex characters) for AES-256.")

        # Initialize cipher (will use a random IV for each encryption)
        self.backend = default_backend()

        self.JWT_SECRET = os.getenv("JWT_SECRET")
        self.JWT_ALGORITHM = "HS256"
        if not self.JWT_SECRET:
            raise ValueError("JWT_SECRET environment variable not set.")

        # Initialize Sentry SDK for production
        if self.SENTRY_DSN:
            sentry_logging = LoggingIntegration(
                level="INFO",        # Capture info and above as breadcrumbs
                event_level="ERROR"  # Send errors as events
            )
            sentry_sdk.init(
                dsn=self.SENTRY_DSN,
                integrations=[
                    sentry_logging,
                    QuartIntegration(),
                ],
                traces_sample_rate=1.0,  # Adjust based on your needs
                profiles_sample_rate=1.0,  # Profile 100% of sampled transactions
                environment=self.ENV
            )
            logger.info("Sentry SDK initialized.")

    def get_repository(self) -> IRepository:
        """Returns the appropriate repository based on the environment."""
        if self.ENV.lower() == "local":
            logger.info("Using FileSystemRepository for local environment.")
            return TinyDbRepository()
        else:
            logger.info("Using DynamoDbRepository for production environment.")
            return DynamoDbRepository(encryption_key=self.ENCRYPTION_KEY)

class BaseHandler(ABC):
    """Abstract base class for all handlers."""

    @abstractmethod
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
        pass

async def download_audio_in_memory(message: Message, user_id: int) -> bytes:
    """Downloads audio from a Telegram message into RAM."""
    if message.voice:
        file = await message.voice.get_file()
        mime_type = message.voice.mime_type
    elif message.audio:
        file = await message.audio.get_file()
        mime_type = message.audio.mime_type
    else:
        raise ValueError("Message does not contain audio or voice data.")

    # Download the file as bytes
    # Note: For async, this should be handled appropriately
    audio_bytes = await file.download_as_bytearray()  # IT'S INCORRECT for new versions of python-telegram-bot
    logger.info(f"Downloaded audio for user {user_id}, MIME type: {mime_type}")

    return bytes(audio_bytes)

# Pydantic Models
class CalendarEvent(BaseModel):
    name: str = Field(..., description="Name of the event")
    date: str = Field(..., description="Date of the event in YYYY-MM-DD format")
    time: str = Field(..., description="Time of the event in HH:MM (24-hour) format")
    timezone: str = Field(..., description="IANA timezone string")
    description: Optional[str] = Field(None, description="Description of the event")
    connection_info: Optional[str] = Field(
        None, description="Connection information for the event, such as links and passwords"
    )

class EventIdentifier(BaseModel):
    event_text: str = Field(
        ..., description="Info to identify the event to delete. All info that helps to identify the event in one string."
    )

class RescheduleDetails(BaseModel):
    event_text: str = Field(
        ..., description="Info to identify the event to reschedule. All info that helps to identify the event in one string."
    )
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

    def __init__(self, config: Config, repository: IRepository):
        self.config = config
        self.repository = repository
        self.SCOPES = self.config.SCOPES
        self.logger = logger
        self.users_timezone_cache: dict[int, str] = {}
        self.jwt_secret = config.JWT_SECRET
        self.jwt_algorithm = config.JWT_ALGORITHM

    def generate_auth_url(self, user_id: int) -> str:
        """Generates the OAuth 2.0 authorization URL for the user with JWT-encoded state."""
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

        # Create JWT token containing the user_id and expiration time
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=1),  # Token expires in 1 hour
            "iat": datetime.utcnow(),
        }
        state_jwt = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        authorization_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=state_jwt,  # Use JWT as state
            prompt='consent'  # Forces consent screen to ensure refresh_token is received
        )

        self.logger.info(f"Generated auth URL with JWT state for user {user_id}.")
        return authorization_url

    async def get_credentials(self, user_id: int, authorization_code: Optional[str] = None) -> Credentials | None:
        """Gets or refreshes credentials for a user."""
        max_retries = 2
        retry_delay = 3  # seconds
        creds = None

        if authorization_code:
            # Existing logic to fetch tokens using authorization_code
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

            with start_span_smart(op="google_calendar", description="Get Credentials"):
                # Fetch token using authorization_code
                try:
                    flow.fetch_token(code=authorization_code)
                    creds = flow.credentials

                    # Save credentials using the repository
                    await self.repository.save_tokens(user_id, json.loads(creds.to_json()))
                    self.logger.info(f"Credentials obtained and saved for user {user_id}.")
                except Exception as e:
                    self.logger.error(f"Error fetching tokens for user {user_id}: {e}")
                    sentry_sdk.capture_exception(e)
                    return None

        else:
            tokens = await self.repository.get_tokens(user_id)
            if tokens:
                creds = Credentials.from_authorized_user_info(info=tokens, scopes=self.SCOPES)
                if creds and creds.expired and creds.refresh_token:
                    for attempt in range(1, max_retries + 1):
                        try:
                            creds.refresh(Request())
                            # Save the refreshed credentials
                            await self.repository.save_tokens(user_id, json.loads(creds.to_json()))
                            self.logger.info(f"Credentials refreshed and saved for user {user_id}.")
                            break  # Exit retry loop on success
                        except RefreshError as e:
                            # Token has been revoked or expired
                            self.logger.error(f"Failed to refresh tokens for user {user_id}: {e}")
                            sentry_sdk.capture_exception(e)
                            await self.repository.delete_tokens(user_id)
                            await self.repository.delete_user_state(user_id)
                            return None
                        except HTTPError as e:
                            # Network or other HTTP errors
                            self.logger.error(f"HTTP error when refreshing tokens for user {user_id}: {e}")
                            sentry_sdk.capture_exception(e)
                            if attempt < max_retries:
                                self.logger.info(
                                    f"Retrying token refresh in {retry_delay} seconds... (Attempt {attempt}/{max_retries})")
                                await asyncio.sleep(retry_delay)
                            else:
                                self.logger.error(f"Max retries reached. Unable to refresh tokens for user {user_id}.")
                                return None
            else:
                self.logger.debug(f"No tokens found for user {user_id}.")
                return None

        return creds

    async def create_event(self, event: CalendarEvent, user_id: int) -> Dict[str, Any]:
        """Creates an event in Google Calendar."""
        with start_span_smart(op="google_calendar", description="Creating event"):
            try:
                creds = await self.get_credentials(user_id)
                if not creds:
                    raise ValueError("Invalid credentials.")
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

                with start_span_smart(op="google_calendar", description="Inserting event"):
                    created_event = await asyncio.to_thread(
                        service.events().insert, calendarId='primary', body=event_body
                    )
                    created_event = await asyncio.to_thread(created_event.execute)
                    self.logger.info(f"Created event: {created_event}")
                    return created_event
            except Exception as e:
                self.logger.error(f"Failed to create event: {e}")
                sentry_sdk.capture_exception(e)
                raise e

    async def delete_event(self, event_text: str, user_id: int, update: Update) -> Dict[str, Any]:
        """Deletes an event from Google Calendar based on identifier using LLM for relevance."""
        with start_span_smart(op="google_calendar", description="Deleting event"):
            try:
                creds = await self.get_credentials(user_id)
                if not creds:
                    raise ValueError("Invalid credentials.")
                service = build('calendar', 'v3', credentials=creds)

                # Fetch events to find the most relevant one
                now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
                events_result = await asyncio.to_thread(
                    service.events().list,
                    calendarId='primary',
                    timeMin=now,
                    maxResults=20,
                    singleEvents=True,
                    orderBy='startTime'
                )
                events_result = await asyncio.to_thread(events_result.execute)
                events = events_result.get('items', [])

                if len(events) == 0:
                    self.logger.info("No events found in the calendar.")
                    return {"status": "not_found"}

                await update.message.reply_text(f"Searching in {len(events)} events for the most relevant one...")

                # Utilize LLM to find the most relevant event based on identifier
                relevant_event: Optional[RelevantEventResponse] = await self.find_relevant_event_with_llm(event_text, events)

                if not relevant_event or not relevant_event.found_something:
                    # TODO: Retry with more maxResults value
                    self.logger.info("LLM did not find anything relevant.")
                    return {"status": "not_found"}

                if relevant_event.uncertain_match:
                    self.logger.info("LLM is not sure about the event. Awaiting user confirmation.")
                    return {"status": "requires_confirmation", "event": relevant_event}

                event_id = relevant_event.event_id
                await asyncio.to_thread(service.events().delete, calendarId='primary', eventId=event_id).execute()
                self.logger.info(f"Deleted event: {relevant_event}")
                return {"status": "deleted", "event": relevant_event}
            except Exception as e:
                self.logger.error(f"Failed to delete event: {e}")
                sentry_sdk.capture_exception(e)
                return {"status": "error", "error": str(e)}

    async def get_user_timezone(self, user_id: int) -> Optional[str]:
        """Fetch the user's timezone using Google Calendar API."""
        if self.users_timezone_cache.get(user_id):
            return self.users_timezone_cache[user_id]

        creds = await self.get_credentials(user_id)
        service = build('calendar', 'v3', credentials=creds)
        timezone: dict[str, str] = service.settings().get(setting='timezone').execute()
        return timezone['value']

    async def reschedule_event(self, details: Dict[str, Any], user_id: int, update: Update) -> Dict[str, Any]:
        """Reschedules an existing event based on identifier and new details using LLM for relevance."""
        # TODO: Reimplement this feature
        with start_span_smart(op="google_calendar", description="Rescheduling event"):
            try:
                event_text = details.get('event_text')
                new_date = details.get('new_date')
                new_time = details.get('new_time')
                new_timezone = details.get('new_timezone', str(get_localzone()))

                creds = await self.get_credentials(user_id)
                if not creds:
                    raise ValueError("Invalid credentials.")
                service = build('calendar', 'v3', credentials=creds)

                # Fetch events to find the most relevant one
                now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
                events_result = await asyncio.to_thread(
                    service.events().list,
                    calendarId='primary',
                    timeMin=now,
                    maxResults=20,
                    singleEvents=True,
                    orderBy='startTime'
                )
                events_result = await asyncio.to_thread(events_result.execute)
                events = events_result.get('items', [])

                if len(events) == 0:
                    self.logger.info("No events found in the calendar.")
                    return {"status": "not_found"}

                await update.message.reply_text(f"Searching in {len(events)} events for the most relevant one...")

                # Utilize LLM to find the most relevant event based on identifier
                relevant_event: Optional[RelevantEventResponse] = await self.find_relevant_event_with_llm(event_text, events)

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
                    new_timezone = str(await self.get_user_timezone(user_id))

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

                updated_event = await asyncio.to_thread(
                    service.events().update,
                    calendarId='primary',
                    eventId=event_id,
                    body=new_event
                )
                updated_event = await asyncio.to_thread(updated_event.execute)
                self.logger.info(f"Rescheduled event: {updated_event}")
                return {"status": "rescheduled", "event": updated_event}
            except Exception as e:
                self.logger.error(f"Failed to reschedule event: {e}")
                sentry_sdk.capture_exception(e)
                return {"status": "error", "error": str(e)}

    async def find_relevant_event_with_llm(self, event_text: str, events: List[Dict[str, Any]]) -> Optional[RelevantEventResponse]:
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
            with start_span_smart(op="llm", description="Finding relevant event"):
                response = await litellm.acompletion(
                    model=SEARCH_MODEL,
                    messages=messages,
                    temperature=0,  # IT'S VERY IMPORTANT TO SET TEMPERATURE TO 0.
                    response_format=RelevantEventResponse,
                    retries=3,
                    fallbacks=[FallbacksModels.SearchFallbacks],
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

    def __init__(self, config: Config, google_calendar_service: GoogleCalendarService):
        self.config = config
        self.google_calendar_service = google_calendar_service
        self.logger = logger
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

    async def get_completion(self, model: str, messages: List[Dict[str, Any]], tool_choice: str = "auto") -> Dict[str, Any]:
        """Make a completion request to LiteLLM with function calling."""
        try:
            response = await litellm.acompletion(
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

    async def execute_function(
        self,
        function_name: str,
        function_args: Dict[str, Any],
        user_id: int,
        update: Update
    ) -> Dict[str, Any]:
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
            await update.message.reply_text(f"{action_info}")

        # Proceed to execute the function
        try:
            if function_name == "add_event":
                event = CalendarEvent(**function_args)
                result = await self.google_calendar_service.create_event(event, user_id)
                self.logger.info(f"Event added: {result}")
                return {"status": "added", "event": result}
            elif function_name == "delete_event":
                identifier = EventIdentifier(**function_args)
                result = await self.google_calendar_service.delete_event(identifier.event_text, user_id, update)
                return result
            elif function_name == "reschedule_event":
                details = RescheduleDetails(**function_args)
                result = await self.google_calendar_service.reschedule_event(details.model_dump(), user_id, update)
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

    def __init__(
        self,
        litellm_service: LiteLLMService,
        google_calendar_service: GoogleCalendarService,
        repository: IRepository,
    ):
        self.litellm_service = litellm_service
        self.google_calendar_service = google_calendar_service
        self.repository = repository
        self.logger = logger

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        with sentry_sdk.start_transaction(op="handler", name="InputHandler.handle"):
            self.logger.info("Input handler triggered")
            user = update.effective_user
            user_id: int = user.id

            # Set Sentry user context
            sentry_sdk.set_user({"id": user_id, "username": user.username, "first_name": user.first_name})

            # Check if user has valid credentials
            try:
                user_state = await self.repository.get_user_state(user_id)
                if not user_state or not user_state.get("authenticated"):
                    await update.message.reply_text(
                        "🔒 You need to authenticate before accessing the bot's functionalities.\n"
                        "Please enter password to authenticate."
                    )
                    return BotStates.AUTHENTICATE

                current_date = datetime.utcnow().strftime("%Y-%m-%d")

                # Retrieve user's daily request count and last request date
                daily_count, last_request_date = await self.repository.get_daily_request_count(user_id)

                if last_request_date != current_date:
                    # Reset count for new day
                    await self.repository.reset_daily_request_count(user_id, current_date)
                    daily_count = 0
                    self.logger.info(f"Reset daily request count for user {user_id} for new day.")

                if daily_count >= 20:
                    now = datetime.utcnow()
                    next_day = now + timedelta(days=1)
                    reset_time = datetime(year=next_day.year, month=next_day.month, day=next_day.day)
                    time_remaining = reset_time - now
                    hours, remainder = divmod(int(time_remaining.total_seconds()), 3600)
                    minutes, _ = divmod(remainder, 60)
                    await update.message.reply_text(
                        f"🚫 You have reached your daily limit of 20 requests. Please try again in {hours} hours and {minutes} minutes."
                    )
                    return ConversationHandler.END

                # Increment the daily request count
                await self.repository.increment_daily_request_count(user_id)

                creds = await self.google_calendar_service.get_credentials(user_id)
                if not creds or not creds.valid:
                    # Generate auth URL and prompt user
                    auth_url = self.google_calendar_service.generate_auth_url(user_id)
                    await update.message.reply_text(
                        "🔒 Your Google Calendar authorization has expired or been revoked. Please re-authorize access to continue using the bot's functionalities.",
                        reply_markup=telegram.InlineKeyboardMarkup([
                            [telegram.InlineKeyboardButton("Re-authorize", url=auth_url)]
                        ])
                    )
                    return ConversationHandler.END

                # Inform the user that the message is being processed
                with start_span_smart(op="telegram", description="Send Processing Message"):
                    await update.message.reply_text("Processing your request...")

                # Handle voice or audio messages
                is_voice = bool(update.message.voice or update.message.audio)
                if is_voice:
                    duration = update.message.voice.duration if update.message.voice else update.message.audio.duration
                    if duration > 20:
                        await update.message.reply_text(
                            "Your audio message is longer than 20 seconds. Please send a shorter audio message (maximum 20 seconds)."
                        )
                        return ConversationHandler.END

                    # Handle voice or audio messages: download and transcribe
                    audio_bytes = await download_audio_in_memory(update.message, user_id)
                    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
                    messages = [
                        {"role": "system", "content": self.get_system_prompt(await self.google_calendar_service.get_user_timezone(user_id))},
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
                    user_message: Optional[str] = update.message.text_markdown_v2_urled or update.message.caption_markdown_v2_urled
                    self.logger.info(f"Extracted user message: {user_message}")

                    if not user_message and not update.message.voice and not update.message.audio:
                        await update.message.reply_text("Unsupported message type. Please send text, audio, or voice messages.")
                        return ConversationHandler.END

                    messages = [
                        {"role": "system", "content": self.get_system_prompt(await self.google_calendar_service.get_user_timezone(user_id))},
                        {"role": "user", "content": user_message},
                    ]

                try:
                    model = COMMANDS_MODEL_VOICE if is_voice else COMMANDS_MODEL_TEXT
                    response: Dict[str, Any] = {}
                    with start_span_smart(op="llm", description="Get completion"):
                        response = await self.litellm_service.get_completion(
                            model=model,
                            messages=messages,
                            tool_choice="auto",
                        )

                    self.logger.info(f"LLM Response:\n{response}")
                    # IT'S 100% RIGHT TO USE response['choices'][0]['message'].
                    response_message = response['choices'][0]['message']
                    tool_calls: List[Dict[str, Any]] = self.litellm_service.parse_function_calls(response_message)

                    if tool_calls:
                        # A function needs to be called
                        for tool_call in tool_calls:
                            function_name: str = tool_call['function']['name']
                            function_args: Dict[str, Any] = json.loads(tool_call['function']['arguments'])

                            self.logger.info(f"Function call detected: {function_name} with args: {function_args}")

                            # Execute the function using LiteLLMService
                            result: Dict[str, Any] = await self.litellm_service.execute_function(
                                function_name, function_args, user_id, update
                            )

                            # Send action info to the user
                            action_info: Optional[str] = result.get("action_info")
                            if action_info:
                                with start_span_smart(op="telegram", description="Send Action Info"):
                                    await update.message.reply_text(f"About to perform: {action_info}")

                            # Check if confirmation is required
                            if result.get("status") == "requires_confirmation":
                                event: Optional[RelevantEventResponse] = result.get("event")
                                if event:
                                    with start_span_smart(op="telegram", description="Send Confirmation Request"):
                                        await update.message.reply_text(f"Found event: {event.event_name}")

                                with start_span_smart(op="telegram", description="Send Confirmation Request"):
                                    await update.message.reply_text("Do you want to proceed with this action? (Yes/No)")

                                # Save pending action to the repository
                                pending_action = {
                                    "function_name": function_name,
                                    "function_args": function_args
                                }
                                await self.repository.save_user_state(user_id, {"pending_action": pending_action})

                                return BotStates.CONFIRMATION
                            elif result.get("status") == "not_found":
                                await update.message.reply_text("Sorry, I couldn't find the event. Please try again.")
                            elif result.get("status") == "error":
                                await update.message.reply_text(f"Error: {result.get('error')}")
                            else:
                                # Action does not require confirmation; execute and inform the user
                                if result.get("status") == "deleted":
                                    event_name: str = result.get("event", {}).get('summary', 'the event')
                                    event_time: str = result.get("event", {}).get('start', {}).get('dateTime', 'the specified time')
                                    await update.message.reply_text(
                                        f"I have deleted the event '{event_name}' scheduled at '{event_time}'."
                                    )
                                elif result.get("status") == "rescheduled":
                                    event: Dict[str, Any] = result.get("event", {})
                                    event_name: str = event.get('summary', 'the event')
                                    new_time: str = event.get('start', {}).get('dateTime', 'the new specified time')
                                    await update.message.reply_text(
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
                                    await update.message.reply_text(reply_text)
                        return BotStates.PARSE_INPUT
                    else:
                        await update.message.reply_text("Sorry, I couldn't understand the request. Please try again.")
                        return ConversationHandler.END

                except Exception as e:
                    self.logger.error(f"Error with LiteLLM completion: {e}")
                    sentry_sdk.capture_exception(e)
                    await update.message.reply_text(f"An error occurred while processing your request.\n{e}")
                    return ConversationHandler.END

            except Exception as e:
                self.logger.error(f"Unexpected error in InputHandler: {e}")
                sentry_sdk.capture_exception(e)
                await update.message.reply_text(f"An unexpected error occurred.\n{e}")
                return ConversationHandler.END

    def get_system_prompt(self, iana_timezone: str) -> str:
        """Returns the system prompt for LLM."""
        now = datetime.now(pytz.timezone(iana_timezone))
        context = f"""
            Today's date and time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}
            User's Timezone: {iana_timezone} (can be different from the event timezone)
            Day of the week: {now.strftime('%A')}
        """
        return f"""
            You are an intelligent calendar assistant designed to help users manage their events efficiently. Your primary functions are adding new events, deleting existing events, and rescheduling events in the user's calendar.
            
            Here is the current context for the user's calendar:
            <context>
            {context}
            </context>
            
            When a user sends you a message, follow these steps:
            
            1. Analyze the message to determine the requested action (adding, deleting, or rescheduling an event).
            2. Extract all relevant event details provided by the user, such as event name, date, time, timezone, and any optional descriptions.
            3. If any necessary details are missing, attempt to infer them from the context of the user's message.
            4. Detect the language used in the event data and ensure all extracted information remains in that language.
        """

class ConfirmationHandler(BaseHandler):
    """Handler for confirming event actions when LLM is unsure."""

    def __init__(
        self,
        litellm_service: LiteLLMService,
        google_calendar_service: GoogleCalendarService,
        repository: IRepository,
    ):
        self.litellm_service = litellm_service
        self.google_calendar_service = google_calendar_service
        self.repository = repository
        self.logger = logger

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        with sentry_sdk.start_transaction(op="handler", name="ConfirmationHandler.handle"):
            try:
                self.logger.info("Confirmation handler triggered")
                response: str = update.message.text.lower()
                user_id: int = update.effective_user.id

                # Set Sentry user context
                sentry_sdk.set_user({"id": user_id})

                # Retrieve pending action from the repository
                user_state = await self.repository.get_user_state(user_id)
                if not user_state or "pending_action" not in user_state:
                    await update.message.reply_text("No pending actions to confirm.")
                    return ConversationHandler.END

                pending_action = user_state["pending_action"]

                if response in ['yes', 'y', 'да', 'д']:
                    function_name: str = pending_action['function_name']
                    function_args: Dict[str, Any] = pending_action['function_args']

                    # Inform the user that the action is being executed
                    await update.message.reply_text(f"Proceeding with '{function_name}' action.")

                    # Execute the function using LiteLLMService
                    result: Dict[str, Any] = await self.litellm_service.execute_function(
                        function_name, function_args, user_id, update
                    )

                    # Handle the result
                    if result.get("status") == "error":
                        await update.message.reply_text(f"Error: {result.get('error')}")
                    else:
                        # Send appropriate success messages
                        if result.get("status") == "deleted":
                            event_name: str = result.get("event", {}).get('summary', 'the event')
                            event_time: str = result.get("event", {}).get('start', {}).get('dateTime', 'the specified time')
                            await update.message.reply_text(
                                f"I have deleted the event '{event_name}' scheduled at '{event_time}'."
                            )
                        elif result.get("status") == "rescheduled":
                            event: Dict[str, Any] = result.get("event", {})
                            event_name: str = event.get('summary', 'the event')
                            new_time: str = event.get('start', {}).get('dateTime', 'the new specified time')
                            await update.message.reply_text(
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
                            await update.message.reply_text(reply_text)

                    # Clear the pending action from the repository
                    await self.repository.delete_user_state(user_id)
                    return ConversationHandler.END

                elif response in ['no', 'n', 'нет', 'н']:
                    await update.message.reply_text("Okay, action has been cancelled.")
                    # Clear the pending action from the repository
                    await self.repository.delete_user_state(user_id)
                    return ConversationHandler.END
                else:
                    await update.message.reply_text("Please respond with 'Yes' or 'No'. Do you want to proceed with this action?")
                    return BotStates.CONFIRMATION

            except Exception as e:
                self.logger.error(f"Unexpected error in ConfirmationHandler: {e}")
                sentry_sdk.capture_exception(e)
                await update.message.reply_text(f"An unexpected error occurred.\n{e}")
                return ConversationHandler.END

class CancelHandler(BaseHandler):
    """Handler for the /cancel command."""

    def __init__(self):
        self.logger = logger

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            self.logger.info("Cancel handler triggered")
            await update.message.reply_text("Operation cancelled.", reply_markup=ReplyKeyboardRemove())
            return ConversationHandler.END
        except Exception as e:
            self.logger.error(f"Error in CancelHandler: {e}")
            sentry_sdk.capture_exception(e)
            await update.message.reply_text(f"An error occurred while cancelling the operation.\n{e}")
            return ConversationHandler.END

class StartHandler(BaseHandler):
    """Handler for the /start command."""

    def __init__(self, repository: IRepository):
        self.logger = logger
        self.repository = repository

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            self.logger.info("Start command received")
            user_id: int = update.effective_user.id

            # Check if the user is already authenticated
            user_state = await self.repository.get_user_state(user_id)
            if user_state and user_state.get("authenticated"):
                await update.message.reply_text(
                    "👋 Welcome back! You are already authenticated. How can I assist you today?"
                )
                return BotStates.PARSE_INPUT

            await update.message.reply_text(
                "🔒 To access the bot's functionalities, please enter the access password."
            )
            return BotStates.AUTHENTICATE
        except Exception as e:
            self.logger.error(f"Error in StartHandler: {e}")
            sentry_sdk.capture_exception(e)
            await update.message.reply_text(f"⚠️ An error occurred while starting the bot.\n{e}")
            return ConversationHandler.END

class PasswordHandler(BaseHandler):
    """Handler for processing user password input."""

    def __init__(self, repository: IRepository):
        self.repository = repository
        self.logger = logger
        self.PASSWORD = "Эволюция Кода".lower()  # Define the password in lowercase for case-insensitive comparison

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        with sentry_sdk.start_transaction(op="handler", name="PasswordHandler.handle"):
            try:
                self.logger.info("Password handler triggered")
                user_id: int = update.effective_user.id
                user_input: str = update.message.text.strip().lower()

                if user_input == self.PASSWORD:
                    # Save authenticated status in user state
                    await self.repository.save_user_state(user_id, {"authenticated": True})
                    self.logger.info(f"User {user_id} authenticated successfully.")
                    await update.message.reply_text("✅ Authentication successful! You now have access to the bot's functionalities.")
                    await update.message.reply_text(
                        "Hi! I can help you manage your Google Calendar events.\n\n"
                        "You can:\n"
                        "- Add a new event by sending event details.\n"
                        "- Delete an event by sending a command like 'Delete [event name]'.\n"
                        "- Send audio messages with event details or commands."
                    )
                    return BotStates.PARSE_INPUT
                else:
                    self.logger.warning(f"User {user_id} provided an incorrect password.")
                    await update.message.reply_text("❌ Incorrect password. Please try again or type /cancel to exit.")
                    return BotStates.AUTHENTICATE  # Remain in the AUTHENTICATE state
            except Exception as e:
                self.logger.error(f"Error in PasswordHandler: {e}")
                sentry_sdk.capture_exception(e)
                await update.message.reply_text(f"⚠️ An error occurred during authentication.\n{e}")
                return ConversationHandler.END