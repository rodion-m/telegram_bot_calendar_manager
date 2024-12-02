"""
Server module for Telegram bot running on Google App Engine.
Handles webhook-based updates and Google Calendar authentication.
"""

import os
from queue import Queue
from typing import Optional

import flask
import sentry_sdk
from flask import Flask, request, Response
from telegram import Update, Bot
from telegram.ext import (
    Dispatcher,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
)

from src.logic import (
    Config,
    LoggerSetup,
    GoogleCalendarService,
    LiteLLMService,
    StartHandler,
    ConfirmationHandler,
    InputHandler,
    CancelHandler,
    BotStates,
)

# Initialize Flask app
app = Flask(__name__)

# Global variables for services
config: Optional[Config] = None
dispatcher: Optional[Dispatcher] = None
logger = None
google_calendar_service: Optional[GoogleCalendarService] = None


def create_app() -> Flask:
    """
    Initialize the Flask application and all required services.
    Returns:
        Flask: Configured Flask application
    """
    global config, dispatcher, logger, google_calendar_service

    # Initialize core services
    config = Config() # Sentry initialization is done in Config
    logger = LoggerSetup.setup_logging(config.LOGGING_MIN_LEVEL)

    # Initialize services
    repository = config.get_repository()
    google_calendar_service = GoogleCalendarService(config, logger, repository)
    litellm_service = LiteLLMService(config, logger, google_calendar_service)

    # Initialize bot handlers
    start_handler = StartHandler(logger)
    input_handler = InputHandler(logger, litellm_service, google_calendar_service, repository)
    confirmation_handler = ConfirmationHandler(logger, litellm_service, google_calendar_service, repository)
    cancel_handler = CancelHandler(logger)

    bot = Bot(token=config.TELEGRAM_TOKEN)

    # Initialize the Update Queue
    update_queue = Queue()

    # Initialize dispatcher without updater (webhook mode)
    dispatcher = Dispatcher(
        bot=bot,
        update_queue=update_queue,
        use_context=True,
        context_types=None  # Uses default ContextTypes
    )

    # Configure conversation handler
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start_handler.handle),
            MessageHandler(
                (Filters.text | Filters.voice | Filters.audio | Filters.caption) & ~Filters.command,
                input_handler.handle
            )
        ],
        states={
            BotStates.PARSE_INPUT: [
                MessageHandler(Filters.text | Filters.voice | Filters.audio, input_handler.handle)
            ],
            BotStates.CONFIRMATION: [
                MessageHandler(
                    Filters.regex('^(Yes|No|Y|N|yes|no|y|n|да|Да|д|Нет|нет|н)$'),
                    confirmation_handler.handle
                )
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel_handler.handle)],
    )

    dispatcher.add_handler(conv_handler)

    return app


# Create the Flask app
app = create_app()


@app.route('/_ah/warmup')
def warmup() -> str:
    """Handle App Engine warmup requests."""
    return 'OK'


@app.route('/', methods=['GET'])
def health_check() -> str:
    """Basic health check endpoint."""
    return 'OK'


@app.route('/webhook', methods=['POST'])
def webhook() -> str:
    """
    Handle incoming webhook requests from Telegram.
    Returns:
        str: Status response
    """
    if not dispatcher:
        logger.error("Dispatcher not initialized")
        return "Error: Dispatcher not initialized", 500

    try:
        update = Update.de_json(request.get_json(force=True), dispatcher.bot)
        dispatcher.process_update(update)
        return "OK"
    except Exception as e:
        logger.error(f"Error processing update: {e}")
        sentry_sdk.capture_exception(e)
        return "Error processing update", 500


@app.route('/google_callback')
def google_callback() -> Response:
    """
    Handle Google OAuth2 callback requests.
    Returns:
        Response: Flask response object
    """
    code = request.args.get('code')
    state = request.args.get('state')

    if not code or not state:
        return Response("Missing required parameters", status=400)

    try:
        user_id = int(state)
        creds = google_calendar_service.get_credentials(user_id, authorization_code=code)
        return Response(
            "Authorization successful! You can now return to the bot and continue.",
            status=200
        )
    except ValueError:
        return Response("Invalid state parameter", status=400)
    except Exception as e:
        logger.error(f"Error in Google callback: {e}")
        sentry_sdk.capture_exception(e)
        return Response("Authorization failed! Please try again.", status=500)


if __name__ == '__main__':
    # Local development server
    if os.getenv("ENV", "production").lower() == "local":
        app.run(host='localhost', port=8080, debug=True)
    else:
        app.run()