# main.py

import os
import sys
from queue import Queue
from threading import Thread

import sentry_sdk
from flask import Flask, request
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters, ConversationHandler

from src.logic import (
    Config,
    GoogleCalendarService,
    LiteLLMService,
    StartHandler,
    ConfirmationHandler,
    InputHandler,
    CancelHandler,
    BotStates
)
from loguru import logger

logger.add(sys.stdout, level="INFO")
logger.add(sys.stderr, level="ERROR")

# Initialize Flask App
app = Flask(__name__)

# Load Configuration
config = Config()

# Initialize Repository based on environment
repository = config.get_repository()

# Initialize Google Calendar Service
google_calendar_service = GoogleCalendarService(config, repository)

# Initialize LiteLLM Service
litellm_service = LiteLLMService(config, google_calendar_service)

# Initialize Telegram Bot
bot = Bot(token=config.TELEGRAM_TOKEN)

# Initialize the Update Queue
update_queue = Queue()

# Initialize Dispatcher with the update_queue
# THE Dispatcher call parameters ordering is 100% correct.
dispatcher = Dispatcher(
    bot=bot,
    update_queue=update_queue,
    workers=4,  # Number of worker threads; adjust as needed
    use_context=True,
    context_types=None  # Uses default ContextTypes
)

# Initialize Handlers
start_handler = StartHandler()
input_handler = InputHandler(litellm_service, google_calendar_service, repository)
confirmation_handler = ConfirmationHandler(litellm_service, google_calendar_service, repository)
cancel_handler = CancelHandler()

# Define the ConversationHandler with states PARSE_INPUT and CONFIRMATION
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
            MessageHandler(Filters.regex('^(Yes|No|Y|N|yes|no|y|n|да|Да|д|Нет|нет|н)$'), confirmation_handler.handle)
        ],
    },
    fallbacks=[CommandHandler('cancel', cancel_handler.handle)],
)

dispatcher.add_handler(conv_handler)

# Function to set the webhook
def set_webhook():
    """Sets the webhook for the Telegram bot."""
    webhook_url = os.getenv("WEBHOOK_URL")  # Ensure this is set to your public HTTPS URL
    if not webhook_url:
        logger.error("WEBHOOK_URL environment variable not set.")
        return
    webhook_endpoint = f"{webhook_url}/webhook"
    success = bot.set_webhook(url=webhook_endpoint)
    if success:
        logger.info(f"Webhook set successfully to {webhook_endpoint}")
    else:
        logger.error(f"Failed to set webhook to {webhook_endpoint}")


# Function to start the Dispatcher
def start_dispatcher():
    """Starts the Dispatcher in a separate thread."""
    dispatcher_thread = Thread(target=dispatcher.start, name="DispatcherThread", daemon=True)
    dispatcher_thread.start()
    logger.info("Dispatcher thread started.")


# Flask route for the root
@app.route('/', methods=['GET'])
def index():
    return 'OK', 200


# Flask route for handling Telegram webhooks
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        update_json = request.get_json(force=True)
        update = Update.de_json(update_json, bot)
        # Enqueue the update for the Dispatcher to process
        update_queue.put(update)
        logger.debug(f"Update enqueued: {update}")
        return "OK", 200
    except Exception as e:
        logger.error(f"Error in webhook: {e}")
        sentry_sdk.capture_exception(e)
        return "Internal Server Error", 500


# Flask route for handling Google OAuth callback
@app.route('/google_callback', methods=['GET'])
def google_callback():
    code = request.args.get('code')
    state = request.args.get('state')  # Contains user_id

    if not code or not state:
        logger.warning("Missing authorization code or state parameter.")
        return "Missing authorization code or state parameter.", 400

    try:
        user_id = int(state)
    except ValueError:
        logger.error("Invalid state parameter.")
        return "Invalid state parameter.", 400

    try:
        creds = google_calendar_service.get_credentials(user_id, authorization_code=code)
        # At this point, credentials are stored. Notify the user via Telegram.
        bot.send_message(chat_id=user_id, text="Authorization successful! You can now continue using the bot.")
        logger.info(f"User {user_id} authorized successfully.")
        return "Authorization successful! You can now return to the bot and continue.", 200
    except Exception as e:
        logger.error(f"Error in Google callback: {e}")
        sentry_sdk.capture_exception(e)
        return "Authorization failed! Please try again.", 200


def run_local():
    """Runs the bot in local development mode using polling."""
    logger.info("Starting bot in local mode with polling.")

    from telegram.ext import Updater

    # Initialize Updater for polling
    updater = Updater(token=config.TELEGRAM_TOKEN, use_context=True)
    local_dispatcher = updater.dispatcher

    # Add handlers to the local dispatcher
    local_dispatcher.add_handler(conv_handler)

    # Start polling
    updater.start_polling()
    logger.info("Bot is polling for updates.")
    updater.idle()


def run_production():
    """Runs the bot in production mode using webhooks."""
    # Start the Dispatcher
    start_dispatcher()

    # Set the webhook
    set_webhook()

    logger.info("Bot is running in production mode with webhooks.")

    app.run()


# environment = config.ENV.lower()
if __name__ == '__main__':
    run_local()
else:
    run_production()