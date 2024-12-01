#server.py

import os

import flask
import sentry_sdk
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler

from src.logic import Config, LoggerSetup, GoogleCalendarService, LiteLLMService, StartHandler, ConfirmationHandler, \
    InputHandler, CancelHandler, BotStates

config = Config()
logger = LoggerSetup.setup_logging(config.LOGGING_MIN_LEVEL)

# Initialize repository based on environment
repository = config.get_repository()

# Initialize Google Calendar Service
google_calendar_service = GoogleCalendarService(config, logger, repository)

# Initialize LiteLLM Service
litellm_service = LiteLLMService(config, logger, google_calendar_service)

# Initialize Telegram bot
updater = Updater(config.TELEGRAM_TOKEN, use_context=True)
dispatcher = updater.dispatcher

# Initialize handlers
start_handler = StartHandler(logger)
input_handler = InputHandler(logger, litellm_service, google_calendar_service, repository)
confirmation_handler = ConfirmationHandler(logger, litellm_service, google_calendar_service, repository)
cancel_handler = CancelHandler(logger)

# Define the conversation handler with states PARSE_INPUT and CONFIRMATION
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


# Flask app for production
app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return 'OK'

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        update = flask.request.get_json(force=True)
        update = Update.de_json(update, dispatcher.bot)
        dispatcher.process_update(update)
        return "OK"
    except Exception as e:
        logger.error(f"Error in webhook: {e}")
        sentry_sdk.capture_exception(e)
        return "OK"

@app.route('/google_callback')
def google_callback():
    code = flask.request.args.get('code')
    state = flask.request.args.get('state')  # Contains user_id

    if not code or not state:
        return "Missing authorization code or state parameter.", 400

    try:
        user_id = int(state)
    except ValueError:
        return "Invalid state parameter.", 400

    try:
        creds = google_calendar_service.get_credentials(user_id, authorization_code=code)
        # At this point, credentials are stored. You can notify the user accordingly.
        return "Authorization successful! You can now return to the bot and continue."
    except Exception as e:
        logger.error(f"Error in Google callback: {e}")
        sentry_sdk.capture_exception(e)
        return "Authorization failed! Please try again.", 200


def run_local():
    logger.info("Bot started locally. Listening for messages...")

    # Start polling for updates from Telegram
    updater.start_polling()

    # Run the bot until Ctrl-C is pressed or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()


def run_production():
    # The entrypoint for App Engine
    app.run()


if __name__ == '__main__':
    environment = os.getenv("ENV", "local").lower()
    if environment == "local":
        run_local()
    else:
        run_production()