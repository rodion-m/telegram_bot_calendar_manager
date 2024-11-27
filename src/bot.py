# bot.py
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
)

from config import Config
from logging_setup import LoggerSetup
from services.google_calendar_service import GoogleCalendarService
from services.litellm_service import LiteLLMService
from handlers.start_handler import StartHandler
from handlers.input_handler import InputHandler
from handlers.confirmation_handler import ConfirmationHandler
from handlers.cancel_handler import CancelHandler
from src.handlers.input_handler_gemini import InputHandlerGemini
from src.services.google_genai_service import GoogleGenAIService


def main():
    # Initialize configuration
    config = Config()

    # Setup logging
    logger = LoggerSetup.setup_logging()

    # Initialize services
    google_calendar_service = GoogleCalendarService(config, logger)
    google_genai_service = GoogleGenAIService(api_key=config.GEMINI_API_KEY, logger=logger)

    # Initialize Telegram bot
    updater = Updater(config.TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Initialize handlers
    start_handler = StartHandler(logger)
    input_handler = InputHandlerGemini(logger, google_genai_service, google_calendar_service)
    confirmation_handler = ConfirmationHandler(logger, google_calendar_service)
    cancel_handler = CancelHandler(logger)

    # Define conversation states
    PARSE_INPUT, ASK_TIMEZONE, CONFIRMATION = range(3)

    # Define the conversation handler with the states PARSE_INPUT, ASK_TIMEZONE, CONFIRMATION
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start_handler.handle),
            MessageHandler(
                (Filters.text | Filters.voice | Filters.audio | Filters.caption) & ~Filters.command,
                input_handler.handle
            )
        ],
        states={
            PARSE_INPUT: [
                MessageHandler(
                    (Filters.text | Filters.voice | Filters.audio | Filters.caption) & ~Filters.command,
                    input_handler.handle
                ),
            ],
            ASK_TIMEZONE: [
                MessageHandler(
                    Filters.text & ~Filters.command,
                    input_handler.handle  # You might want a separate handler for receiving timezone
                ),
            ],
            CONFIRMATION: [
                MessageHandler(
                    Filters.text & ~Filters.command,
                    confirmation_handler.handle
                ),
            ],
            # Additional states for delete and reschedule can be added here
        },
        fallbacks=[CommandHandler('cancel', cancel_handler.handle)],
    )

    dispatcher.add_handler(conv_handler)

    logger.info("Bot started. Listening for messages...")

    # Start polling for updates from Telegram
    updater.start_polling()

    # Run the bot until Ctrl-C is pressed or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()