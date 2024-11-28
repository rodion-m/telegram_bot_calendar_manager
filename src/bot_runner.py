# bot_runner.py
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
)

from src.logic import Config, LoggerSetup, GoogleCalendarService, LiteLLMService, StartHandler, ConfirmationHandler, \
    InputHandler, CancelHandler


def main():
    # Initialize configuration
    config = Config()

    # Setup logging
    logger = LoggerSetup.setup_logging()

    # Initialize services
    google_calendar_service = GoogleCalendarService(config, logger)
    litellm_service = LiteLLMService(config, logger, google_calendar_service)

    # Initialize Telegram bot
    updater = Updater(config.TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Initialize handlers
    start_handler = StartHandler(logger)
    input_handler = InputHandler(logger, litellm_service, google_calendar_service)
    confirmation_handler = ConfirmationHandler(logger, litellm_service, google_calendar_service)
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