# main.py

import asyncio
import os

import sentry_sdk
import uvicorn
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, request, Response
from loguru import logger
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
    ExtBot
)

from src.logic import (
    Config,
    GoogleCalendarService,
    LiteLLMService,
    StartHandler,
    ConfirmationHandler,
    InputHandler,
    CancelHandler,
    BotStates,
)

# Initialize Flask App
app = Flask(__name__)

# Load Configuration
config = Config()

# Initialize Services
repository = config.get_repository()
google_calendar_service = GoogleCalendarService(config, repository)
litellm_service = LiteLLMService(config, google_calendar_service)

# Initialize Telegram Bot Application
application = Application.builder().token(config.TELEGRAM_TOKEN).build()

# Initialize Handlers
start_handler = StartHandler()
input_handler = InputHandler(litellm_service, google_calendar_service, repository)
confirmation_handler = ConfirmationHandler(litellm_service, google_calendar_service, repository)
cancel_handler = CancelHandler()

# Define the ConversationHandler with states PARSE_INPUT and CONFIRMATION
conv_handler = ConversationHandler(
    entry_points=[
        CommandHandler("start", start_handler.handle),
        MessageHandler(
            (filters.TEXT | filters.VOICE | filters.AUDIO | filters.Caption) & ~filters.COMMAND,
            input_handler.handle,
        ),
    ],
    states={
        BotStates.PARSE_INPUT: [
            MessageHandler(filters.TEXT | filters.VOICE | filters.AUDIO, input_handler.handle)
        ],
        BotStates.CONFIRMATION: [
            MessageHandler(
                filters.Regex(
                    r'^(Yes|No|Y|N|yes|no|y|n|да|Да|д|Нет|нет|н)$'
                ),
                confirmation_handler.handle,
            )
        ],
    },
    fallbacks=[CommandHandler("cancel", cancel_handler.handle)],
)

application.add_handler(conv_handler)

# Set Sentry DSN if available
if config.SENTRY_DSN:
    sentry_sdk.init(
        dsn=config.SENTRY_DSN,
        integrations=[],
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
        environment=config.ENV,
    )
    logger.info("Sentry SDK initialized.")

# Function to set the webhook
async def set_webhook():
    """Sets the webhook for the Telegram bot."""
    webhook_url = os.getenv("WEBHOOK_URL")  # Ensure this is set to your public HTTPS URL
    if not webhook_url:
        logger.error("WEBHOOK_URL environment variable not set.")
        return False
    webhook_endpoint = f"{webhook_url}/telegram"
    success = await application.bot.set_webhook(webhook_endpoint)
    if success:
        logger.info(f"Webhook set successfully to {webhook_endpoint}")
    else:
        logger.error(f"Failed to set webhook to {webhook_endpoint}")
    return success

# Flask route for the root
@app.route('/', methods=['GET'])
def index():
    return 'OK', 200

# Flask route for handling Google OAuth callback
@app.route('/google_callback', methods=['GET'])
async def google_callback():
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
        if creds:
            # Notify the user via Telegram
            await application.bot.send_message(
                chat_id=user_id,
                text="Authorization successful! You can now continue using the bot."
            )
            logger.info(f"User {user_id} authorized successfully.")
            return "Authorization successful! You can now return to the bot and continue.", 200
        else:
            logger.error("Failed to obtain credentials.")
            return "Authorization failed! Please try again.", 200
    except Exception as e:
        logger.error(f"Error in Google callback: {e}")
        sentry_sdk.capture_exception(e)
        return "Authorization failed! Please try again.", 200

# Flask route for handling Telegram webhooks
@app.route('/telegram', methods=['POST'])
async def telegram_webhook():
    try:
        update_json = request.get_json(force=True)
        if not update_json:
            logger.error("No update received")
            return Response(status=400)
        update = Update.de_json(update_json, application.bot)
        await application.process_update(update)
        logger.debug(f"Update processed: {update}")
        return Response(status=200)
    except Exception as e:
        logger.error(f"Error in Telegram webhook: {e}")
        sentry_sdk.capture_exception(e)
        return Response(status=500)

# Function to run the bot in local development mode using polling
async def delete_webhook():
    """Removes the webhook for the Telegram bot."""
    await application.bot.set_webhook(None)


def run_local():
    """Runs the bot in local development mode using polling."""
    logger.info("Starting bot in local mode with polling.")

    application.run_polling()

# Function to run the bot in production mode using webhooks
async def run_production():
    """Runs the bot in production mode using webhooks."""
    logger.info("Starting bot in production mode with webhooks.")

    # Set the webhook
    await set_webhook()

    # Serve the Flask app using Uvicorn
    server = uvicorn.Server(
        config=uvicorn.Config(
            app=WsgiToAsgi(app),
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8443)),
            log_level="info",
        )
    )

    await server.serve()

# Entry point
def main():
    environment = config.ENV.lower()
    if environment == "local":
        run_local()
    else:
        asyncio.run(run_production())

if __name__ == '__main__':
    main()