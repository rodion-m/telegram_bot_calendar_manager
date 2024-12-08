# main.py

import signal

import uvicorn
from quart import Quart, request, Response
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from src.logic import *

# Initialize Quart App
app = Quart(__name__)

# Load Configuration
config = Config()

# Initialize Services
repository = config.get_repository()
google_calendar_service = GoogleCalendarService(config, repository)
litellm_service = LiteLLMService(config, google_calendar_service)

# Initialize Telegram Bot Application
application = Application.builder().token(config.TELEGRAM_TOKEN).build()

# Initialize Handlers
start_handler = StartHandler(repository)
password_handler = PasswordHandler(repository=repository)
input_handler = InputHandler(litellm_service, google_calendar_service, repository)
confirmation_handler = ConfirmationHandler(litellm_service, google_calendar_service, repository)
cancel_handler = CancelHandler()

# Define the ConversationHandler with states PARSE_INPUT and CONFIRMATION
conv_handler = ConversationHandler(
    entry_points=[
        CommandHandler("start", start_handler.handle),
        MessageHandler(
            (filters.TEXT | filters.VOICE | filters.AUDIO | filters.CAPTION) & ~filters.COMMAND,
            input_handler.handle,
        ),
    ],
    states={
        BotStates.AUTHENTICATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, password_handler.handle)],
        BotStates.PARSE_INPUT: [
            MessageHandler((filters.TEXT | filters.VOICE | filters.AUDIO | filters.CAPTION) & ~filters.COMMAND, input_handler.handle)
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
    allow_reentry=False
)

application.add_handler(conv_handler)

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

# Quart route for the root
@app.route('/', methods=['GET'])
async def index():
    return 'OK', 200

# Quart route for handling Google OAuth callback
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
        creds = await google_calendar_service.get_credentials(user_id, authorization_code=code)
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

# Quart route for handling Telegram webhooks
@app.route('/telegram', methods=['POST'])
async def telegram_webhook():
    try:
        update_json = await request.get_json(force=True)
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
async def run_local():
    """Runs the bot in local development mode using polling."""
    logger.info("Starting bot in local mode with polling.")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Start the Quart server as a background task
    async def run_quart():
        """Runs the Quart server."""
        logger.info("Starting Quart server.")
        await app.run_task(
            host='0.0.0.0',
            port=8443,
            debug=True,
            certfile=f"{current_dir}/cert.pem",
            keyfile=f"{current_dir}/key.pem",
        )

    quart_task = asyncio.create_task(run_quart())

    logger.info("Quart server started.")

    # Start the Telegram bot polling
    logger.info("Running bot polling.")
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        logger.info("Bot polling started.")

        # Create an asyncio Event to wait indefinitely
        stop_event = asyncio.Event()

        # Define signal handler to set the stop_event
        def shutdown_handler():
            logger.info("Shutdown signal received. Stopping bot...")
            stop_event.set()

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown_handler)
            except NotImplementedError:
                # Signal handling might not be implemented on some platforms (e.g., Windows)
                logger.warning(f"Signal handling for {sig} not implemented on this platform.")

        # Wait until a shutdown signal is received
        await stop_event.wait()

        logger.info("Stopping bot.")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Bot stopped successfully.")

    # Cancel the Quart task
    quart_task.cancel()
    try:
        await quart_task
    except asyncio.CancelledError:
        logger.info("Quart server stopped.")

# Function to run the bot in production mode using webhooks
async def run_production():
    """Runs the bot in production mode using webhooks."""
    logger.info("Starting bot in production mode with webhooks.")

    # Set the webhook
    await set_webhook()

    # Serve the Quart app using Uvicorn
    server = uvicorn.Server(
        config=uvicorn.Config(
            app=app,
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
        asyncio.run(run_local())
    else:
        asyncio.run(run_production())

if __name__ == '__main__':
    main()