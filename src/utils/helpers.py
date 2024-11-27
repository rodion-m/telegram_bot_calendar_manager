# utils/helpers.py
from telegram import Message
from typing import Optional

def download_audio(message: Message, user_id: int, logger) -> Optional[str]:
    """Downloads audio from a Telegram message."""
    try:
        if message.voice:
            file = message.voice.get_file()
            audio_path = f"audio_{user_id}.ogg"
        elif message.audio:
            file = message.audio.get_file()
            audio_path = f"audio_{user_id}.{message.audio.mime_type.split('/')[-1]}"
        else:
            return None
        file.download(audio_path)
        logger.debug(f"Downloaded audio to {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        return None