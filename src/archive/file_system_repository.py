import json
import os
from typing import Dict, Any, Optional

from loguru import logger

from logic import IRepository, Config


class FileSystemRepository(IRepository):
    """Repository implementation using the local filesystem."""

    def __init__(self, config: 'Config'):
        self.token_dir = os.path.abspath("../google_tokens")
        os.makedirs(self.token_dir, exist_ok=True)
        self.logger = logger

    def _get_token_path(self, user_id: int) -> str:
        return os.path.join(self.token_dir, f"token_{user_id}.json")

    def _get_state_path(self, user_id: int) -> str:
        return os.path.join(self.token_dir, f"state_{user_id}.json")

    def save_tokens(self, user_id: int, tokens: Dict[str, Any]) -> None:
        path = self._get_token_path(user_id)
        with open(path, 'w') as f:
            json.dump(tokens, f)
        self.logger.info(f"Saved tokens for user {user_id} to {path}")

    def get_tokens(self, user_id: int) -> Optional[Dict[str, Any]]:
        path = self._get_token_path(user_id)
        if os.path.exists(path):
            with open(path, 'r') as f:
                tokens = json.load(f)
            self.logger.info(f"Retrieved tokens for user {user_id} from {path}")
            return tokens
        return None

    def delete_tokens(self, user_id: int) -> None:
        path = self._get_token_path(user_id)
        if os.path.exists(path):
            os.remove(path)
            self.logger.info(f"Deleted tokens for user {user_id} from {path}")

    def save_user_state(self, user_id: int, state: Dict[str, Any]) -> None:
        path = self._get_state_path(user_id)
        with open(path, 'w') as f:
            json.dump(state, f)
        self.logger.info(f"Saved user state for user {user_id} to {path}")

    def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        path = self._get_state_path(user_id)
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
            self.logger.info(f"Retrieved user state for user {user_id} from {path}")
            return state
        return None

    def delete_user_state(self, user_id: int) -> None:
        path = self._get_state_path(user_id)
        if os.path.exists(path):
            os.remove(path)
            self.logger.info(f"Deleted user state for user {user_id} from {path}")