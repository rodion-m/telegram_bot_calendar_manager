from typing import Dict, Any, Optional

from loguru import logger

from logic import IRepository, start_span_smart, Config


class FirestoreRepository(IRepository):
    """Repository implementation using Google Firestore."""

    def __init__(self, config: 'Config'):
        from google.cloud import firestore
        self.client = firestore.AsyncClient()
        self.collection = config.FIRESTORE_COLLECTION
        self.logger = logger

    from google.cloud import firestore
    async def get_user_document(self, user_id: int) -> firestore.AsyncDocumentReference:
        """Get a reference to the user's document."""
        with start_span_smart(op="firestore", description="Get User Document"):
            return self.client.collection(self.collection).document(str(user_id))

    async def save_tokens(self, user_id: int, tokens: Dict[str, Any]) -> None:
        """Save OAuth tokens to Firestore."""
        with start_span_smart(op="firestore", description="Save Tokens"):
            user_doc = await self.get_user_document(user_id)
            await user_doc.set({"tokens": tokens}, merge=True)
            self.logger.info(f"Saved tokens for user {user_id} to Firestore.")

    async def get_tokens(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve OAuth tokens from Firestore."""
        user_doc = await self.get_user_document(user_id)
        with start_span_smart(op="firestore", description="Get Tokens"):
            doc = await user_doc.get()
            if doc.exists:
                tokens = doc.to_dict().get("tokens")
                self.logger.info(f"Retrieved tokens for user {user_id} from Firestore.")
                return tokens
        return None

    async def delete_tokens(self, user_id: int) -> None:
        """Delete OAuth tokens from Firestore."""
        from google.cloud import firestore
        with start_span_smart(op="firestore", description="Delete Tokens"):
            user_doc = await self.get_user_document(user_id)
            await user_doc.update({"tokens": firestore.DELETE_FIELD})
            self.logger.info(f"Deleted tokens for user {user_id} from Firestore.")

    async def save_user_state(self, user_id: int, state: Dict[str, Any]) -> None:
        """Save user state to Firestore."""
        with start_span_smart(op="firestore", description="Save User State"):
            user_doc = await self.get_user_document(user_id)
            await user_doc.set({"state": state}, merge=True)
            self.logger.info(f"Saved user state for user {user_id} to Firestore.")

    async def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve user state from Firestore."""
        with start_span_smart(op="firestore", description="Get User State"):
            user_doc = await self.get_user_document(user_id)
            doc = await user_doc.get()
            if doc.exists:
                state = doc.to_dict().get("state")
                self.logger.info(f"Retrieved user state for user {user_id} from Firestore.")
                return state
            return None

    async def delete_user_state(self, user_id: int) -> None:
        """Delete user state from Firestore."""
        with start_span_smart(op="firestore", description="Delete User State"):
            user_doc = await self.get_user_document(user_id)
            from google.cloud import firestore
            await user_doc.update({"state": firestore.DELETE_FIELD})
            self.logger.info(f"Deleted user state for user {user_id} from Firestore.")