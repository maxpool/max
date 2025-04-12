"""
Chat history module for Max Discord Bot.

This module manages conversation history for users across different channels.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

class ChatHistory:
    """
    Manages chat history for multiple users across Discord channels.
    Provides methods to add, retrieve, and manage conversation history.
    """
    
    def __init__(self, max_history_length: int = 10):
        """
        Initialize the chat history manager.
        
        Args:
            max_history_length: Maximum number of message pairs to store per user
        """
        self.max_history_length = max_history_length
        # Structure: {(user_id, channel_id): [(human_message, ai_message), ...]}
        self.histories = defaultdict(list)
        
    def add_exchange(self, 
                    user_id: str,
                    channel_id: str, 
                    human_message: str, 
                    ai_message: str) -> None:
        """
        Add a human-AI message exchange to history.
        
        Args:
            user_id: Discord user ID
            channel_id: Discord channel ID
            human_message: User's message text
            ai_message: Bot's response text
        """
        key = (user_id, channel_id)
        self.histories[key].append((human_message, ai_message))
        
        # Maintain max history length
        if len(self.histories[key]) > self.max_history_length:
            self.histories[key].pop(0)
    
    def get_history(self, 
                   user_id: str, 
                   channel_id: str) -> List[Tuple[str, str]]:
        """
        Get plain text history for a specific user and channel.
        
        Args:
            user_id: Discord user ID
            channel_id: Discord channel ID
            
        Returns:
            List of (human_message, ai_message) pairs
        """
        return self.histories.get((user_id, channel_id), [])
    
    def get_langchain_messages(self, 
                              user_id: str, 
                              channel_id: str) -> List:
        """
        Get history formatted as LangChain message objects.
        
        Args:
            user_id: Discord user ID
            channel_id: Discord channel ID
            
        Returns:
            List of alternating HumanMessage and AIMessage objects
        """
        history = self.get_history(user_id, channel_id)
        messages = []
        
        for human_msg, ai_msg in history:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
            
        return messages
    
    def clear_history(self, user_id: str, channel_id: Optional[str] = None) -> None:
        """
        Clear chat history for a user, optionally in a specific channel.
        
        Args:
            user_id: Discord user ID
            channel_id: Optional Discord channel ID
        """
        if channel_id:
            # Clear for specific channel
            self.histories.pop((user_id, channel_id), None)
        else:
            # Clear all channels for user
            keys_to_delete = [k for k in self.histories.keys() if k[0] == user_id]
            for key in keys_to_delete:
                self.histories.pop(key, None) 