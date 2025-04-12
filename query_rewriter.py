"""
Query Rewriter module for Max Discord Bot.

This module rewrites ambiguous or context-dependent queries by incorporating
relevant context from past messages in the conversation.
"""

from typing import List, Dict, Optional, Any, Tuple
from logger import rewriter_logger

class QueryRewriter:
    """
    Rewrites ambiguous or context-dependent queries using Claude 3.7 to incorporate
    relevant context from previous messages.
    """
    
    def __init__(self, model="claude-3-7-sonnet-20250219"):
        """
        Initialize the QueryRewriter with the specified model.
        
        Args:
            model: The Claude model to use for query rewriting
        """
        self.model = model
        rewriter_logger.debug(f"QueryRewriter initialized with model: {model}")
        
    async def rewrite_query(self, 
                           query: str, 
                           recent_messages: List[Dict[str, Any]], 
                           referenced_message: Optional[str] = None,
                           llm_client: Any = None) -> Tuple[str, bool]:
        """
        Rewrites a query by incorporating context from previous messages.
        
        Args:
            query: The current user query that might be ambiguous
            recent_messages: List of recent messages in the conversation
                             (can include thread history or reply chain messages)
            referenced_message: A specific message being replied to (if any)
            llm_client: The LLM client to use for rewriting
            
        Returns:
            Tuple containing:
                - The rewritten query (or original if no rewrite needed)
                - Boolean indicating if the query was rewritten
        """
        rewriter_logger.debug(f"Attempting to rewrite query: '{query}'")
        rewriter_logger.debug(f"With referenced message: '{referenced_message[:50] if referenced_message else None}'")
        rewriter_logger.debug(f"Context messages count: {len(recent_messages) if recent_messages else 0}")
        
        # If query is already detailed, return as is
        if len(query.split()) > 15:
            rewriter_logger.debug("Query is already detailed (>15 words), skipping rewrite")
            return query, False
        
        # Prepare context from recent messages
        context_text = self._format_messages_for_context(recent_messages)
        
        # Determine conversation type for better rewriting
        conversation_type = "general conversation"
        if any(msg.get("content", "").startswith("```") for msg in recent_messages):
            conversation_type = "conversation about code"
            rewriter_logger.debug("Detected code-related conversation")
        
        # Check if we're in a thread based on volume of messages
        is_thread = len(recent_messages) >= 7
        thread_context = ""
        if is_thread:
            thread_context = "This conversation is happening in a Discord thread. "
            rewriter_logger.debug("Detected thread conversation")
        
        # Add referenced message as high-priority context if available
        reference_text = ""
        if referenced_message:
            reference_text = f"The user is specifically replying to this message: \"{referenced_message}\"\n\n"
            rewriter_logger.debug("Added referenced message to context")
            
        prompt = f"""
        You are a query rewriting system for an AI assistant. You need to rewrite potentially ambiguous or context-dependent queries 
        into standalone, self-contained queries that clearly express the user's intent by incorporating relevant context.

        Current user query: "{query}"

        {thread_context}This appears to be a {conversation_type}.
        {reference_text}Recent conversation context:
        {context_text}

        Task: Rewrite the user's query to be self-contained by incorporating the necessary context.
        
        Guidelines:
        1. If the query is already clear and specific, return it unchanged
        2. If the query contains pronouns (it, this, that) or references something mentioned earlier, incorporate that context
        3. If the query is very brief like "what are some latest techniques", add context about the domain/topic from the conversation
        4. Make the rewritten query natural, as if the user had written it themselves
        5. Keep the rewritten query concise while being specific
        6. DO NOT add information that wasn't implied or directly stated in the context
        7. DO NOT change the user's original intent
        
        IMPORTANT: Your response MUST be a valid JSON object in exactly this format with no extra text before or after:
        {{
            "rewritten_query": "The rewritten query with necessary context incorporated",
            "was_rewritten": true/false
        }}
        
        ONLY respond with this JSON object and NOTHING ELSE.
        """
        
        try:
            rewriter_logger.debug("Invoking LLM for query rewriting")
            response = llm_client.invoke(prompt)
            response_text = response.content
            
            try:
                import json
                result = json.loads(response_text)
                rewritten_query = result.get("rewritten_query", query)
                was_rewritten = result.get("was_rewritten", False)
                
                if was_rewritten:
                    rewriter_logger.info(f"Query rewritten: '{rewritten_query[:100]}...'")
                else:
                    rewriter_logger.debug("Query not rewritten")
                
                return rewritten_query, was_rewritten
            except json.JSONDecodeError:
                # Fallback if model doesn't return valid JSON
                rewriter_logger.warning("Failed to parse JSON response from LLM, returning original query")
                return query, False
                
        except Exception as e:
            rewriter_logger.error(f"Error rewriting query: {e}", exc_info=True)
            return query, False
            
    def _format_messages_for_context(self, messages: List[Dict[str, Any]]) -> str:
        """
        Formats recent messages into a string for context.
        
        Args:
            messages: List of message dictionaries containing content and author
            
        Returns:
            Formatted string containing relevant conversation context
        """
        if not messages:
            return ""
            
        context_lines = []
        for msg in messages:
            author = msg.get("author", "User")
            content = msg.get("content", "")
            if content:
                context_lines.append(f"{author}: {content}")
                
        # Include more messages when they're available from thread or reply context
        # This assumes thread/reply context is higher quality than general history
        max_messages = 10 if len(messages) > 7 else 5
        rewriter_logger.debug(f"Using {max_messages} messages for context")
        return "\n".join(context_lines[-max_messages:])  # Include more messages when available 