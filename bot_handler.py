"""
Bot handler for Max Discord Bot.

This module integrates LLM handling, query routing, and chat history management
to provide a complete Discord bot interaction flow.
"""

import asyncio
import random

from llm_handler import LLMHandler
from query_router import QueryRouter
from chat_history import ChatHistory
from query_rewriter import QueryRewriter
from prompts import (
    get_gemini_prompt,
    get_perplexity_prompt,
    get_clarification_message,
    get_reference_prompt,
)
from logger import handler_logger

class BotHandler:
    """
    Main handler for Max Discord Bot interactions.
    Integrates various components for routing and processing user queries.
    """

    def __init__(self):
        """Initialize the Bot Handler with necessary components."""
        handler_logger.info("Initializing BotHandler")
        self.query_router = QueryRouter()
        self.llm_handler = LLMHandler()
        # Set the classifier model property for use in get_client method
        self.llm_handler.classifier_model = self.query_router.classifier_model
        self.chat_history = ChatHistory()
        self.query_rewriter = QueryRewriter()

        # Default clarification questions for vague queries
        self.clarification_questions = {
            "general": "what specific information or help you're looking for?",
            "code": "what programming language or framework you're using?",
            "compare": "which specific aspects you'd like compared?",
            "error": "what error message you're receiving and what you were trying to do?",
            "help": "what specific task or concept you need help with?",
        }

        # Friendly greeting responses
        self.greeting_responses = [
            "Hey there! What's up in your AI adventures today? 🚀",
            "Hi! Ready to talk about some cool AI stuff? What's on your mind?",
            "Hello! Got any interesting tech questions brewing today? ✨",
            "What's up! Looking for AI insights or just saying hi? Either way, I'm here! 👋",
            "Yo! What's happening in your AI world today?",
            "Hey! How can I help with your AI journey today?",
            "Hi there! Ready to explore some AI topics?",
            "Hello! What AI curiosities are you pondering today? 💭",
            "Heya! What's on your AI radar today?",
            "Greetings! Ready to dive into some AI discussions? 🤖"
        ]
        handler_logger.debug("BotHandler initialized with all components")

    async def process_message(self, 
                             user_id: str, 
                             channel_id: str, 
                             message_content: str,
                             is_clarification: bool = False,
                             is_reply: bool = False,
                             referenced_message: str = None,
                             referenced_user_id: str = None,
                             context_messages: list = None) -> str:
        """
        Process incoming Discord message and generate a response.
        
        Args:
            user_id: Discord user ID
            channel_id: Discord channel ID
            message_content: The content of the user's message
            is_clarification: Whether this message is a clarification to a previous query
            is_reply: Whether this message is a reply to a bot message
            referenced_message: Content of the message being referenced/replied to
            referenced_user_id: User ID of the person who sent the referenced message
            context_messages: List of thread or reply chain messages for additional context
            
        Returns:
            Bot's response message
        """
        # Log received data for debugging
        handler_logger.debug(f"Processing message from user {user_id}: '{message_content[:50]}...'")
        handler_logger.debug(f"Referenced message: '{referenced_message[:50] if referenced_message else None}'")
        handler_logger.debug(f"Context messages count: {len(context_messages) if context_messages else 0}")

        # Handle message references (when someone asks the bot to answer someone else's question)
        is_reference_request = False

        # Get the classifier LLM client
        handler_logger.debug("Getting classifier LLM client")
        classifier_llm = self.llm_handler.get_llm(self.query_router.classifier_model)

        # Get chat history for context
        chat_history = self.chat_history.get_langchain_messages(user_id, channel_id)
        handler_logger.debug(f"Retrieved {len(chat_history) if chat_history else 0} messages from chat history")

        # Convert chat history to format needed for query rewriting
        recent_messages = []
        if chat_history:
            for msg in chat_history:
                author = "User" if hasattr(msg, "type") and msg.type == "human" else "Max"
                content = msg.content if hasattr(msg, "content") else ""
                recent_messages.append({"author": author, "content": content})

        # Add thread or reply chain context if available
        if context_messages:
            # Merge context with recent messages, prioritizing context_messages
            # since they're more likely to contain immediate relevant context
            combined_messages = context_messages + [msg for msg in recent_messages if not any(
                cm["content"] == msg["content"] for cm in context_messages if "content" in cm and "content" in msg
            )]
            recent_messages = combined_messages[:20]  # Limit to 20 messages to avoid token issues
            handler_logger.debug(f"Combined recent messages count: {len(recent_messages)}")

        # If we have a referenced message, we need to handle it appropriately
        if referenced_message:
            handler_logger.debug("Processing referenced message")
            # Check if message content indicates a reference request or is very short (likely just a mention)
            if await self._is_reference_request(message_content) or len(message_content.strip()) < 10:
                handler_logger.debug("Message appears to be a reference request")
                # Try to rewrite the query using the referenced message as context
                handler_logger.debug("Attempting to rewrite query with context")
                rewritten_query, was_rewritten = await self.query_rewriter.rewrite_query(
                    query=message_content,
                    recent_messages=recent_messages,
                    referenced_message=referenced_message,
                    llm_client=classifier_llm
                )

                # If query was successfully rewritten, use that
                if was_rewritten:
                    actual_query = rewritten_query
                    handler_logger.info(f"Query rewritten: '{actual_query[:50]}...'")
                else:
                    # Use the referenced message as the query instead
                    actual_query = referenced_message
                    # Add context about the reference to the prompt
                    message_content = f"Please answer this question: {referenced_message}"
                    is_reference_request = True
                    handler_logger.info(f"Using referenced message as query: '{referenced_message[:50]}...'")
            else:
                # User mentioned the bot in a reply but seems to be asking their own question
                actual_query = message_content
                handler_logger.debug("Using original message content as query")
        else:
            # No referenced message, but still try to rewrite for context if needed
            handler_logger.debug("No referenced message, checking if query needs context")
            rewritten_query, was_rewritten = await self.query_rewriter.rewrite_query(
                query=message_content,
                recent_messages=recent_messages,
                llm_client=classifier_llm
            )

            # If query was successfully rewritten, use that
            if was_rewritten:
                actual_query = rewritten_query
                handler_logger.info(f"Query rewritten with context: '{actual_query[:50]}...'")
            else:
                # No referenced message, just use the original content
                actual_query = message_content

        # Check if this is a simple greeting message
        if self._is_greeting(actual_query):
            handler_logger.debug("Message is a greeting, sending greeting response")
            return self._get_greeting_response()

        # Route query to appropriate LLM
        handler_logger.debug("Routing query to appropriate LLM")
        provider, model_name, params, is_ai_related = await self.query_router.route_query(actual_query, classifier_llm)
        handler_logger.info(f"Query routed to {provider}/{model_name} (AI-related: {is_ai_related})")

        # Get the appropriate LLM
        handler_logger.debug(f"Getting LLM: {provider}/{model_name}")
        llm = self.llm_handler.get_llm(model_name=model_name, provider=provider, **params)

        # Choose the right prompt template based on provider and request type
        if is_reference_request:
            # Use the special reference handling prompt
            prompt = get_reference_prompt(provider, chat_history if chat_history else None)
            handler_logger.debug("Using reference prompt")
        elif provider == "perplexity":
            prompt = get_perplexity_prompt(chat_history if chat_history else None)
            handler_logger.debug("Using Perplexity prompt")
        else:
            prompt = get_gemini_prompt(chat_history if chat_history else None)
            handler_logger.debug("Using Gemini prompt")

        # Build chain components
        if chat_history:
            chain_input = {"query": actual_query, "chat_history": chat_history}
        else:
            chain_input = {"query": actual_query}

        try:
            # Invoke the model and get response
            handler_logger.debug("Invoking LLM")
            chain = prompt | llm
            response = await asyncio.to_thread(chain.invoke, chain_input)

            # Process the response content
            response_text = response.content
            handler_logger.debug(f"Got response from LLM ({len(response_text)} chars)")

            # Add citations for Perplexity responses
            if provider == "perplexity" and hasattr(response, "additional_kwargs") and "citations" in response.additional_kwargs:
                citations = response.additional_kwargs.get("citations", [])
                if citations:
                    handler_logger.debug(f"Adding {len(citations)} citations to response")
                    response_text += "\n\n**Sources:**\n"
                    for i, citation in enumerate(citations, 1):
                        response_text += f"{i}. {citation}\n"

            # Save the history using the original message if it's a reference request
            # This helps maintain more natural context in the chat history
            history_message = actual_query

            # Add the exchange to chat history
            handler_logger.debug("Adding exchange to chat history")
            self.chat_history.add_exchange(
                user_id=user_id,
                channel_id=channel_id,
                human_message=history_message,
                ai_message=response_text
            )

            return response_text

        except Exception as e:
            # Handle errors gracefully
            handler_logger.error(f"Error processing message: {e}", exc_info=True)
            return f"I encountered an issue while processing your question. Could you try rephrasing it? (Error: {str(e)[:100]}...)"

    def _is_greeting(self, message: str) -> bool:
        """
        Check if the message is a simple greeting.
        
        Args:
            message: The user's message
            
        Returns:
            Boolean indicating if the message is a greeting
        """
        message_lower = message.lower().strip()

        # If the message is long (more than 5 words), it's not just a greeting
        # This prevents questions that start with "hi" or "hello" from being treated as greetings
        if len(message_lower.split()) > 2:
            return False

        # Common greetings to check for
        common_greetings = ["hey", "hello", "hi", "sup", "yo", "greetings", "hiya", "howdy", 
                            "good morning", "good afternoon", "good evening",
                            "morning", "afternoon", "evening", "what's up", "whats up",
                            "what up", "hey there", "hello there", "hi there",
                            "heya", "heyy", "hiii", "hiiii", "heyyy", "hellooo",
                            "wassup", "what is up", "what's happening", "whats happening"]

        # Greetings specifically for Max
        max_greetings = ["hey max", "hello max", "hi max", "yo max", "sup max", "howdy max",
                         "hey max!", "hello max!", "hi max!", "what's up max", "whats up max"]

        # Check for exact matches
        if message_lower in common_greetings or message_lower in max_greetings:
            return True

        # For longer messages, only consider it a greeting if it's very simple
        # Don't check for startswith, as this catches legitimate questions that begin with greetings
        if len(message_lower.split()) <= 3:
            for greeting in ["hey", "hello", "hi", "sup", "yo", "heya"]:
                if greeting in message_lower:
                    return True

            # Check for mentions of 'max' in short greetings
            if "max" in message_lower and any(greeting in message_lower for greeting in ["hey", "hello", "hi", "sup", "yo"]):
                return True

        return False

    async def _is_reference_request(self, message: str) -> bool:
        """
        Check if the message is asking the bot to address a referenced message.
        
        Args:
            message: The user's message
            
        Returns:
            Boolean indicating if the message is a reference request
        """
        message_lower = message.lower().strip()

        # If the message is very short (just mentioning the bot)
        if len(message_lower) < 5:
            handler_logger.debug("Message too short, treating as reference request")
            return True

        # Get the classifier LLM client
        classifier_llm = self.llm_handler.get_llm(self.query_router.classifier_model)

        # Use the router's coreference detection instead of fixed patterns
        try:
            handler_logger.debug("Checking for coreference in message")
            result = await self.query_router.detect_coreference(message, classifier_llm)
            handler_logger.debug(f"Coreference detection result: {result}")
            return result
        except Exception as e:
            handler_logger.error(f"Error in coreference detection: {e}", exc_info=True)
            # Fallback to a basic check if LLM-based detection fails
            basic_check = "this" in message_lower and len(message_lower.split()) < 10
            handler_logger.debug(f"Falling back to basic coreference check: {basic_check}")
            return basic_check

    def _get_greeting_response(self) -> str:
        """
        Get a random friendly greeting response.
        
        Returns:
            A greeting response
        """
        response = random.choice(self.greeting_responses)
        handler_logger.debug(f"Selected greeting response: '{response}'")
        return response

    def _get_clarification_question(self, message: str) -> str:
        """
        Select an appropriate clarification question based on message content.
        
        Args:
            message: The user's message
            
        Returns:
            Specific clarification question to ask
        """
        # Get the classifier LLM client
        classifier_llm = self.llm_handler.get_llm(self.query_router.classifier_model)

        # Define the classification prompt
        classification_prompt = f"""
        Classify the following message into one of these categories:
        - code: Questions about programming, development, or technical issues
        - compare: Requests to compare technologies, frameworks, or concepts
        - error: Questions about errors, bugs, or technical problems
        - help: General requests for assistance or support
        - general: Other types of questions or requests
        
        Message: {message}
        
        Category:"""

        try:
            # Get classification from LLM
            handler_logger.debug("Classifying message for clarification")
            classification_result = classifier_llm.invoke(classification_prompt).content.strip().lower()

            # Extract the category from the response
            for category in self.clarification_questions.keys():
                if category in classification_result:
                    handler_logger.debug(f"Message classified as '{category}' for clarification")
                    return self.clarification_questions[category]

            # Default to general if no match found
            handler_logger.debug("No specific category found, using general clarification")
            return self.clarification_questions["general"]

        except Exception as e:
            # Handle errors gracefully and default to general question
            handler_logger.error(f"Error classifying message for clarification: {e}")
            return self.clarification_questions["general"] 
