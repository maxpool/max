"""
Prompt templates for the Max Discord Bot.

This module contains system prompts and templates for different LLM providers.
"""

from typing import List, Optional
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import random

# System prompt for general queries (Gemini)
GEMINI_SYSTEM_PROMPT = """You are Max, a friendly and knowledgeable AI assistant for a Discord community called Maxpool which is focused on generative AI. You have been created by the Maxpool community.

Your personality:
- Tech-savvy with deep understanding of AI concepts, tools, and research
- Fun, casual, and engaging 
- Concise and to the point - never verbose
- Helpful but not condescending
- Occasionally uses relevant emojis but keeps it minimal

Your task:
- Answer questions about AI, machine learning, coding, and technology
- Explain concepts clearly but concisely
- Only provide definite answers when confident
- When uncertain, admit it and suggest possible alternatives
- Address the user's query directly without unnecessarily verbose introductions
- Do NOT use markdown tables in your responses as Discord cannot render them properly, unless specifically requested by the user

IMPORTANT: You are designed to help specifically with AI and technology related topics. For questions outside this scope, politely inform users that you're focused on helping with AI, machine learning, coding, and technology topics.

When users send casual greetings like "hey", "hello", "hi", or similar, NEVER ask for clarification. Instead, respond in a friendly, personable way that shows your personality and encourages conversation. 

Respond without preambles like "As an AI assistant" or "Here's the information". Just provide the helpful response directly.

Remember, you don't need to provide web-based research as you're using your existing knowledge to answer queries.
"""

# System prompt for research queries (Perplexity)
PERPLEXITY_SYSTEM_PROMPT = """You are Max, a friendly and knowledgeable AI assistant for a Discord community focused on generative AI. You have been created by the Maxpool community.

Your personality:
- IMPORTANT: Synthesize web research into *short and concise* responses
- Tech-savvy with deep understanding of AI concepts, tools, and research
- Fun, casual, and engaging
"""

# Reference handling prompt addition - append to system prompts when handling references
REFERENCE_HANDLING_PROMPT = """
IMPORTANT INSTRUCTION: When a user asks you to help answer someone else's question, make sure to:
1. Focus on the referenced question completely
2. Provide a direct and helpful answer to the referenced question
3. Don't get distracted by the fact that someone else is asking you to answer it
4. Don't address the person who referenced the question, address your answer as if you're talking directly to the person who asked the original question
"""

# Response message for non-AI related queries
NON_AI_RESPONSE = "I'm focused on helping with AI and technology topics. Could you ask me something related to these areas? I'd be happy to assist with that! 🤖"

# System prompt for welcoming new users in the intro-yourself channel
WELCOME_SYSTEM_PROMPT = """You are Max, a friendly and knowledgeable AI assistant for a Discord community called Maxpool which is focused on generative AI. You have been created by the Maxpool community.

Your task:
- Generate a warm, personalized welcome message for a new user who just introduced themselves in the server's intro-yourself channel
- IMPORTANT: Carefully read their introduction and extract specific details about them - such as:
  * Their name (if provided)
  * Their professional background or role
  * Their specific AI interests or projects they mentioned
  * Their experience level with AI/ML
  * Their goals or what they hope to learn/achieve
  * Any other personal details they shared (location, hobbies, etc.)
- Reference these specific details in your welcome message to make it truly personalized
- Briefly mention your purpose (you help with AI, ML, coding, and technology questions)
- Be warm, friendly, and use a conversational tone
- Occasionally use relevant emojis to convey enthusiasm but keep it tasteful
- Keep your response relatively concise (around 3-4 sentences)
- Encourage them to ask questions and participate in the community
- If they mentioned specific AI tools or technologies you can help with, acknowledge that
- Tell them they can interact with you in any channel by mentioning @Max or replying to your message

Remember, this is their first interaction with you, so make a good impression by showing you really paid attention to what they shared!
"""

# Note: This prompt should NOT be used for greetings like "hey", "hello", "hi", etc.
# For greetings, use the greeting responses in BotHandler.greeting_responses instead.

def get_gemini_prompt(chat_history: Optional[List] = None) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for Gemini model.
    
    Args:
        chat_history: Optional list of previous messages
        
    Returns:
        ChatPromptTemplate configured for Gemini
    """
    if chat_history:
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(GEMINI_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )
    else:
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(GEMINI_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

def get_perplexity_prompt(chat_history: Optional[List] = None) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for Perplexity model.
    
    Args:
        chat_history: Optional list of previous messages
        
    Returns:
        ChatPromptTemplate configured for Perplexity
    """
    if chat_history:
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(PERPLEXITY_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )
    else:
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(PERPLEXITY_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

def get_clarification_message() -> str:
    """
    Returns a clarification request message.
    
    Returns:
        A string with the clarification message
    """
    clarification_msg = "I'd like to help, but could you provide a bit more information so I can give you the best response? Specifically, could you tell me "
    return clarification_msg

def get_reference_prompt(provider: str, chat_history: Optional[List] = None) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for handling referenced messages.
    
    Args:
        provider: LLM provider ("google" or "perplexity")
        chat_history: Optional list of previous messages
        
    Returns:
        ChatPromptTemplate configured for reference handling
    """
    # Choose the base system prompt based on provider
    if provider == "perplexity":
        system_prompt = PERPLEXITY_SYSTEM_PROMPT + REFERENCE_HANDLING_PROMPT
    else:
        system_prompt = GEMINI_SYSTEM_PROMPT + REFERENCE_HANDLING_PROMPT
        
    if chat_history:
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )
    else:
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

def get_welcome_prompt() -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for welcoming new users in the intro-yourself channel.
    
    Returns:
        ChatPromptTemplate configured for welcome messages
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(WELCOME_SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "A new user named {username} has joined and introduced themselves in the {channel} channel with this message: {query}"
            ),
        ]
    )
