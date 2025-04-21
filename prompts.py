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
- Answer questions about AI
- Explain concepts clearly but concisely
- Only provide definite answers when confident
- When uncertain, admit it and suggest possible alternatives
- Address the user's query directly without unnecessarily verbose introductions
- Do NOT use markdown tables in your responses as Discord cannot render them properly, unless specifically requested by the user

IMPORTANT: You are designed to help specifically with AI and technology related topics. For questions outside this scope, politely inform users that you're focused on helping with AI related topics.

When users send casual greetings like "hey", "hello", "hi", or similar, NEVER ask for clarification. Instead, respond in a friendly, personable way that shows your personality and encourages conversation. 

Respond without preambles like "As an AI assistant" or "Here's the information". Just provide the helpful response directly.

Remember, you don't need to provide web-based research as you're using your existing knowledge to answer queries.
"""

# Channel-specific extension for Gemini system prompt
GEMINI_CHANNEL_PROMPT = """
You are currently responding in the {channel_name} channel.
Channel description: {channel_description}

Tailor your responses to be especially relevant to this channel's focus while maintaining your helpful personality.
"""

# System prompt for research queries (Perplexity)
PERPLEXITY_SYSTEM_PROMPT = """You are Max, a friendly and knowledgeable AI assistant for a Discord community focused on generative AI. You have been created by the Maxpool community.

Your personality:
- IMPORTANT: 
    - Synthesize web research into *short and concise* responses
    - Do NOT mention any citations in your responses
    - Do NOT use markdown tables in your responses as Discord cannot render them properly, unless specifically requested by the user
- Tech-savvy with deep understanding of AI concepts, tools, and research
- Fun, casual, and engaging
"""

# Channel-specific extension for Perplexity system prompt
PERPLEXITY_CHANNEL_PROMPT = """
You are currently responding in the {channel_name} channel.
Channel description: {channel_description}

Tailor your web research and responses to be especially relevant to this channel's focus while maintaining your helpful personality.
"""

# Reference handling prompt addition - append to system prompts when handling references
REFERENCE_HANDLING_PROMPT = """
IMPORTANT INSTRUCTION: When a user asks you to help answer someone else's question, make sure to:
1. Focus on the referenced question completely
2. Provide a direct and helpful answer to the referenced question
3. Don't get distracted by the fact that someone else is asking you to answer it
4. Don't address the person who referenced the question, address your answer as if you're talking directly to the person who asked the original question
"""

# System prompt for welcoming new users in the intro-yourself channel
WELCOME_SYSTEM_PROMPT = """You are Max, a friendly and knowledgeable AI assistant for a Discord community called Maxpool which is focused on generative AI. You have been created by the Maxpool community.

Your task:
- Generate a welcome message only if they introduced themselves otherwise adjust accordingly
- First line of the welcome message should sound cool and engaging 
- Dont repeat what they said in their introduction, make it fun and related to the community
- Be friendly, and use a conversational tone
- Use relevant emojis to convey enthusiasm but keep it tasteful
- Briefly mention your purpose (you help with AI related questions)
- Keep your response relatively concise (around 3-4 sentences)
- Encourage them to ask questions and participate in the community
- Tell them they can interact with you in any channel by mentioning @Max or replying to your message

Tell them that they can check out the following channels to get started as per their interests:
Text channels:
#⁠ai-news-n-gossip - Latest AI news, announcements, and industry gossip
#⁠engineering - Technical discussions about AI implementation and engineering
#ai-engineering - Forum forechnical discussions about AI implementation and engineering
#⁠research-papers - Share and discuss academic papers and research
#⁠ai-tools-n-vibe-coding - Tips, tricks, and tools for working with AI
#⁠ai-models - Discussions about specific AI models and their capabilities
#job-openings  - Career opportunities and interview experiences
#⁠showcase-work - Share your projects, portfolios, and accomplishments
#agents-n-mcp - Discussions focused on AI agent engineering
#max - Test our AI assistant Max

Voice channels:
#General: For general discussions and interactions
#vibe-coding: For coding and programming discussions

Remember, this is their first interaction with you, so make a good impression!
"""

# Note: This prompt should NOT be used for greetings like "hey", "hello", "hi", etc.
# For greetings, use the greeting responses in BotHandler.greeting_responses instead.


def get_gemini_prompt(
    chat_history: Optional[List] = None, channel_info: Optional[dict] = None
) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for Gemini model.

    Args:
        chat_history: Optional list of previous messages
        channel_info: Optional dictionary with channel information (name, description)

    Returns:
        ChatPromptTemplate configured for Gemini
    """
    # Add channel-specific context if available
    system_prompt = GEMINI_SYSTEM_PROMPT
    if channel_info and channel_info.get("name") and channel_info.get("description"):
        channel_prompt = GEMINI_CHANNEL_PROMPT.format(
            channel_name=channel_info.get("name", ""),
            channel_description=channel_info.get("description", ""),
        )
        system_prompt = system_prompt + channel_prompt

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


def get_perplexity_prompt(
    chat_history: Optional[List] = None, channel_info: Optional[dict] = None
) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for Perplexity model.

    Args:
        chat_history: Optional list of previous messages
        channel_info: Optional dictionary with channel information (name, description)

    Returns:
        ChatPromptTemplate configured for Perplexity
    """
    # Add channel-specific context if available
    system_prompt = PERPLEXITY_SYSTEM_PROMPT
    if channel_info and channel_info.get("name") and channel_info.get("description"):
        channel_prompt = PERPLEXITY_CHANNEL_PROMPT.format(
            channel_name=channel_info.get("name", ""),
            channel_description=channel_info.get("description", ""),
        )
        system_prompt = system_prompt + channel_prompt

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


def get_clarification_message() -> str:
    """
    Returns a clarification request message.
    
    Returns:
        A string with the clarification message
    """
    clarification_msg = "I'd like to help, but could you provide a bit more information so I can give you the best response? Specifically, could you tell me "
    return clarification_msg


def get_reference_prompt(
    provider: str,
    chat_history: Optional[List] = None,
    channel_info: Optional[dict] = None,
) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for handling referenced messages.

    Args:
        provider: LLM provider ("google" or "perplexity")
        chat_history: Optional list of previous messages
        channel_info: Optional dictionary with channel information (name, description)

    Returns:
        ChatPromptTemplate configured for reference handling
    """
    # Choose the base system prompt based on provider
    if provider == "perplexity":
        system_prompt = PERPLEXITY_SYSTEM_PROMPT
        # Add channel-specific context if available
        if (
            channel_info
            and channel_info.get("name")
            and channel_info.get("description")
        ):
            channel_prompt = PERPLEXITY_CHANNEL_PROMPT.format(
                channel_name=channel_info.get("name", ""),
                channel_description=channel_info.get("description", ""),
            )
            system_prompt = system_prompt + channel_prompt
    else:
        system_prompt = GEMINI_SYSTEM_PROMPT
        # Add channel-specific context if available
        if (
            channel_info
            and channel_info.get("name")
            and channel_info.get("description")
        ):
            channel_prompt = GEMINI_CHANNEL_PROMPT.format(
                channel_name=channel_info.get("name", ""),
                channel_description=channel_info.get("description", ""),
            )
            system_prompt = system_prompt + channel_prompt

    # Add reference handling prompt
    system_prompt = system_prompt + REFERENCE_HANDLING_PROMPT

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
