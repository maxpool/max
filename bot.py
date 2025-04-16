# This example requires the 'message_content' intent.

import discord
import os
import asyncio
import argparse
from dotenv import load_dotenv
from bot_handler import BotHandler
from logger import setup_logging, bot_logger
from prompts import get_welcome_prompt

# ID of the specific server where the bot should respond
ALLOWED_GUILD_ID = 1345864769806274661

def parse_arguments():
    """Parse command line arguments for the bot"""
    parser = argparse.ArgumentParser(description="Max Discord Bot")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Save logs to specified file")
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Set up logging based on verbosity flag
setup_logging(verbose=args.verbose, log_file=args.log_file)

# Load environment variables
load_dotenv()

# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize Discord client
client = discord.Client(intents=intents)

# Initialize the bot handler
bot_handler = BotHandler()

# Track if a message is expecting clarification
awaiting_clarification = {}  # {user_id: {channel_id: bool}}

@client.event
async def on_ready():
    """Log when the bot has successfully connected to Discord."""
    bot_logger.info(f'Bot connected as {client.user}')

@client.event
async def on_message(message):
    """Process incoming Discord messages."""
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if message is from the allowed server (guild)
    if message.guild is None or message.guild.id != ALLOWED_GUILD_ID:
        bot_logger.debug(
            f"Ignoring message from server ID {message.guild.id if message.guild else 'DM'}"
        )
        return

    # Get user and channel IDs as strings
    user_id = str(message.author.id)
    channel_id = str(message.channel.id)

    bot_logger.debug(f"Received message from {message.author.display_name} ({user_id}): '{message.content[:50]}...' in channel {channel_id}")

    # Special handling for intro-yourself
    if message.channel.name == "intro-yourself":
        bot_logger.info(
            f"Processing welcome message for new user {message.author.display_name} in intro-yourself channel"
        )
        await process_welcome_message(message, "Welcome {username} to Maxpool! 🎉")
        return

    # Initialize user in awaiting_clarification dict if not exists
    if user_id not in awaiting_clarification:
        awaiting_clarification[user_id] = {}

    # Track if this is a clarification to a previous query
    is_clarification = awaiting_clarification.get(user_id, {}).get(channel_id, False)

    # Reset clarification flag after processing
    if is_clarification:
        bot_logger.debug(f"Processing clarification from user {user_id}")
        awaiting_clarification[user_id][channel_id] = False

    # Check if the bot is mentioned or message is a reply to the bot
    should_respond = False
    reply_to_bot = False
    referenced_content = None
    referenced_user_id = None
    in_thread = isinstance(message.channel, discord.Thread)
    existing_thread = None
    thread_history = []
    reply_chain = []

    # Collect thread history if we're in a thread
    if in_thread:
        bot_logger.debug(f"Message is in thread: {message.channel.name}")
        try:
            # Get recent messages in the thread for context (limit to 10)
            async for msg in message.channel.history(limit=10):
                if msg.id != message.id:  # Skip the current message
                    thread_history.append({
                        "author": "User" if msg.author != client.user else "Max",
                        "content": msg.content
                    })
            # Reverse to get chronological order
            thread_history.reverse()
            bot_logger.debug(f"Collected {len(thread_history)} messages from thread history")
        except Exception as e:
            bot_logger.error(f"Error collecting thread history: {e}")

    # Always check if this is a reply to any message
    if message.reference is not None and hasattr(message.reference, 'resolved') and message.reference.resolved:
        referenced_message = message.reference.resolved

        # Case 1: Reply to the bot's message
        if referenced_message.author == client.user:
            should_respond = True
            reply_to_bot = True
            content = message.content.strip()
            bot_logger.debug(f"Message is a reply to bot's message")

            # Try to collect reply chain context
            try:
                # Start with the current reply
                current_ref = message.reference
                # Trace back through the reply chain
                while current_ref and hasattr(current_ref, 'resolved') and current_ref.resolved:
                    msg = current_ref.resolved
                    reply_chain.append({
                        "author": "User" if msg.author != client.user else "Max",
                        "content": msg.content
                    })
                    # Move to the previous reference in the chain
                    current_ref = msg.reference
                # Reverse to get chronological order
                reply_chain.reverse()
                bot_logger.debug(f"Collected {len(reply_chain)} messages from reply chain")
            except Exception as e:
                bot_logger.error(f"Error collecting reply chain: {e}")

        # Case 2: Reply to any message + bot is mentioned (including user's own message)
        elif client.user in message.mentions:
            should_respond = True
            referenced_content = referenced_message.content.strip()
            referenced_user_id = str(referenced_message.author.id)
            content = message.content.replace(f'<@{client.user.id}>', '').strip()
            bot_logger.debug(f"Message is a reply to user {referenced_user_id}'s message with bot mention")

        # Case 3: Reply to own message + bot is mentioned - ensure we capture referenced content
        # This specifically helps with scenarios like "thoughts on this paper?"
        elif referenced_message.author.id == message.author.id and client.user in message.mentions:
            should_respond = True
            referenced_content = referenced_message.content.strip()
            referenced_user_id = str(referenced_message.author.id)
            content = message.content.replace(f'<@{client.user.id}>', '').strip()
            bot_logger.debug(f"Message is a reply to user's own message with bot mention")

    # Case 4: Direct mention of the bot (no reply)
    elif client.user in message.mentions:
        should_respond = True
        content = message.content.replace(f'<@{client.user.id}>', '').strip()
        bot_logger.debug(f"Message contains direct bot mention")

        # For vague coreferences without a direct reply, try to build context
        # Use bot_handler to check if this message likely contains a coreference
        if await bot_handler._is_reference_request(content):
            try:
                # Get a few recent messages for context
                recent_messages = []
                async for msg in message.channel.history(limit=5):
                    if msg.id != message.id and msg.author.id == message.author.id:
                        recent_messages.append(msg)

                # If we found recent messages from the same user
                if recent_messages:
                    # Use the most recent message as context
                    referenced_content = recent_messages[0].content.strip()
                    referenced_user_id = str(recent_messages[0].author.id)
                    bot_logger.debug(f"Found reference context from previous message: '{referenced_content[:50]}...'")
            except Exception as e:
                bot_logger.error(f"Error building context from history: {e}")

    # If the message should be processed
    if should_respond:
        bot_logger.info(f"Processing message from user {user_id}: '{content[:50]}...'")

        # Generate thread name from content for new threads
        thread_name = f"{message.author.display_name}'s question"
        if content and len(content) > 5:
            # Create a thread name from the first few words of the question
            thread_name = content[:50] + "..." if len(content) > 50 else content

        # Show typing indicator while processing
        async with message.channel.typing():
            try:
                # Use thread_history if available, otherwise use reply_chain for context
                context_messages = thread_history if thread_history else reply_chain

                # Process message through bot handler
                response = await bot_handler.process_message(
                    user_id=user_id,
                    channel_id=channel_id,
                    message_content=content,
                    is_clarification=is_clarification,
                    is_reply=reply_to_bot,
                    referenced_message=referenced_content,
                    referenced_user_id=referenced_user_id,
                    context_messages=context_messages
                )

                # Check if response is asking for clarification
                if ("Could you provide a bit more detail" in response or 
                    "help me give you a more accurate and helpful response" in response):
                    awaiting_clarification[user_id][channel_id] = True
                    bot_logger.debug(f"Bot is requesting clarification from user {user_id}")

                # Determine how to respond based on context
                if in_thread:
                    # We're already in a thread, just reply in it
                    bot_logger.debug(f"Responding in existing thread")
                    await send_chunked_response(message, response)
                else:
                    # Create a new thread for the conversation
                    bot_logger.debug(f"Creating new thread '{thread_name}' for conversation")
                    thread = await message.create_thread(
                        name=thread_name,
                        auto_archive_duration=1440  # Auto-archive after 24 hours
                    )

                    # Add the initial response to the thread
                    await send_chunked_response(message, response, thread=thread)

            except Exception as e:
                bot_logger.error(f"Error in message processing: {e}", exc_info=True)
                await message.reply("Sorry, I encountered an error while processing your message. Please try again.")

async def send_chunked_response(message, response, thread=None):
    """Send a response in chunks if it's too long."""
    target = thread if thread else message.channel
    
    if len(response) > 2000:
        # Split into chunks of max 2000 characters
        chunks = [response[i:i+1990] for i in range(0, len(response), 1990)]
        
        bot_logger.debug(f"Splitting response into {len(chunks)} chunks")
        
        # Send the first chunk as a reply or in thread
        if thread:
            await thread.send(chunks[0] + "...")
        else:
            await message.reply(chunks[0] + "...")
        
        # Send remaining chunks
        for i, chunk in enumerate(chunks[1:], 1):
            prefix = "..." if i < len(chunks) - 1 else ""
            suffix = "..." if i < len(chunks) - 1 else ""
            await target.send(prefix + chunk + suffix)
    else:
        # Send as a single message
        if thread:
            await thread.send(response)
        else:
            await message.reply(response)

async def process_welcome_message(message, thread_name_template):
    """Process messages in welcome-enabled channels and generate welcome responses."""
    try:
        # Check if message is from the allowed server (guild)
        if message.guild is None or message.guild.id != ALLOWED_GUILD_ID:
            bot_logger.debug(
                f"Ignoring welcome message from server ID {message.guild.id if message.guild else 'DM'}"
            )
            return

        # Get the user's introduction message
        introduction = message.content

        # Get the classifier LLM client for generating the welcome
        classifier_llm = bot_handler.llm_handler.get_llm(bot_handler.query_router.classifier_model)

        # Create welcome prompt
        welcome_prompt = get_welcome_prompt()

        # Build chain components with additional metadata
        chain_input = {
            "query": introduction,
            "username": message.author.display_name,
            "channel": message.channel.name
        }

        # Show typing indicator while processing
        async with message.channel.typing():
            # Invoke the model and get response
            chain = welcome_prompt | classifier_llm
            response = await asyncio.to_thread(chain.invoke, chain_input)

            # Create a friendly thread name for the welcome
            thread_name = thread_name_template.format(username=message.author.display_name)

            # Create a thread for the welcome message
            bot_logger.debug(f"Creating welcome thread: '{thread_name}'")
            thread = await message.create_thread(
                name=thread_name,
                auto_archive_duration=1440  # Auto-archive after 24 hours
            )

            # Send the welcome message in the thread
            await thread.send(response.content)

    except Exception as e:
        bot_logger.error(f"Error processing welcome message: {e}", exc_info=True)
        # Don't send error message to avoid disrupting the introduction flow

if __name__ == "__main__":
    bot_logger.info("Starting Max Discord Bot")
    client.run(os.getenv('DISCORD_MAX_TOKEN'))
