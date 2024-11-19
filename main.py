import os
import discord
from discord import app_commands
from discord.ext import commands
import google.generativeai as genai
from datetime import datetime
import csv
import logging
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Global Configuration
XP_PER_MESSAGE = 1
LEVEL_THRESHOLD = 100
MESSAGE_DELAY = 1  # Delay in seconds between split messages

# Set up logging
def setup_logging():
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'bot_errors.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('discord_bot')

logger = setup_logging()

# Database setup
Base = declarative_base()

class MessageLog(Base):
    __tablename__ = 'message_logs'
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    username = Column(String)
    channel_id = Column(String)
    channel_name = Column(String)
    message_count = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(Integer, default=1)
    xp = Column(Float, default=0)

# Bot Configuration
class BotConfig:
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    DISCORD_CLIENT_ID = os.getenv('DISCORD_CLIENT_ID')
    GUILD_ID = int(os.getenv('GUILD_ID'))
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize Gemini
genai.configure(api_key=BotConfig.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def split_response(text, limit=1900):
    """Split a long response into chunks that fit within Discord's message limit."""
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Check if the text contains code blocks
    code_block_start = "```"
    is_code_block = text.strip().startswith(code_block_start)
    
    # Split by newlines first to maintain formatting
    lines = text.split('\n')
    
    for line in lines:
        # If a single line is longer than the limit, split it by spaces
        if len(line) > limit:
            words = line.split(' ')
            for word in words:
                if current_length + len(word) + 1 > limit:
                    chunk_text = '\n'.join(current_chunk)
                    if is_code_block:
                        # Close code block if it's open
                        if chunk_text.count(code_block_start) % 2 != 0:
                            chunk_text += "\n```"
                    chunks.append(chunk_text)
                    current_chunk = [word]
                    current_length = len(word)
                    if is_code_block:
                        # Start new code block
                        current_chunk.insert(0, "```")
                        current_length += 3
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
        # If adding the line would exceed limit, start a new chunk
        elif current_length + len(line) + 1 > limit:
            chunk_text = '\n'.join(current_chunk)
            if is_code_block:
                # Close code block if it's open
                if chunk_text.count(code_block_start) % 2 != 0:
                    chunk_text += "\n```"
            chunks.append(chunk_text)
            current_chunk = [line]
            current_length = len(line)
            if is_code_block:
                # Start new code block
                current_chunk.insert(0, "```")
                current_length += 3
        # Add line to current chunk
        else:
            current_chunk.append(line)
            current_length += len(line) + 1
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        if is_code_block:
            # Close code block if it's open
            if chunk_text.count(code_block_start) % 2 != 0:
                chunk_text += "\n```"
        chunks.append(chunk_text)
    
    return chunks

class DiscordBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix='/', intents=intents)
        self.guild = None
        self.setup_database()
        self.setup_csv_logging()

    def setup_database(self):
        self.engine = create_engine('sqlite:///discord_stats.db')
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def setup_csv_logging(self):
        self.logs_dir = 'logs'
        os.makedirs(self.logs_dir, exist_ok=True)
        self.init_csv_log()

    def init_csv_log(self):
        log_file = self.get_current_log_file()
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'user_id',
                    'username',
                    'channel_id',
                    'channel_name',
                    'message_content',
                    'message_type',
                    'level',
                    'xp'
                ])

    def get_current_log_file(self):
        current_date = datetime.now().strftime('%Y-%m')
        return os.path.join(self.logs_dir, f'chat_logs_{current_date}.csv')

    async def setup_hook(self):
        try:
            self.guild = await self.fetch_guild(BotConfig.GUILD_ID)
            logger.info(f"Bot initialized for guild: {self.guild.name}")
            await self.tree.sync(guild=discord.Object(id=BotConfig.GUILD_ID))
            logger.info("Slash commands synced with Discord")
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")

bot = DiscordBot()

@bot.tree.command(
    name="api",
    description="Ask a question to the Gemini AI",
    guild=discord.Object(id=BotConfig.GUILD_ID)
)
async def api_command(interaction: discord.Interaction, prompt: str):
    try:
        await interaction.response.defer()
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Split the response into chunks if it's too long
        chunks = split_response(response_text)
        total_chunks = len(chunks)
        
        # Send the first chunk as the initial response
        first_chunk = chunks[0]
        if total_chunks > 1:
            first_chunk = f"(1/{total_chunks})\n{first_chunk}"
        await interaction.followup.send(first_chunk)
        
        # Send any remaining chunks as follow-up messages
        for i, chunk in enumerate(chunks[1:], 2):
            # Add chunk number if there are multiple chunks
            numbered_chunk = f"({i}/{total_chunks})\n{chunk}"
            await interaction.channel.send(numbered_chunk)
            # Add delay between messages to prevent rate limiting
            await asyncio.sleep(MESSAGE_DELAY)
            
        logger.info(f"API command used by {interaction.user.name} - Response split into {len(chunks)} messages")
    except Exception as e:
        error_msg = f"Error using Gemini API: {str(e)}"
        logger.error(error_msg)
        await interaction.followup.send(error_msg)

@bot.tree.command(
    name="leaderboard",
    description="Show the server's activity leaderboard",
    guild=discord.Object(id=BotConfig.GUILD_ID)
)
async def leaderboard_command(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        session = bot.Session()
        
        top_users = session.query(MessageLog)\
            .order_by(MessageLog.message_count.desc())\
            .limit(10)\
            .all()

        user_stats = session.query(MessageLog)\
            .filter_by(user_id=str(interaction.user.id))\
            .first()

        leaderboard_text = "📊 **Chat Activity Leaderboard**\n\n"
        
        if not top_users:
            leaderboard_text += "No activity recorded yet!\n"
        else:
            for i, user in enumerate(top_users, 1):
                leaderboard_text += f"{i}. {user.username}: {user.message_count} messages (Level {user.level})\n"
        
        if user_stats:
            user_rank = session.query(MessageLog)\
                .filter(MessageLog.message_count > user_stats.message_count)\
                .count() + 1
            leaderboard_text += f"\nYour Rank: #{user_rank}"
        else:
            leaderboard_text += "\nYou haven't sent any messages yet!"

        await interaction.followup.send(leaderboard_text)
        logger.info(f"Leaderboard displayed for {interaction.user.name}")
        
    except Exception as e:
        error_msg = f"Error displaying leaderboard: {str(e)}"
        logger.error(error_msg)
        await interaction.followup.send(error_msg)
    finally:
        session.close()

@bot.event
async def on_message(message):
    try:
        if message.guild.id != BotConfig.GUILD_ID or message.author == bot.user:
            return

        session = bot.Session()
        try:
            user_log = session.query(MessageLog).filter_by(
                user_id=str(message.author.id)
            ).first()

            if not user_log:
                user_log = MessageLog(
                    user_id=str(message.author.id),
                    username=message.author.name,
                    channel_id=str(message.channel.id),
                    channel_name=message.channel.name,
                    message_count=0,
                    xp=0
                )
                session.add(user_log)
                session.flush()

            user_log.message_count += 1
            user_log.xp += XP_PER_MESSAGE
            user_log.channel_id = str(message.channel.id)
            user_log.channel_name = message.channel.name

            new_level = int(user_log.xp / LEVEL_THRESHOLD) + 1
            if new_level > user_log.level:
                user_log.level = new_level
                await message.channel.send(
                    f'🎉 Congratulations {message.author.mention}! You reached level {new_level}!'
                )

            session.commit()

            with open(bot.get_current_log_file(), 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    str(message.author.id),
                    message.author.name,
                    str(message.channel.id),
                    message.channel.name,
                    message.content[:100] + '...' if len(message.content) > 100 else message.content,
                    'chat',
                    user_log.level,
                    user_log.xp
                ])

        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            session.rollback()
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")

@bot.tree.command(
    name="stats",
    description="Show your activity statistics",
    guild=discord.Object(id=BotConfig.GUILD_ID)
)
async def stats_command(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        session = bot.Session()
        
        user_stats = session.query(MessageLog)\
            .filter_by(user_id=str(interaction.user.id))\
            .first()

        if user_stats:
            stats_text = f"**Your Stats**\n"
            stats_text += f"Level: {user_stats.level}\n"
            stats_text += f"XP: {user_stats.xp:.1f}\n"
            stats_text += f"Messages: {user_stats.message_count}\n"
            stats_text += f"Next Level: {(user_stats.level * LEVEL_THRESHOLD) - user_stats.xp:.1f} XP needed"
            
            await interaction.followup.send(stats_text)
        else:
            await interaction.followup.send("No stats available yet. Start chatting to gain XP!")
            
    except Exception as e:
        error_msg = f"Error displaying stats: {str(e)}"
        logger.error(error_msg)
        await interaction.followup.send(error_msg)
    finally:
        session.close()

if __name__ == "__main__":
    try:
        logger.info("Starting Discord bot...")
        bot.run(BotConfig.DISCORD_TOKEN)
    except Exception as e:
        logger.critical(f"Critical error starting bot: {str(e)}")