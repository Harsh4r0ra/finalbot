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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import plotly.express as px
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import uvicorn
from threading import Thread
from pydantic import BaseModel
from typing import List, Dict, Any
import json

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

# Create static directory
static_path = Path("static")
static_path.mkdir(exist_ok=True)

# Define response models
class UserStats(BaseModel):
    user_id: str
    username: str
    message_count: int
    level: int
    xp: float

class ActivityGraphData(BaseModel):
    data: List[Dict[str, Any]]
    layout: Dict[str, Any]

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

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML)
static_path = Path("static")
static_path.mkdir(exist_ok=True)

# Define API routes before mounting static files
@app.get("/api/chat-stats", response_model=List[UserStats])
async def get_chat_stats():
    try:
        session = bot.Session()
        stats = session.query(MessageLog).all()
        result = [
            UserStats(
                user_id=stat.user_id,
                username=stat.username,
                message_count=stat.message_count,
                level=stat.level,
                xp=stat.xp
            ) for stat in stats
        ]
        return result
    except Exception as e:
        logger.error(f"Error getting chat stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.get("/api/activity-graph", response_model=ActivityGraphData)
async def get_activity_graph():
    try:
        log_file = bot.get_current_log_file()
        if not os.path.exists(log_file):
            return ActivityGraphData(data=[], layout={})

        # Read the CSV file
        df = pd.read_csv(log_file)
        
        # Convert timestamp using a more flexible approach
        try:
            # Try parsing with exact format first
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            try:
                # If that fails, try automatic parsing
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            except Exception:
                try:
                    # Last resort: try parsing as is
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    logger.error(f"Failed to parse timestamps: {str(e)}")
                    return ActivityGraphData(data=[], layout={})

        # Print some debug information
        logger.info(f"Sample timestamp from data: {df['timestamp'].iloc[0] if len(df) > 0 else 'No data'}")
        
        # Group by hour and count messages
        hourly_activity = df.groupby(df['timestamp'].dt.hour)['message_content'].count().reset_index()
        hourly_activity.columns = ['hour', 'message_count']
        
        # Ensure we have all hours represented (0-23)
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_activity = pd.merge(all_hours, hourly_activity, on='hour', how='left')
        hourly_activity['message_count'] = hourly_activity['message_count'].fillna(0)
        
        # Sort by hour
        hourly_activity = hourly_activity.sort_values('hour')

        # Create the plot
        fig = px.bar(
            hourly_activity,
            x='hour',
            y='message_count',
            title='Messages per Hour',
            labels={'hour': 'Hour of Day', 'message_count': 'Number of Messages'}
        )

        # Customize the layout
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                ticktext=[f'{h:02d}:00' for h in range(24)],
                tickvals=list(range(24)),
                title='Hour of Day'
            ),
            yaxis=dict(
                title='Number of Messages',
                gridcolor='rgba(0,0,0,0.1)'
            ),
            plot_bgcolor='white',
            bargap=0.2
        )

        # Convert to JSON-serializable format
        fig_dict = fig.to_dict()
        
        # Use a custom JSON encoder for safety
        def json_encode_safe(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            return str(obj)
        
        cleaned_data = json.loads(json.dumps(fig_dict['data'], default=json_encode_safe))
        cleaned_layout = json.loads(json.dumps(fig_dict['layout'], default=json_encode_safe))

        return ActivityGraphData(
            data=cleaned_data,
            layout=cleaned_layout
        )
    except Exception as e:
        logger.error(f"Error generating activity graph: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    
# Dashboard HTML
dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Discord Bot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-8">Discord Server Analytics</h1>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="bg-white p-6 rounded-lg shadow-lg">
                <h2 class="text-xl font-semibold mb-4">User Activity</h2>
                <div id="activityGraph"></div>
            </div>
            
            <div class="bg-white p-6 rounded-lg shadow-lg">
                <h2 class="text-xl font-semibold mb-4">Top Users</h2>
                <div id="leaderboard" class="space-y-2"></div>
            </div>
        </div>
    </div>

    <script>
        async function fetchData() {
            try {
                // Fetch stats
                const statsResponse = await fetch('/api/chat-stats');
                if (!statsResponse.ok) {
                    throw new Error(`HTTP error! status: ${statsResponse.status}`);
                }
                const stats = await statsResponse.json();
                
                // Fetch graph data
                const graphResponse = await fetch('/api/activity-graph');
                if (!graphResponse.ok) {
                    throw new Error(`HTTP error! status: ${graphResponse.status}`);
                }
                const graphData = await graphResponse.json();
                
                // Update activity graph
                Plotly.newPlot('activityGraph', graphData.data, graphData.layout);
                
                // Update leaderboard
                const leaderboardEl = document.getElementById('leaderboard');
                const sortedUsers = stats.sort((a, b) => b.message_count - a.message_count).slice(0, 10);
                
                leaderboardEl.innerHTML = sortedUsers.map((user, index) => `
                    <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
                        <div>
                            <span class="font-bold">${index + 1}. ${user.username}</span>
                            <span class="text-sm text-gray-600">Level ${user.level}</span>
                        </div>
                        <div class="text-right">
                            <div class="font-medium">${user.message_count} messages</div>
                            <div class="text-sm text-gray-600">${Math.floor(user.xp)} XP</div>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error fetching data:', error);
                document.getElementById('activityGraph').innerHTML = '<p class="text-red-500">Error loading activity graph</p>';
                document.getElementById('leaderboard').innerHTML = '<p class="text-red-500">Error loading leaderboard data</p>';
            }
        }

        // Initial load
        fetchData();
        
        // Refresh every 30 seconds
        setInterval(fetchData, 10000);
    </script>
</body>
</html>"""


# Write dashboard HTML file
try:
    with open(static_path / "index.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    print(f"Dashboard HTML file created successfully at {static_path / 'index.html'}")
except Exception as e:
    print(f"Error creating dashboard HTML file: {str(e)}")

# Mount static files after writing HTML
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# Discord Bot Commands
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
        
        chunks = split_response(response_text)
        total_chunks = len(chunks)
        
        first_chunk = chunks[0]
        if total_chunks > 1:
            first_chunk = f"(1/{total_chunks})\n{first_chunk}"
        await interaction.followup.send(first_chunk)
        
        for i, chunk in enumerate(chunks[1:], 2):
            numbered_chunk = f"({i}/{total_chunks})\n{chunk}"
            await interaction.channel.send(numbered_chunk)
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

@bot.event
async def on_message(message):
    try:
        if message.guild.id != BotConfig.GUILD_ID or message.author == bot.user:
            return

        session = bot.Session()
        try:
            user_log = session.query(MessageLog)\
                .filter_by(user_id=str(message.author.id))\
                .first()

            if user_log:
                # Update existing user log
                user_log.message_count += 1
                user_log.xp += XP_PER_MESSAGE
                user_log.timestamp = datetime.utcnow()
                
                # Level up check
                new_level = int(user_log.xp / LEVEL_THRESHOLD) + 1
                if new_level > user_log.level:
                    user_log.level = new_level
                    await message.channel.send(f"🎉 Congratulations {message.author.mention}! You've reached level {new_level}!")
            else:
                # Create new user log
                user_log = MessageLog(
                    user_id=str(message.author.id),
                    username=message.author.name,
                    channel_id=str(message.channel.id),
                    channel_name=message.channel.name,
                    message_count=1,
                    xp=XP_PER_MESSAGE,
                    level=1
                )
                session.add(user_log)

            # Log message to CSV
            with open(bot.get_current_log_file(), 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow(),
                    message.author.id,
                    message.author.name,
                    message.channel.id,
                    message.channel.name,
                    message.content,
                    'message',
                    user_log.level,
                    user_log.xp
                ])

            session.commit()
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            session.rollback()
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error in on_message: {str(e)}")

@bot.event
async def on_message(message):
    try:
        if message.guild.id != BotConfig.GUILD_ID or message.author == bot.user:
            return

        session = bot.Session()
        try:
            user_log = session.query(MessageLog)\
                .filter_by(user_id=str(message.author.id))\
                .first()

            if user_log:
                # Update existing user log
                user_log.message_count += 1
                user_log.xp += XP_PER_MESSAGE
                user_log.timestamp = datetime.utcnow()
                
                # Level up check
                new_level = int(user_log.xp / LEVEL_THRESHOLD) + 1
                if new_level > user_log.level:
                    user_log.level = new_level
                    await message.channel.send(f"🎉 Congratulations {message.author.mention}! You've reached level {new_level}!")
            else:
                # Create new user log
                user_log = MessageLog(
                    user_id=str(message.author.id),
                    username=message.author.name,
                    channel_id=str(message.channel.id),
                    channel_name=message.channel.name,
                    message_count=1,
                    xp=XP_PER_MESSAGE,
                    level=1
                )
                session.add(user_log)

            # Log message to CSV with consistent timestamp format
            current_time = datetime.utcnow()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
            
            with open(bot.get_current_log_file(), 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    formatted_time,
                    message.author.id,
                    message.author.name,
                    message.channel.id,
                    message.channel.name,
                    message.content,
                    'message',
                    user_log.level,
                    user_log.xp
                ])

            session.commit()
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            session.rollback()
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error in on_message: {str(e)}")

@app.get("/api/activity-graph")
async def get_activity_graph():
    try:
        # Read the current month's log file
        log_file = bot.get_current_log_file()
        if not os.path.exists(log_file):
            return {"data": [], "layout": {}}

        df = pd.read_csv(log_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by hour and count messages
        hourly_activity = df.groupby(df['timestamp'].dt.hour)['message_content'].count().reset_index()
        hourly_activity.columns = ['hour', 'message_count']

        # Create the plot
        fig = px.bar(
            hourly_activity,
            x='hour',
            y='message_count',
            title='Messages per Hour',
            labels={'hour': 'Hour of Day', 'message_count': 'Number of Messages'}
        )

        return {
            "data": fig.data,
            "layout": fig.layout
        }
    except Exception as e:
        logger.error(f"Error generating activity graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_discord_bot():
    bot.run(BotConfig.DISCORD_TOKEN)

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    discord_thread = Thread(target=run_discord_bot)
    api_thread = Thread(target=run_fastapi)
    
    discord_thread.start()
    api_thread.start()
    
    discord_thread.join()
    api_thread.join()