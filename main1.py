import os
import logging
import discord
from discord import app_commands
from discord.ext import commands
import google.generativeai as genai
from datetime import datetime, timedelta
import csv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, func
from sqlalchemy.orm import declarative_base, sessionmaker
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
import asyncio
import uvicorn
from threading import Thread

# Load environment variables
load_dotenv()

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

# Global Configuration
XP_PER_MESSAGE = 1
LEVEL_THRESHOLD = 100
MESSAGE_DELAY = 1

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
                    'timestamp', 'user_id', 'username', 'channel_id',
                    'channel_name', 'message_content', 'message_type',
                    'level', 'xp'
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

# Initialize the bot
bot = DiscordBot()

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory
os.makedirs("static", exist_ok=True)

html_content = """
<!DOCTYPE html>
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
            <!-- Activity Graph -->
            <div class="bg-white p-6 rounded-lg shadow-lg">
                <h2 class="text-xl font-semibold mb-4">User Activity</h2>
                <div id="activityGraph"></div>
            </div>
            
            <!-- Leaderboard -->
            <div class="bg-white p-6 rounded-lg shadow-lg">
                <h2 class="text-xl font-semibold mb-4">Top Users</h2>
                <div id="leaderboard" class="space-y-2"></div>
            </div>
        </div>
    </div>

    <script>
        async function fetchData() {
            try {
                const statsResponse = await fetch('/api/chat-stats');
                const stats = await statsResponse.json();
                
                const graphResponse = await fetch('/api/activity-graph');
                const graphData = await graphResponse.json();
                
                // Update activity graph with the new format
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
            }
        }

        // Initial load
        fetchData();
        
        // Refresh every 30 seconds
        setInterval(fetchData, 30000);
    </script>
</body>
</html>
"""

with open("static/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML page"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving dashboard")

@app.get("/api/chat-stats")
async def get_chat_stats():
    """Get chat statistics for all users"""
    try:
        session = bot.Session()
        stats = session.query(MessageLog).all()
        return JSONResponse(content=[
            {
                "username": stat.username,
                "message_count": stat.message_count,
                "level": stat.level,
                "xp": stat.xp
            }
            for stat in stats
        ])
    except Exception as e:
        logger.error(f"Error fetching chat stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching chat statistics")
    finally:
        session.close()

@app.get("/api/activity-graph")
async def get_activity_graph():
    """Get activity graph data"""
    try:
        session = bot.Session()
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        
        activity = session.query(
            func.strftime('%Y-%m-%d %H:00:00', MessageLog.timestamp).label('hour'),
            func.count().label('message_count')
        ).filter(
            MessageLog.timestamp >= seven_days_ago
        ).group_by(
            'hour'
        ).all()
        
        df = pd.DataFrame(activity, columns=['hour', 'message_count'])
        df['hour'] = pd.to_datetime(df['hour'])
        
        # Create simplified data structure for the graph
        graph_data = {
            "data": [
                {
                    "x": df['hour'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    "y": df['message_count'].tolist(),
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Messages"
                }
            ],
            "layout": {
                "title": "Message Activity (Last 7 Days)",
                "xaxis": {
                    "title": "Time",
                    "type": "date"
                },
                "yaxis": {
                    "title": "Number of Messages"
                },
                "hovermode": "x unified"
            }
        }
        
        return JSONResponse(content=graph_data)
    
    except Exception as e:
        logger.error(f"Error generating activity graph: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating activity graph")
    finally:
        session.close()

# Bot event handlers
@bot.event
async def on_message(message):
    try:
        if message.author == bot.user or (message.guild and message.guild.id != BotConfig.GUILD_ID):
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
            user_log.timestamp = datetime.utcnow()

            new_level = int(user_log.xp / LEVEL_THRESHOLD) + 1
            if new_level > user_log.level:
                user_log.level = new_level
                await message.channel.send(
                    f'🎉 Congratulations {message.author.mention}! You reached level {new_level}!'
                )

            # Log to CSV
            with open(bot.get_current_log_file(), 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    str(message.author.id),
                    message.author.name,
                    str(message.channel.id),
                    message.channel.name,
                    message.content,
                    'message',
                    user_log.level,
                    user_log.xp
                ])

            session.commit()

        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            session.rollback()
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")

# Main execution
if __name__ == "__main__":
    try:
        logger.info("Starting Discord bot and web dashboard...")
        
        # Create and start the FastAPI server in a separate thread
        def run_api():
            uvicorn.run(app, host="0.0.0.0", port=8000)
        
        api_thread = Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Start the Discord bot
        bot.run(BotConfig.DISCORD_TOKEN)
        
    except Exception as e:
        logger.critical(f"Critical error starting services: {str(e)}")