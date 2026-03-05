# ---------------------------------------------------------------------------
# 01_agent.py — The bare minimum Telegram bot.
#
# No memory, no tools, no personality.
# Each message is a completely fresh, independent call to GPT-4o.
# Closing and reopening Telegram starts from zero every time.
#
# Run: python 01_agent.py
# ---------------------------------------------------------------------------

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

load_dotenv()

# Direct LLM client — no graph, no state, no memory
client = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))

async def handle_message(update: Update, context):
    user_message = update.message.text

    # Each call is stateless — the model sees only this single message
    response = await client.ainvoke([HumanMessage(content=user_message)])

    await update.message.reply_text(response.content)

app = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.run_polling()
