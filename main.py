#!/usr/bin/env python3
import asyncio
import os
import sqlite3
import threading
import logging
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.channels import JoinChannelRequest
from googletrans import Translator
from cryptography.fernet import Fernet
import jwt
from datetime import datetime, timedelta
from prettytable import PrettyTable
from functools import wraps
import marshmallow
from marshmallow import Schema, fields, ValidationError
from logging.handlers import RotatingFileHandler
import spacy
from spacy.matcher import Matcher
import MetaTrader5 as mt5
import pytesseract
from PIL import Image

# Configuration
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')  # No default in production
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
limiter = Limiter(app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

# Logging setup
handler = RotatingFileHandler('fx_copier.log', maxBytes=1000000, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Environment variables validation
required_env_vars = ['SECRET_KEY', 'TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'ENCRYPTION_KEY']
for var in required_env_vars:
    if not os.environ.get(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"{var} environment variable is required")

TELEGRAM_API_ID = int(os.environ.get('TELEGRAM_API_ID'))
TELEGRAM_API_HASH = os.environ.get('TELEGRAM_API_HASH')
RISK_FACTOR = float(os.environ.get('RISK_FACTOR', 0.01))
PORT = int(os.environ.get('PORT', 5000))

# Encryption setup
ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY')
fernet = Fernet(ENCRYPTION_KEY)

# Allowed FX symbols
SYMBOLS = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAGUSD', 'XAUUSD']

os.makedirs('sessions', exist_ok=True)

# spaCy setup
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define patterns for spaCy matcher
patterns = [
    [
        {"LOWER": {"IN": ["buy", "sell"]}},
        {"TEXT": {"IN": SYMBOLS}},
        {"LOWER": {"IN": ["at", "@"]}},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["sl", "stoploss", "stop loss"]}},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["tp", "takeprofit", "take profit"]}},
        {"LIKE_NUM": True}
    ],
    [
        {"LOWER": {"IN": ["buy", "sell"]}},
        {"LOWER": {"IN": ["limit", "stop"]}},
        {"TEXT": {"IN": SYMBOLS}},
        {"LOWER": {"IN": ["at", "@"]}},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["sl", "stoploss", "stop loss"]}},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["tp", "takeprofit", "take profit"]}},
        {"LIKE:NUM": True}
    ],
    [
        {"LOWER": {"IN": ["buy", "sell"]}},
        {"TEXT": {"IN": SYMBOLS}},
        {"LOWER": {"IN": ["at", "@"]}},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["sl", "stoploss", "stop loss"]}},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["tp1", "takeprofit1", "take profit 1"]}},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["tp2", "takeprofit2", "take profit 2"]}},
        {"LIKE_NUM": True}
    ]
]
for i, pattern in enumerate(patterns):
    matcher.add(f"TRADE_PATTERN_{i}", [pattern])

# Database setup
db_lock = threading.Lock()

def get_db_connection():
    with db_lock:
        conn = sqlite3.connect('fx_copier.db', check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        phone TEXT UNIQUE,
        session_string TEXT,
        risk_percentage REAL,
        subscription_status TEXT,
        reverse_trades BOOLEAN DEFAULT 0,
        execution_delay INTEGER DEFAULT 0,
        role TEXT DEFAULT 'user',
        copier_settings TEXT DEFAULT '{}'
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS mt_accounts (
        user_id INTEGER,
        account_id TEXT,
        login TEXT,
        password TEXT,
        server TEXT,
        balance REAL,
        equity REAL,
        margin REAL,
        free_margin REAL,
        status TEXT,
        last_updated TEXT,
        PRIMARY KEY (user_id, account_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS channels (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        channel_username TEXT,
        signal_format TEXT,
        is_active INTEGER,
        use_ai_parsing BOOLEAN DEFAULT 0,
        filter_keywords TEXT DEFAULT '[]'
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        channel_id INTEGER,
        original_message TEXT,
        parsed_data TEXT,
        status TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        signal_id INTEGER,
        symbol TEXT,
        entry REAL,
        sl REAL,
        tp TEXT,
        position_size REAL,
        result TEXT,
        trailing_sl REAL,
        retry_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'open'
    )''')
    conn.commit()
    conn.close()

init_db()

# Marshmallow Schemas
class RegisterSchema(Schema):
    phone = fields.Str(required=True, validate=marshmallow.validate.Length(min=10, max=15))

class TelegramCodeSchema(Schema):
    phone = fields.Str(required=True, validate=marshmallow.validate.Length(min=10, max=15))
    code = fields.Str(required=False)
    phone_code_hash = fields.Str(required=False)
    password = fields.Str(required=False)

class ChannelSchema(Schema):
    channel_username = fields.Str(required=True, validate=marshmallow.validate.Length(min=5))

class MTAccountSchema(Schema):
    account_id = fields.Str(required=True)
    login = fields.Str(required=True)
    password = fields.Str(required=True)
    server = fields.Str(required=True)

class CopierSettingsSchema(Schema):
    auto_copy = fields.Boolean(required=False, default=False)
    risk_per_trade = fields.Float(required=False, default=0.0)
    max_daily_loss = fields.Float(required=False, default=0.0)
    max_open_trades = fields.Integer(required=False, default=0)

# Authentication decorators
def token_required(f):
    @wraps(f)
    async def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or 'Bearer ' not in token:
            return jsonify({'error': 'Token is missing or invalid'}), 401
        try:
            token = token.split("Bearer ")[1]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return await f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    async def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            token = token.split("Bearer ")[1]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT role FROM users WHERE id = ?', (data['user_id'],))
            user = c.fetchone()
            conn.close()
            if not user or user['role'] != 'admin':
                return jsonify({'error': 'Admin access required'}), 403
            request.user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return await f(*args, **kwargs)
    return decorated

def generate_token(user_id):
    payload = {'user_id': user_id, 'exp': datetime.utcnow() + timedelta(hours=24)}
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

# Async helper
def run_async_in_thread(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Signal Processing
translator = Translator()

def parse_signal_ai(signal_text):
    try:
        doc = nlp(signal_text)
        matches = matcher(doc)
        if not matches:
            return None

        trade = {
            "order_type": None,
            "symbol": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profits": []
        }

        for match_id, start, end in matches:
            span = doc[start:end]
            tokens = [token.text.lower() for token in span]

            if tokens[0] in ["buy", "sell"]:
                trade["order_type"] = tokens[0].capitalize()
                if tokens[1] in ["limit", "stop"]:
                    trade["order_type"] += f" {tokens[1].capitalize()}"

            for token in tokens:
                if token.upper() in SYMBOLS:
                    trade["symbol"] = token.upper()
                    break

            numbers = [float(token.text) for token in span if token.like_num]
            for i, token in enumerate(tokens):
                if token in ["at", "@"] and i + 1 < len(tokens) and tokens[i + 1].replace('.', '').isdigit():
                    trade["entry_price"] = float(tokens[i + 1])
                elif token in ["sl", "stoploss", "stop loss"] and i + 1 < len(tokens) and tokens[i + 1].replace('.', '').isdigit():
                    trade["stop_loss"] = float(tokens[i + 1])
                elif token in ["tp", "takeprofit", "take profit", "tp1", "takeprofit1", "take profit 1"] and i + 1 < len(tokens) and tokens[i + 1].replace('.', '').isdigit():
                    trade["take_profits"].append(float(tokens[i + 1]))
                elif token in ["tp2", "takeprofit2", "take profit 2"] and i + 1 < len(tokens) and tokens[i + 1].replace('.', '').isdigit():
                    trade["take_profits"].append(float(tokens[i + 1]))

        if all(trade[key] is not None for key in ["order_type", "symbol", "entry_price", "stop_loss"]) and trade["take_profits"]:
            return trade
        return None
    except Exception as e:
        logger.error(f"spaCy parsing error: {str(e)}")
        return None

def ocr_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text if text.strip() else ""
    except Exception as e:
        logger.error(f"Tesseract OCR error: {str(e)}")
        return ""

def parse_signal(signal_text, user, channel):
    try:
        lang = translator.detect(signal_text).lang
        if lang != 'en':
            signal_text = translator.translate(signal_text, dest='en').text

        try:
            filter_keywords = json.loads(channel.get('filter_keywords', '[]'))
        except json.JSONDecodeError:
            logger.warning(f"Invalid filter_keywords for channel {channel.get('channel_username')}")
            filter_keywords = []

        if any(keyword.lower() in signal_text.lower() for keyword in filter_keywords):
            logger.info(f"Signal filtered out due to keywords: {signal_text}")
            return None

        if channel['use_ai_parsing']:
            trade = parse_signal_ai(signal_text)
            if trade and trade['symbol'] in SYMBOLS:
                if user['reverse_trades']:
                    trade['order_type'] = 'Sell' if trade['order_type'] == 'Buy' else 'Buy'
                return trade

        lines = [line.strip() for line in signal_text.strip().split('\n') if line.strip()]
        trade = {}
        order_types = {'buy': 'Buy', 'sell': 'Sell', 'buy limit': 'Buy Limit', 'sell limit': 'Sell Limit', 'buy stop': 'Buy Stop', 'sell stop': 'Sell Stop'}
        for ot in order_types:
            if ot.lower() in lines[0].lower():
                trade['order_type'] = order_types[ot]
                break
        else:
            return None

        parts = lines[0].split()
        for part in parts:
            if part.upper() in SYMBOLS:
                trade['symbol'] = part.upper()
                break
        else:
            return None

        for line in lines[1:]:
            line_lower = line.lower()
            if 'entry' in line_lower or 'price' in line_lower:
                numbers = [float(s) for s in line.split() if s.replace('.', '').isdigit()]
                if numbers:
                    trade['entry_price'] = numbers[0]
            elif 'sl' in line_lower or 'stop loss' in line_lower:
                numbers = [float(s) for s in line.split() if s.replace('.', '').isdigit()]
                if numbers:
                    trade['stop_loss'] = numbers[0]
            elif 'tp' in line_lower or 'take profit' in line_lower:
                numbers = [float(s) for s in line.split() if s.replace('.', '').isdigit()]
                trade['take_profits'] = numbers if numbers else []

        if 'symbol' not in trade or 'entry_price' not in trade or 'stop_loss' not in trade:
            return None

        if user['reverse_trades']:
            trade['order_type'] = 'Sell' if trade['order_type'] == 'Buy' else 'Buy'

        return trade
    except Exception as e:
        logger.error(f"Signal parsing error: {str(e)}")
        return None

# Telegram Integration
async def get_user_telegram_client(user_id):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT session_string FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()
        if not user or not user['session_string']:
            raise Exception("Telegram not connected")
        decrypted_session = fernet.decrypt(user['session_string'].encode()).decode()
        client = TelegramClient(decrypted_session, TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.connect()
        if not await client.is_user_authorized():
            logger.error(f"Invalid session for user {user_id}")
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('UPDATE users SET session_string = NULL WHERE id = ?', (user_id,))
            conn.commit()
            conn.close()
            raise Exception("Session invalid")
        logger.info(f"Telegram client connected for user {user_id}")
        return client
    except Exception as e:
        logger.error(f"Failed to get Telegram client for user {user_id}: {str(e)}")
        raise

async def start_telegram_client(user_id, phone):
    try:
        client = TelegramClient(f'sessions/{user_id}', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.start(phone=phone)
        conn = get_db_connection()
        c = conn.cursor()
        encrypted_session = fernet.encrypt(client.session.save().encode()).decode()
        c.execute('UPDATE users SET session_string = ? WHERE id = ?', (encrypted_session, user_id))
        conn.commit()
        conn.close()
        logger.info(f"Telegram client started for user {user_id}")
        return client
    except SessionPasswordNeededError:
        socketio.emit('reauth_needed', {'user_id': user_id, 'message': '2FA required'}, namespace='/signals')
        raise
    except Exception as e:
        logger.error(f"Failed to start Telegram client for user {user_id}: {str(e)}")
        raise

async def monitor_channels(user_id, client):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id, channel_username, use_ai_parsing, filter_keywords FROM channels WHERE user_id = ? AND is_active = 1', (user_id,))
        channels = c.fetchall()
        conn.close()

        @client.on(events.NewMessage(chats=[ch['channel_username'] for ch in channels]))
        async def handler(event):
            user = get_user(user_id)
            channel = next((ch for ch in channels if ch['channel_username'] == event.message.chat.username), None)
            signal_text = event.message.text
            if event.photo:
                photo_path = await event.download_media(file=f'temp_{user_id}.jpg')
                signal_text = ocr_image(photo_path)
                os.remove(photo_path)
            trade = parse_signal(signal_text, user, channel)
            if trade:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute('INSERT INTO signals (user_id, channel_id, original_message, parsed_data, status) VALUES (?, ?, ?, ?, ?)',
                          (user_id, channel['id'], signal_text, json.dumps(trade), 'pending'))
                signal_id = c.lastrowid
                conn.commit()
                conn.close()
                socketio.emit('new_signal', {'user_id': user_id, 'signal': trade, 'signal_id': signal_id}, namespace='/signals')
                logger.info(f"New signal received for user {user_id}: {trade}")

        await client.run_until_disconnected()
    except Exception as e:
        logger.error(f"Error monitoring channels for user {user_id}: {str(e)}")
        socketio.emit('error', {'user_id': user_id, 'message': f"Channel monitoring failed: {str(e)}"}, namespace='/signals')

# MetaTrader Integration
async def connect_metatrader(user_id, trade, enter_trade=False, signal_id=None):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT account_id, login, password, server, balance FROM mt_accounts WHERE user_id = ?', (user_id,))
        account = c.fetchone()
        conn.close()
        if not account:
            raise Exception('No MetaTrader account linked')

        if not mt5.initialize():
            raise Exception('MetaTrader5 initialization failed')

        decrypted_password = fernet.decrypt(account['password'].encode()).decode()
        if not mt5.login(login=int(account['login']), password=decrypted_password, server=account['server']):
            raise Exception('MetaTrader5 login failed')

        account_info = mt5.account_info()
        if not account_info:
            raise Exception('Failed to retrieve account info')

        if trade['entry_price'] == 'NOW':
            symbol_info = mt5.symbol_info(trade['symbol'])
            if not symbol_info:
                raise Exception(f'Symbol {trade["symbol"]} not found')
            trade['entry_price'] = symbol_info.bid if trade['order_type'] == 'Buy' else symbol_info.ask

        trade_info = get_trade_information(trade, account_info.balance)
        if enter_trade:
            user = get_user(user_id)
            trade_result = await execute_trade_with_retry(user, trade, user_id, signal_id)
            trade_info['result'] = trade_result
        return trade_info
    except Exception as e:
        logger.error(f"MetaTrader connection error for user {user_id}: {str(e)}")
        socketio.emit('error', {'user_id': user_id, 'message': f"MetaTrader error: {str(e)}"}, namespace='/signals')
        return {'error': str(e)}
    finally:
        mt5.shutdown()

async def execute_trade_with_retry(user, trade, user_id, signal_id, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            result = await execute_trade(user, trade, user_id, signal_id)
            logger.info(f"Trade executed successfully for user {user_id}: {trade}")
            return result
        except Exception as e:
            retry_count += 1
            logger.warning(f"Trade execution failed for user {user_id} (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count == max_retries:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute('UPDATE trades SET status = ?, result = ? WHERE signal_id = ?', ('failed', str(e), signal_id))
                conn.commit()
                conn.close()
                logger.error(f"Max retries reached for trade execution for user {user_id}: {str(e)}")
                raise Exception(f"Trade execution failed after {max_retries} attempts: {str(e)}")
            await asyncio.sleep(2 ** retry_count)  # Exponential backoff

async def execute_trade(user, trade, user_id, signal_id):
    try:
        await asyncio.sleep(user['execution_delay'])
        symbol_info = mt5.symbol_info(trade['symbol'])
        if not symbol_info:
            raise Exception(f'Symbol {trade["symbol"]} not found')

        point = symbol_info.point
        volume_per_tp = trade['position_size'] / len(trade['take_profits'])
        trade_ids = []

        for tp in trade['take_profits']:
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': trade['symbol'],
                'volume': volume_per_tp,
                'type': mt5.ORDER_TYPE_BUY if trade['order_type'] == 'Buy' else mt5.ORDER_TYPE_SELL,
                'price': trade['entry_price'],
                'sl': trade['stop_loss'],
                'tp': tp,
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }
            if 'Limit' in trade['order_type']:
                request['action'] = mt5.TRADE_ACTION_PENDING
                request['type'] = mt5.ORDER_TYPE_BUY_LIMIT if trade['order_type'] == 'Buy Limit' else mt5.ORDER_TYPE_SELL_LIMIT
            elif 'Stop' in trade['order_type']:
                request['action'] = mt5.TRADE_ACTION_PENDING
                request['type'] = mt5.ORDER_TYPE_BUY_STOP if trade['order_type'] == 'Buy Stop' else mt5.ORDER_TYPE_SELL_STOP

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f'Trade failed: {result.comment}')
            trade_ids.append(result.order)

        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO trades (user_id, signal_id, symbol, entry, sl, tp, position_size, result, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                  (user_id, signal_id, trade['symbol'], trade['entry_price'], trade['stop_loss'], json.dumps(trade['take_profits']), trade['position_size'], 'open', 'open'))
        trade_id = c.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Trade executed for user {user_id}: {trade['symbol']}")
        return 'Trade executed successfully'
    except Exception as e:
        logger.error(f"Trade execution failed for user {user_id}: {str(e)}")
        raise

# Routes
@app.route('/')
async def index():
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
@limiter.limit("5 per minute")
async def register():
    try:
        schema = RegisterSchema()
        data = schema.load(request.json)
        phone = data['phone']
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO users (phone, risk_percentage, subscription_status) VALUES (?, ?, ?)', (phone, RISK_FACTOR, 'active'))
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        threading.Thread(target=run_async_in_thread, args=(start_telegram_client(user_id, phone),)).start()
        logger.info(f"User registered: {phone}, ID: {user_id}")
        return jsonify({'token': generate_token(user_id), 'user_id': user_id})
    except ValidationError as err:
        logger.error(f"Registration validation error: {err.messages}")
        return jsonify({'error': err.messages}), 400
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/telegram/request_code', methods=['POST'])
@token_required
async def request_telegram_code():
    try:
        schema = TelegramCodeSchema()
        data = schema.load(request.json)
        phone = data['phone']
        user_id = request.user_id
        client = TelegramClient(f'sessions/temp_{user_id}', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.connect()
        result = await client.send_code_request(phone)
        logger.info(f"Telegram code requested for user {user_id}, phone: {phone}")
        return jsonify({'phone_code_hash': result.phone_code_hash, 'status': 'code_sent'})
    except ValidationError as err:
        logger.error(f"Telegram code request validation error: {err.messages}")
        return jsonify({'error': err.messages}), 400
    except Exception as e:
        logger.error(f"Telegram code request error for user {request.user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/telegram/verify_code', methods=['POST'])
@token_required
async def verify_telegram_code():
    try:
        schema = TelegramCodeSchema()
        data = schema.load(request.json)
        phone = data['phone']
        code = data['code']
        phone_code_hash = data['phone_code_hash']
        password = data.get('password')
        user_id = request.user_id
        client = TelegramClient(f'sessions/temp_{user_id}', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.connect()
        await client.sign_in(phone, code, phone_code_hash=phone_code_hash)
        if password:
            await client.sign_in(password=password)
        encrypted_session = fernet.encrypt(client.session.save().encode()).decode()
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('UPDATE users SET session_string = ? WHERE id = ?', (encrypted_session, user_id))
        conn.commit()
        conn.close()
        logger.info(f"Telegram verified for user {user_id}")
        return jsonify({'status': 'telegram_connected'})
    except ValidationError as err:
        logger.error(f"Telegram code verification validation error: {err.messages}")
        return jsonify({'error': err.messages}), 400
    except Exception as e:
        logger.error(f"Telegram code verification error for user {request.user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/telegram/status', methods=['GET'])
@token_required
async def telegram_status():
    user_id = request.user_id
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT session_string FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    if user and user['session_string']:
        return jsonify({'status': 'connected'})
    return jsonify({'status': 'not_connected'})

@app.route('/api/channels/search', methods=['GET'])
@token_required
async def search_channels():
    try:
        query = request.args.get('query', '')
        user_id = request.user_id
        client = await get_user_telegram_client(user_id)
        result = await client.get_dialogs()
        channels = [
            {'username': dialog.entity.username, 'title': dialog.entity.title, 'description': 'Sample channel'}
            for dialog in result if dialog.is_channel and query.lower() in dialog.entity.title.lower()
        ]
        logger.info(f"Channels searched for user {user_id}: {len(channels)} found")
        return jsonify({'channels': channels})
    except Exception as e:
        logger.error(f"Channel search error for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/channels/join', methods=['POST'])
@token_required
async def join_channel():
    try:
        schema = ChannelSchema()
        data = schema.load(request.json)
        channel_username = data['channel_username']
        user_id = request.user_id
        client = await get_user_telegram_client(user_id)
        await client(JoinChannelRequest(channel_username))
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO channels (user_id, channel_username, is_active) VALUES (?, ?, ?)',
                  (user_id, channel_username, 1))
        conn.commit()
        conn.close()
        logger.info(f"User {user_id} joined channel {channel_username}")
        return jsonify({'status': 'channel_joined'})
    except ValidationError as err:
        logger.error(f"Channel join validation error: {err.messages}")
        return jsonify({'error': err.messages}), 400
    except Exception as e:
        logger.error(f"Channel join error for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/channels/preview_signals', methods=['POST'])
@token_required
async def preview_channel_signals():
    try:
        schema = ChannelSchema()
        data = schema.load(request.json)
        channel_username = data['channel_username']
        user_id = request.user_id
        client = await get_user_telegram_client(user_id)
        messages = await client.get_messages(channel_username, limit=5)
        user = get_user(user_id)
        signals = [
            {
                'message': msg.text,
                'parsed': parse_signal(msg.text, user, {'use_ai_parsing': False, 'filter_keywords': '[]'}),
                'timestamp': msg.date.strftime('%Y-%m-%d %H:%M:%S')
            }
            for msg in messages if parse_signal(msg.text, user, {'use_ai_parsing': False, 'filter_keywords': '[]'})
        ]
        logger.info(f"Previewed signals for user {user_id} from channel {channel_username}")
        return jsonify({'recent_signals': signals})
    except ValidationError as err:
        logger.error(f"Signal preview validation error: {err.messages}")
        return jsonify({'error': err.messages}), 400
    except Exception as e:
        logger.error(f"Signal preview error for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/channels/settings', methods=['POST'])
@token_required
async def channel_settings():
    try:
        schema = CopierSettingsSchema()
        data = schema.load(request.json)
        channel_id = data.get('channel_id')
        use_ai_parsing = data.get('use_ai_parsing', False)
        filter_keywords = data.get('filter_keywords', [])
        user_id = request.user_id
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('UPDATE channels SET use_ai_parsing = ?, filter_keywords = ? WHERE id = ? AND user_id = ?',
                  (use_ai_parsing, json.dumps(filter_keywords), channel_id, user_id))
        conn.commit()
        conn.close()
        logger.info(f"Channel settings updated for user {user_id}, channel {channel_id}")
        return jsonify({'status': 'settings_updated'})
    except ValidationError as err:
        logger.error(f"Channel settings validation error: {err.messages}")
        return jsonify({'error': err.messages}), 400
    except Exception as e:
        logger.error(f"Channel settings error for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/channels/toggle', methods=['POST'])
@token_required
async def toggle_channel():
    try:
        schema = ChannelSchema()
        data = schema.load(request.json)
        channel_id = data['channel_id']
        user_id = request.user_id
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('UPDATE channels SET is_active = NOT is_active WHERE id = ? AND user_id = ?', (channel_id, user_id))
        conn.commit()
        conn.close()
        logger.info(f"Channel toggled for user {user_id}, channel {channel_id}")
        return jsonify({'status': 'channel_toggled'})
    except ValidationError as err:
        logger.error(f"Channel toggle validation error: {err.messages}")
        return jsonify({'error': err.messages}), 400
    except Exception as e:
        logger.error(f"Channel toggle error for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/copier/settings', methods=['GET', 'POST'])
@token_required
async def copier_settings():
    user_id = request.user_id
    conn = get_db_connection()
    c = conn.cursor()
    if request.method == 'GET':
        c.execute('SELECT copier_settings FROM users WHERE id = ?', (user_id,))
        settings = c.fetchone()['copier_settings']
        conn.close()
        return jsonify(json.loads(settings) if settings else {})
    if request.method == 'POST':
        try:
            schema = CopierSettingsSchema()
            settings = schema.load(request.json)
            c.execute('UPDATE users SET copier_settings = ? WHERE id = ?', (json.dumps(settings), user_id))
            conn.commit()
            conn.close()
            logger.info(f"Copier settings updated for user {user_id}")
            return jsonify({'status': 'settings_updated'})
        except ValidationError as err:
            logger.error(f"Copier settings validation error: {err.messages}")
            return jsonify({'error': err.messages}), 400
        except Exception as e:
            logger.error(f"Copier settings error for user {user_id}: {str(e)}")
            return jsonify({'error': str(e)}), 400

@app.route('/api/mt/accounts', methods=['GET'])
@token_required
async def get_mt_accounts():
    try:
        user_id = request.user_id
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT account_id, server, balance, equity, margin, free_margin, status, last_updated FROM mt_accounts WHERE user_id = ?', (user_id,))
        accounts = c.fetchall()
        conn.close()
        logger.info(f"MT accounts retrieved for user {user_id}")
        return jsonify({'accounts': [dict(acc) for acc in accounts]})
    except Exception as e:
        logger.error(f"MT accounts retrieval error for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/mt/test_connection', methods=['POST'])
@token_required
async def test_mt_connection():
    try:
        schema = MTAccountSchema()
        data = schema.load(request.json)
        login = data['login']
        password = data['password']
        server = data['server']
        if not mt5.initialize():
            return jsonify({'error': 'MetaTrader5 initialization failed'}), 400
        if not mt5.login(login=int(login), password=password, server=server):
            return jsonify({'error': 'MetaTrader5 login failed'}), 400
        account_info = mt5.account_info()
        mt5.shutdown()
        logger.info(f"MT connection tested successfully for login {login}")
        return jsonify({
            'connection_status': 'success',
            'account_info': {
                'balance': account_info.balance,
                'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ping': 45  # Placeholder
            }
        })
    except ValidationError as err:
        logger.error(f"MT connection test validation error: {err.messages}")
        return jsonify({'error': err.messages}), 400
    except Exception as e:
        logger.error(f"MT connection test error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/add_mt_account', methods=['POST'])
@token_required
@limiter.limit("5 per hour")
async def add_mt_account():
    try:
        schema = MTAccountSchema()
        data = schema.load(request.json)
        user_id = request.user_id
        encrypted_password = fernet.encrypt(data['password'].encode()).decode()
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO mt_accounts (user_id, account_id, login, password, server, balance, status�

System: Trader5 initialization failed'}), 400
        if not mt5.login(login=int(login), password=password, server=server):
            return jsonify({'error': 'MetaTrader5 login failed'}), 400
        account_info = mt5.account_info()
        mt5.shutdown()
        c.execute('INSERT OR REPLACE INTO mt_accounts (user_id, account_id, login, password, server, balance, status, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                  (user_id, data['account_id'], data['login'], encrypted_password, data['server'], account_info.balance, 'connected', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        logger.info(f"MT account added for user {user_id}: {data['account_id']}")
        return jsonify({'status': 'MT account added successfully'})
    except ValidationError as err:
        logger.error(f"MT account validation error: {err.messages}")
        return jsonify({'error': err.messages}), 400
    except Exception as e:
        logger.error(f"Error adding MT account for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/copy_signal', methods=['POST'])
@token_required
async def copy_signal():
    try:
        data = request.json
        signal_id = data['signal_id']
        user_id = request.user_id
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT parsed_data FROM signals WHERE id = ? AND user_id = ?', (signal_id, user_id))
        signal = c.fetchone()
        conn.close()
        if not signal:
            return jsonify({'error': 'Signal not found'}), 404
        trade = json.loads(signal['parsed_data'])
        result = await connect_metatrader(user_id, trade, enter_trade=True, signal_id=signal_id)
        logger.info(f"Signal copied for user {user_id}, signal_id: {signal_id}")
        return jsonify({'status': 'signal_copied', 'result': result})
    except Exception as e:
        logger.error(f"Signal copy error for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/admin/users')
@admin_required
async def admin_users():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id, phone, role, reverse_trades, execution_delay, subscription_status FROM users')
        users = c.fetchall()
        conn.close()
        logger.info(f"Admin accessed user management")
        return render_template('admin_users.html', users=[dict(u) for u in users])
    except Exception as e:
        logger.error(f"Admin users error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/admin/analytics')
@admin_required
async def admin_analytics():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) as active_users FROM users WHERE subscription_status = ?', ('active',))
        active_users = c.fetchone()['active_users']
        c.execute('SELECT COUNT(*) as signals_copied FROM trades')
        signals_copied = c.fetchone()['signals_copied']
        c.execute('SELECT COUNT(*) as successful_trades FROM trades WHERE status = ?', ('closed_success',))
        successful_trades = c.fetchone()['successful_trades']
        conn.close()
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["Active Users", active_users])
        table.add_row(["Signals Copied", signals_copied])
        table.add_row(["Successful Trades", successful_trades])
        logger.info(f"Admin accessed analytics")
        return jsonify({
            'analytics': {
                'active_users': active_users,
                'signals_copied': signals_copied,
                'successful_trades': successful_trades
            }
        })
    except Exception as e:
        logger.error(f"Admin analytics error: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Helper Functions
def get_user(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def get_trade_information(trade, balance):
    # Placeholder function for trade calculations
    return {
        'symbol': trade['symbol'],
        'position_size': balance * RISK_FACTOR / len(trade['take_profits']),
        'entry_price': trade['entry_price'],
        'stop_loss': trade['stop_loss'],
        'take_profits': trade['take_profits']
    }

async def startup_recovery():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE session_string IS NOT NULL')
        users = c.fetchall()
        conn.close()
        for user in users:
            threading.Thread(target=run_async_in_thread, args=(monitor_channels(user['id'], await get_user_telegram_client(user['id'])),)).start()
        logger.info("Startup recovery completed")
    except Exception as e:
        logger.error(f"Startup recovery error: {str(e)}")

if __name__ == '__main__':
    import eventlet
    eventlet.monkey_patch()
    threading.Thread(target=run_async_in_thread, args=(startup_recovery(),)).start()
    socketio.run(app, host='0.0.0.0', port=PORT)
