import pytest
from unittest.mock import patch
from main import parse_signal, parse_signal_ai, connect_metatrader, get_user

@pytest.fixture
def user():
    return {'id': 1, 'reverse_trades': False, 'execution_delay': 0, 'copier_settings': '{}'}

@pytest.fixture
def channel():
    return {'use_ai_parsing': False, 'filter_keywords': '[]', 'channel_username': 'test_channel'}

def test_parse_buy_signal(user, channel):
    signal = "Buy EURUSD at 1.2000 SL 1.1900 TP 1.2100"
    trade = parse_signal(signal, user, channel)
    assert trade is not None
    assert trade['order_type'] == 'Buy'
    assert trade['symbol'] == 'EURUSD'
    assert trade['entry_price'] == 1.2000
    assert trade['stop_loss'] == 1.1900
    assert trade['take_profits'] == [1.2100]

def test_parse_sell_limit_signal(user, channel):
    signal = "Sell Limit GBPUSD at 1.4000 SL 1.4100 TP 1.3900
