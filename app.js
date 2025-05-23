class FXSignalCopier {
    constructor() {
        this.token = null;
        this.userId = null;
        this.socket = io('/signals');
        this.phoneCodeHash = null;
        this.currentChannelId = null;
        this.initEventListeners();
        this.checkServiceWorker();
        this.loadOnboarding();
    }

    initEventListeners() {
        // Authentication
        document.getElementById('login-btn').addEventListener('click', () => this.login());
        document.getElementById('register-btn').addEventListener('click', () => this.register());

        // Telegram
        document.getElementById('connect-telegram-btn').addEventListener('click', () => this.showTelegramModal());
        document.getElementById('submit-telegram-btn').addEventListener('click', () => this.connectTelegram());

        // Channels
        document.getElementById('search-channels-btn').addEventListener('click', () => this.searchChannels());
        document.getElementById('channels-table').addEventListener('click', (e) => {
            if (e.target.classList.contains('join-channel')) this.joinChannel(e.target.dataset.username);
            if (e.target.classList.contains('toggle-channel')) this.toggleChannel(e.target.dataset.id);
            if (e.target.classList.contains('channel-settings')) this.showChannelSettings(e.target.dataset.id);
        });
        document.getElementById('save-channel-settings').addEventListener('click', () => this.saveChannelSettings());

        // Settings
        document.getElementById('save-settings-btn').addEventListener('click', () => this.saveSettings());

        // MetaTrader Accounts
        document.getElementById('add-mt-account-btn').addEventListener('click', () => this.showMTModal());
        document.getElementById('submit-mt-account').addEventListener('click', () => this.addMTAccount());

        // Signals
        document.getElementById('signals-table').addEventListener('click', (e) => {
            if (e.target.classList.contains('copy-signal')) {
                this.copySignal(e.target.dataset.signalId);
            }
        });

        // SocketIO
        this.socket.on('new_signal', (data) => this.handleNewSignal(data));
        this.socket.on('error', (data) => this.handleError(data));
        this.socket.on('reauth_needed', (data) => this.handleReauthNeeded(data));
    }

    async login() {
        const phone = document.getElementById('phone').value;
        if (!phone || !/^\+?\d{10,15}$/.test(phone)) return alert('Please enter a valid phone number');
        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phone })
            });
            const data = await response.json();
            if (data.token) {
                this.token = data.token;
                this.userId = data.user_id;
                this.showDashboard();
                this.loadTelegramStatus();
                this.loadChannels();
                this.loadSettings();
                this.loadMTAccounts();
                this.checkOnboarding();
            } else {
                alert(data.error || 'Login failed');
            }
        } catch (e) {
            alert('Login failed: ' + e.message);
        }
    }

    async register() {
        const phone = document.getElementById('phone').value;
        if (!phone || !/^\+?\d{10,15}$/.test(phone)) return alert('Please enter a valid phone number');
        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phone })
            });
            const data = await response.json();
            if (data.token) {
                this.token = data.token;
                this.userId = data.user_id;
                this.showDashboard();
                this.loadTelegramStatus();
                this.loadChannels();
                this.loadSettings();
                this.loadMTAccounts();
                this.checkOnboarding();
            } else {
                alert(data.error || 'Registration failed');
            }
        } catch (e) {
            alert('Registration failed: ' + e.message);
        }
    }

    showDashboard() {
        document.getElementById('auth-section').style.display = 'none';
        document.getElementById('telegram-section').style.display = 'block';
        document.getElementById('channels-section').style.display = 'block';
        document.getElementById('settings-section').style.display = 'block';
        document.getElementById('mt-accounts-section').style.display = 'block';
        document.getElementById('trading-section').style.display = 'block';
        document.getElementById('onboarding-section').style.display = 'block';
    }

    async loadTelegramStatus() {
        try {
            const response = await fetch('/api/telegram/status', {
                headers: { 'Authorization': `Bearer ${this.token}` }
            });
            const data = await response.json();
            document.getElementById('telegram-status').textContent = `Status: ${data.status === 'connected' ? 'Connected' : 'Not Connected'}`;
        } catch (e) {
            alert('Failed to load Telegram status: ' + e.message);
        }
    }

    showTelegramModal() {
        document.getElementById('telegram-modal').style.display = 'block';
        document.getElementById('telegram-phone').style.display = 'block';
        document.getElementById('telegram-code').style.display = 'none';
        document.getElementById('telegram-password').style.display = 'none';
    }

    async connectTelegram() {
        const phone = document.getElementById('telegram-phone').value;
        const code = document.getElementById('telegram-code').value;
        const password = document.getElementById('telegram-password').value;

        if (!phone || !/^\+?\d{10,15}$/.test(phone)) return alert('Please enter a valid phone number');
        if (code && !/^\d{5}$/.test(code)) return alert('Please enter a valid 5-digit code');

        try {
            if (!code) {
                const response = await fetch('/api/telegram/request_code', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.token}`
                    },
                    body: JSON.stringify({ phone })
                });
                const data = await response.json();
                if (data.phone_code_hash) {
                    document.getElementById('telegram-phone').style.display = 'none';
                    document.getElementById('telegram-code').style.display = 'block';
                    document.getElementById('telegram-password').style.display = 'block';
                    this.phoneCodeHash = data.phone_code_hash;
                } else {
                    alert(data.error || 'Failed to request code');
                }
            } else {
                const response = await fetch('/api/telegram/verify_code', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.token}`
                    },
                    body: JSON.stringify({ phone, code, phone_code_hash: this.phoneCodeHash, password })
                });
                const data = await response.json();
                if (data.status === 'telegram_connected') {
                    alert('Telegram connected successfully');
                    document.getElementById('telegram-modal').style.display = 'none';
                    this.loadTelegramStatus();
                    this.checkOnboarding();
                } else {
                    alert(data.error || 'Telegram connection failed');
                }
            }
        } catch (e) {
            alert('Failed to connect Telegram: ' + e.message);
        }
    }

    async searchChannels() {
        const query = document.getElementById('channel-search').value;
        if (!query.trim()) return alert('Please enter a search query');
        try {
            const response = await fetch(`/api/channels/search?query=${encodeURIComponent(query)}`, {
                headers: { 'Authorization': `Bearer ${this.token}` }
            });
            const data = await response.json();
            const tbody = document.getElementById('channels-table').querySelector('tbody');
            tbody.innerHTML = '';
            data.channels.forEach(channel => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${channel.title}</td>
                    <td>Not Joined</td>
                    <td><button class="join-channel" data-username="${channel.username}">Join</button></td>
                `;
                tbody.appendChild(row);
            });
        } catch (e) {
            alert('Failed to search channels: ' + e.message);
        }
    }

    async joinChannel(username) {
        if (!username) return alert('Invalid channel username');
        try {
            const response = await fetch('/api/channels/join', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({ channel_username: username })
            });
            const data = await response.json();
            if (data.status === 'channel_joined') {
                alert('Channel joined successfully');
                this.loadChannels();
                this.checkOnboarding();
            } else {
                alert(data.error || 'Failed to join channel');
            }
        } catch (e) {
            alert('Failed to join channel: ' + e.message);
        }
    }

    async loadChannels() {
        try {
            const response = await fetch('/api/channels/search?query=', {
                headers: { 'Authorization': `Bearer ${this.token}` }
            });
            const data = await response.json();
            const tbody = document.getElementById('channels-table').querySelector('tbody');
            tbody.innerHTML = '';
            data.channels.forEach(channel => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${channel.title}</td>
                    <td>Active</td>
                    <td>
                        <button class="toggle-channel" data-id="${channel.username}">Toggle</button>
                        <button class="channel-settings" data-id="${channel.username}">Settings</button>
                    </td>
                `;
                tbody.appendChild(row);
            });
        } catch (e) {
            alert('Failed to load channels: ' + e.message);
        }
    }

    showChannelSettings(id) {
        if (!id) return alert('Invalid channel ID');
        document.getElementById('channel-settings-modal').style.display = 'block';
        this.currentChannelId = id;
    }

    async saveChannelSettings() {
        const useAiParsing = document.getElementById('use-ai-parsing').checked;
        const filterKeywords = document.getElementById('filter-keywords').value.split(',').map(k => k.trim()).filter(k => k);
        try {
            const response = await fetch('/api/channels/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({
                    channel_id: this.currentChannelId,
                    use_ai_parsing: useAiParsing,
                    filter_keywords: filterKeywords
                })
            });
            const data = await response.json();
            if (data.status) {
                alert('Channel settings saved');
                document.getElementById('channel-settings-modal').style.display = 'none';
                this.loadChannels();
            } else {
                alert(data.error || 'Failed to save channel settings');
            }
        } catch (e) {
            alert('Failed to save channel settings: ' + e.message);
        }
    }

    async toggleChannel(id) {
        if (!id) return alert('Invalid channel ID');
        try {
            const response = await fetch('/api/channels/toggle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({ channel_id: id })
            });
            const data = await response.json();
            if (data.status) {
                this.loadChannels();
            } else {
                alert(data.error || 'Failed to toggle channel');
            }
        } catch (e) {
            alert('Failed to toggle channel: ' + e.message);
        }
    }

    async loadSettings() {
        try {
            const response = await fetch('/api/copier/settings', {
                headers: { 'Authorization': `Bearer ${this.token}` }
            });
            const data = await response.json();
            document.getElementById('auto-copy').checked = data.auto_copy || false;
            document.getElementById('risk-per-trade').value = data.risk_per_trade || '';
            document.getElementById('max-daily-loss').value = data.max_daily_loss || '';
            document.getElementById('max-open-trades').value = data.max_open_trades || '';
        } catch (e) {
            alert('Failed to load settings: ' + e.message);
        }
    }

    async saveSettings() {
        const settings = {
            auto_copy: document.getElementById('auto-copy').checked,
            risk_per_trade: parseFloat(document.getElementById('risk-per-trade').value) || 0,
            max_daily_loss: parseFloat(document.getElementById('max-daily-loss').value) || 0,
            max_open_trades: parseInt(document.getElementById('max-open-trades').value) || 0
        };
        if (settings.risk_per_trade < 0 || settings.max_daily_loss < 0 || settings.max_open_trades < 0) {
            return alert('Settings values cannot be negative');
        }
        try {
            const response = await fetch('/api/copier/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify(settings)
            });
            const data = await response.json();
            if (data.status === 'settings_updated') {
                alert('Settings saved successfully');
                this.checkOnboarding();
            } else {
                alert(data.error || 'Failed to save settings');
            }
        } catch (e) {
            alert('Failed to save settings: ' + e.message);
        }
    }

    showMTModal() {
        document.getElementById('add-mt-modal').style.display = 'block';
    }

    async addMTAccount() {
        const accountId = document.getElementById('mt-account-id').value;
        const login = document.getElementById('mt-login').value;
        const password = document.getElementById('mt-password').value;
        const server = document.getElementById('mt-server').value;
        if (!accountId || !login || !password || !server) {
            alert('Please enter all fields');
            return;
        }
        if (!/^\d+$/.test(login)) return alert('Login must be numeric');
        try {
            const response = await fetch('/api/add_mt_account', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({ account_id: accountId, login, password, server })
            });
            const data = await response.json();
            if (data.status) {
                alert('MT account added successfully');
                document.getElementById('add-mt-modal').style.display = 'none';
                this.loadMTAccounts();
                this.checkOnboarding();
            } else {
                alert(data.error || 'Failed to add MT account');
            }
        } catch (e) {
            alert('Failed to add MT account: ' + e.message);
        }
    }

    async loadMTAccounts() {
        try {
            const response = await fetch('/api/mt/accounts', {
                headers: { 'Authorization': `Bearer ${this.token}` }
            });
            const data = await response.json();
            const tbody = document.getElementById('mt-accounts-table').querySelector('tbody');
            tbody.innerHTML = '';
            data.accounts.forEach(account => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${account.account_id}</td>
                    <td>${account.login}</td>
                    <td>${account.server}</td>
                    <td>${account.balance}</td>
                    <td>${account.status}</td>
                `;
                tbody.appendChild(row);
            });
        } catch (e) {
            alert('Failed to load MT accounts: ' + e.message);
        }
    }

    handleNewSignal(data) {
        if (data.user_id !== this.userId) return;
        const signal = data.signal;
        const tbody = document.getElementById('signals-table').querySelector('tbody');
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${signal.symbol}</td>
            <td>${signal.order_type}</td>
            <td>${signal.entry_price}</td>
            <td>${signal.stop_loss}</td>
            <td>${signal.take_profits.join(', ')}</td>
            <td><button class="copy-signal" data-signal-id="${data.signal_id}">Copy</button></td>
        `;
        tbody.appendChild(row);
        this.notifySignal(signal);
    }

    async copySignal(signalId) {
        if (!signalId) return alert('Invalid signal ID');
        try {
            const response = await fetch('/api/copy_signal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({ signal_id: signalId })
            });
            const data = await response.json();
            if (data.status) {
                alert('Signal copied successfully');
                this.notifySignalCopied(data.result);
            } else {
                alert(data.error || 'Failed to copy signal');
            }
        } catch (e) {
            alert('Failed to copy signal: ' + e.message);
        }
    }

    handleError(data) {
        if (data.user_id === this.userId) {
            alert('Error: ' + data.message);
        }
    }

    handleReauthNeeded(data) {
        if (data.user_id === this.userId) {
            alert('Telegram re-authentication needed: ' + data.message);
            this.showTelegramModal();
        }
    }

    async checkOnboarding() {
        try {
            const response = await fetch('/api/onboarding/status', {
                headers: { 'Authorization': `Bearer ${this.token}` }
            });
            const data = await response.json();
            const stepsList = document.getElementById('onboarding-steps');
            stepsList.innerHTML = '';
            data.steps.forEach(step => {
                const li = document.createElement('li');
                li.textContent = `${step.step}: ${step.completed ? 'Completed' : 'Pending'}`;
                stepsList.appendChild(li);
            });
            document.getElementById('onboarding-section').style.display = data.current_step === 'complete' ? 'none' : 'block';
        } catch (e) {
            alert('Failed to load onboarding status: ' + e.message);
        }
    }

    checkServiceWorker() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').then(reg => {
                console.log('Service Worker registered', reg);
            }).catch(err => {
                console.error('Service Worker registration failed', err);
            });
        }
    }

    notifySignal(signal) {
        if (Notification.permission === 'granted') {
            new Notification('New Signal', {
                body: `${signal.order_type} ${signal.symbol} at ${signal.entry_price}`,
                data: { signalId: signal.signal_id }
            });
        } else if (Notification.permission !== 'denied') {
            Notification.requestPermission().then(permission => {
                if (permission === 'granted') {
                    new Notification('New Signal', {
                        body: `${signal.order_type} ${signal.symbol} at ${signal.entry_price}`,
                        data: { signalId: signal.signal_id }
                    });
                }
            });
        }
    }

    notifySignalCopied(result) {
        if (Notification.permission === 'granted') {
            new Notification('Signal Copied', {
                body: `Trade executed: ${result}`,
            });
        } else if (Notification.permission !== 'denied') {
            Notification.requestPermission().then(permission => {
                if (permission === 'granted') {
                    new Notification('Signal Copied', {
                        body: `Trade executed: ${result}`,
                    });
                }
            });
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const app = new FXSignalCopier();
});
