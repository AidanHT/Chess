'use strict';

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

let ws = null;
let pieces = {};
let legalMoves = [];
let lastMoveUci = null;
let checkSquare = null;
let gameOver = null;
let isPlayerTurn = false;
let playerColor = 'white';
let turn = 'white';
let moveHistory = [];
let captured = { white: [], black: [] };

// UI
let selectedSquare = null;
let flipped = false;
let dragInfo = null;

// Settings
const settings = {
    strength: 800,
    color: 'white',
    theme: 'green',
    sound: true,
    showLegal: true,
    showCoords: true,
    showAnalysis: true,
};

let audioCtx = null;

// ═══════════════════════════════════════════════════════════════════
// WebSocket
// ═══════════════════════════════════════════════════════════════════

function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => updateStatus('Connected');
    ws.onclose = () => {
        updateStatus('Disconnected \u2014 reconnecting...');
        setTimeout(connect, 2000);
    };
    ws.onerror = () => {};
    ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
}

function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
    }
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'state':
            updateGameState(msg);
            break;
        case 'thinking_start':
            showThinking(true);
            break;
        case 'thinking':
            updateThinkingProgress(msg);
            break;
        case 'engine_move':
            showThinking(false);
            if (msg.analysis) updateAnalysis(msg.analysis);
            playMoveSound(msg.san);
            break;
        case 'game_over':
            showGameOverDialog(msg.result, msg.reason);
            break;
        case 'error':
            console.warn('Server:', msg.message);
            break;
    }
}

function updateGameState(state) {
    pieces = parseFEN(state.fen);
    turn = state.turn;
    legalMoves = state.legal_moves || [];
    lastMoveUci = state.last_move;
    checkSquare = state.check;
    gameOver = state.game_over;
    moveHistory = state.move_history || [];
    captured = state.captured || { white: [], black: [] };
    playerColor = state.player_color;
    isPlayerTurn = state.is_player_turn;

    selectedSquare = null;

    // Auto-flip on new game
    if (moveHistory.length === 0) {
        flipped = playerColor === 'black';
    }

    renderBoard();
    renderMoveHistory();
    renderCaptured();

    if (gameOver) {
        updateStatus(`${gameOver.reason} \u2014 ${gameOver.result}`);
        showGameOverDialog(gameOver.result, gameOver.reason);
    } else {
        updateStatus(isPlayerTurn ? 'Your turn' : 'Engine thinking...');
    }
}

// ═══════════════════════════════════════════════════════════════════
// FEN Parsing
// ═══════════════════════════════════════════════════════════════════

function parseFEN(fen) {
    const result = {};
    const ranks = fen.split(' ')[0].split('/');
    for (let r = 0; r < 8; r++) {
        let file = 0;
        for (const ch of ranks[r]) {
            if (ch >= '1' && ch <= '8') {
                file += parseInt(ch);
            } else {
                const sq = 'abcdefgh'[file] + (8 - r);
                result[sq] = {
                    color: ch === ch.toUpperCase() ? 'w' : 'b',
                    type: ch.toLowerCase(),
                };
                file++;
            }
        }
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════
// Board Rendering
// ═══════════════════════════════════════════════════════════════════

function renderBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';

    const lastFrom = lastMoveUci ? lastMoveUci.slice(0, 2) : null;
    const lastTo = lastMoveUci ? lastMoveUci.slice(2, 4) : null;
    const selectedMoves = selectedSquare ? getLegalMovesFrom(selectedSquare) : [];
    const selectedTargets = new Set(selectedMoves.map(m => m.slice(2, 4)));

    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const rank = flipped ? row + 1 : 8 - row;
            const file = flipped ? 7 - col : col;
            const sq = 'abcdefgh'[file] + rank;
            const isLight = (rank + file) % 2 === 1;

            const cell = document.createElement('div');
            cell.className = 'square ' + (isLight ? 'light' : 'dark');
            cell.dataset.square = sq;

            if (sq === lastFrom || sq === lastTo) cell.classList.add('last-move');
            if (sq === checkSquare) cell.classList.add('check');
            if (sq === selectedSquare) cell.classList.add('selected');

            // Piece
            const piece = pieces[sq];
            if (piece) {
                const img = document.createElement('img');
                img.src = `/pieces/${piece.color}/${piece.type}`;
                img.className = 'piece';
                img.draggable = false;
                cell.appendChild(img);
            }

            // Legal move hints
            if (settings.showLegal && selectedSquare && selectedTargets.has(sq)) {
                const hint = document.createElement('div');
                hint.className = piece ? 'capture-hint' : 'move-hint';
                cell.appendChild(hint);
            }

            // Coordinates
            if (settings.showCoords) {
                if (col === 0) {
                    const lbl = document.createElement('span');
                    lbl.className = 'rank-label';
                    lbl.textContent = rank;
                    cell.appendChild(lbl);
                }
                if (row === 7) {
                    const lbl = document.createElement('span');
                    lbl.className = 'file-label';
                    lbl.textContent = 'abcdefgh'[file];
                    cell.appendChild(lbl);
                }
            }

            cell.addEventListener('mousedown', (e) => onMouseDown(e, sq));
            board.appendChild(cell);
        }
    }
}

function getLegalMovesFrom(from) {
    return legalMoves.filter(m => m.slice(0, 2) === from);
}

// ═══════════════════════════════════════════════════════════════════
// Interaction — click-to-move + drag-and-drop
// ═══════════════════════════════════════════════════════════════════

function onMouseDown(e, sq) {
    if (!isPlayerTurn || gameOver) return;
    e.preventDefault();

    const piece = pieces[sq];
    const myColor = playerColor === 'white' ? 'w' : 'b';
    const isOwn = piece && piece.color === myColor;

    if (selectedSquare) {
        if (sq === selectedSquare) {
            selectedSquare = null;
            renderBoard();
            return;
        }
        if (isOwn) {
            selectedSquare = sq;
            renderBoard();
            startDrag(e, sq);
            return;
        }
        tryMove(selectedSquare, sq);
        return;
    }

    if (isOwn) {
        selectedSquare = sq;
        renderBoard();
        startDrag(e, sq);
    }
}

function startDrag(e, sq) {
    const piece = pieces[sq];
    if (!piece) return;

    const ghost = document.createElement('img');
    ghost.src = `/pieces/${piece.color}/${piece.type}`;
    ghost.className = 'drag-ghost';
    document.body.appendChild(ghost);
    ghost.style.left = e.clientX + 'px';
    ghost.style.top = e.clientY + 'px';

    // Dim original
    const cell = document.querySelector(`[data-square="${sq}"]`);
    const origImg = cell ? cell.querySelector('.piece') : null;
    if (origImg) origImg.style.opacity = '0.3';

    const onMove = (ev) => {
        ghost.style.left = ev.clientX + 'px';
        ghost.style.top = ev.clientY + 'px';
    };

    const onUp = (ev) => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);

        ghost.style.display = 'none';
        const el = document.elementFromPoint(ev.clientX, ev.clientY);
        ghost.style.display = '';
        document.body.removeChild(ghost);

        if (origImg) origImg.style.opacity = '1';

        const target = el ? el.closest('.square') : null;
        const targetSq = target ? target.dataset.square : null;

        if (targetSq && targetSq !== sq) {
            tryMove(sq, targetSq);
        }
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
}

function tryMove(from, to) {
    const moves = getLegalMovesFrom(from);
    const matching = moves.filter(m => m.slice(2, 4) === to);

    if (matching.length === 0) {
        selectedSquare = null;
        renderBoard();
        return;
    }

    // Promotion?
    const promotions = matching.filter(m => m.length === 5);
    if (promotions.length > 0) {
        showPromotionDialog(from, to, promotions);
        return;
    }

    sendMove(matching[0]);
}

function sendMove(uci) {
    const to = uci.slice(2, 4);
    const isCapture = pieces[to] != null;

    selectedSquare = null;
    send({ type: 'move', uci: uci });
    playSound(isCapture ? 'capture' : 'move');
}

// ═══════════════════════════════════════════════════════════════════
// Promotion Dialog
// ═══════════════════════════════════════════════════════════════════

function showPromotionDialog(from, to, moves) {
    const container = document.getElementById('promotion-pieces');
    container.innerHTML = '';

    const color = playerColor === 'white' ? 'w' : 'b';

    for (const type of ['q', 'r', 'b', 'n']) {
        const uci = from + to + type;
        if (!moves.includes(uci)) continue;

        const img = document.createElement('img');
        img.src = `/pieces/${color}/${type}`;
        img.className = 'promo-piece';
        img.addEventListener('click', () => {
            hideModal('promotion-modal');
            sendMove(uci);
        });
        container.appendChild(img);
    }

    showModal('promotion-modal');
}

// ═══════════════════════════════════════════════════════════════════
// Analysis Panel
// ═══════════════════════════════════════════════════════════════════

function showThinking(active) {
    const el = document.getElementById('thinking-indicator');
    el.classList.toggle('hidden', !active);
    if (active) {
        document.getElementById('top-moves').innerHTML = '';
        document.getElementById('search-stats').innerHTML = '';
    }
}

function updateThinkingProgress(data) {
    const text = document.getElementById('thinking-text');
    const pct = data.total > 0 ? Math.round(data.sims / data.total * 100) : 0;
    text.textContent = `Thinking... ${pct}% (${data.elapsed}s)`;
}

function updateAnalysis(data) {
    if (!settings.showAnalysis) return;

    // Eval bar
    const evalW = data.eval || 0;
    const pct = Math.min(96, Math.max(4, (evalW + 1) / 2 * 100));
    document.getElementById('eval-white').style.height = pct + '%';
    document.getElementById('eval-score').textContent =
        (evalW >= 0 ? '+' : '') + evalW.toFixed(2);

    // Top moves
    const container = document.getElementById('top-moves');
    container.innerHTML = '';

    if (data.top_moves && data.top_moves.length > 0) {
        const maxV = data.top_moves[0].visits || 1;

        data.top_moves.forEach((mv, i) => {
            const row = document.createElement('div');
            row.className = 'top-move-row';
            const barPct = Math.round(mv.visits / maxV * 100);
            row.innerHTML =
                `<span class="top-move-rank">${i + 1}.</span>` +
                `<span class="top-move-san">${mv.san}</span>` +
                `<div class="top-move-bar"><div class="top-move-bar-fill" style="width:${barPct}%"></div></div>` +
                `<span class="top-move-winrate">${mv.winrate}%</span>`;
            container.appendChild(row);
        });
    }

    // Stats
    document.getElementById('search-stats').textContent =
        `${data.sims} simulations \u00B7 ${data.elapsed}s`;
}

// ═══════════════════════════════════════════════════════════════════
// Move History
// ═══════════════════════════════════════════════════════════════════

function renderMoveHistory() {
    const container = document.getElementById('moves-list');
    container.innerHTML = '';

    for (const entry of moveHistory) {
        const row = document.createElement('div');
        row.className = 'move-row';
        row.innerHTML =
            `<span class="move-num">${entry.num}.</span>` +
            `<span class="move-white">${entry.white || ''}</span>` +
            `<span class="move-black">${entry.black || ''}</span>`;
        container.appendChild(row);
    }

    container.scrollTop = container.scrollHeight;
}

// ═══════════════════════════════════════════════════════════════════
// Captured Pieces
// ═══════════════════════════════════════════════════════════════════

const PIECE_CHAR = {
    P: '\u2659', N: '\u2658', B: '\u2657', R: '\u2656', Q: '\u2655',
    p: '\u265F', n: '\u265E', b: '\u265D', r: '\u265C', q: '\u265B',
};

const PIECE_VAL = { p: 1, n: 3, b: 3, r: 5, q: 9, P: 1, N: 3, B: 3, R: 5, Q: 9 };

function renderCaptured() {
    const bottomIsWhite = !flipped;

    // Pieces captured BY white shown next to White's name; similarly for Black
    const topPieces = bottomIsWhite ? captured.black : captured.white;
    const botPieces = bottomIsWhite ? captured.white : captured.black;

    document.getElementById('top-captured').textContent =
        topPieces.map(p => PIECE_CHAR[p] || '').join('');
    document.getElementById('bottom-captured').textContent =
        botPieces.map(p => PIECE_CHAR[p] || '').join('');

    // Material advantage
    const wVal = captured.white.reduce((s, p) => s + (PIECE_VAL[p] || 0), 0);
    const bVal = captured.black.reduce((s, p) => s + (PIECE_VAL[p] || 0), 0);
    const diff = wVal - bVal;

    const topAdv = document.getElementById('top-advantage');
    const botAdv = document.getElementById('bottom-advantage');
    topAdv.textContent = '';
    botAdv.textContent = '';

    if (diff > 0) {
        (bottomIsWhite ? botAdv : topAdv).textContent = `+${diff}`;
    } else if (diff < 0) {
        (bottomIsWhite ? topAdv : botAdv).textContent = `+${-diff}`;
    }

    // Player names
    const topColor = bottomIsWhite ? 'Black' : 'White';
    const botColor = bottomIsWhite ? 'White' : 'Black';
    const topIsEngine =
        (bottomIsWhite && playerColor === 'white') ||
        (!bottomIsWhite && playerColor === 'black');

    document.getElementById('top-name').textContent =
        topIsEngine ? `Engine (${topColor})` : `You (${topColor})`;
    document.getElementById('bottom-name').textContent =
        topIsEngine ? `You (${botColor})` : `Engine (${botColor})`;
}

// ═══════════════════════════════════════════════════════════════════
// Sound Effects (Web Audio API — no files needed)
// ═══════════════════════════════════════════════════════════════════

function getAudio() {
    if (!audioCtx) {
        const C = window.AudioContext || window.webkitAudioContext;
        if (C) audioCtx = new C();
    }
    return audioCtx;
}

function playSound(type) {
    if (!settings.sound) return;
    const ctx = getAudio();
    if (!ctx) return;
    if (ctx.state === 'suspended') ctx.resume();

    const now = ctx.currentTime;

    if (type === 'move') {
        noiseBurst(ctx, 0.03, 0.15, 4);
    } else if (type === 'capture') {
        noiseBurst(ctx, 0.06, 0.3, 2.5);
    } else if (type === 'check') {
        const osc = ctx.createOscillator();
        osc.frequency.value = 880;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0.08, now);
        g.gain.exponentialRampToValueAtTime(0.001, now + 0.12);
        osc.connect(g).connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.12);
    } else if (type === 'gameEnd') {
        [523, 659, 784].forEach((f, i) => {
            const osc = ctx.createOscillator();
            osc.frequency.value = f;
            const g = ctx.createGain();
            g.gain.setValueAtTime(0.04, now + i * 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, now + i * 0.1 + 0.5);
            osc.connect(g).connect(ctx.destination);
            osc.start(now + i * 0.1);
            osc.stop(now + i * 0.1 + 0.5);
        });
    }
}

function noiseBurst(ctx, duration, volume, decay) {
    const len = Math.floor(ctx.sampleRate * duration);
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) {
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay);
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const g = ctx.createGain();
    g.gain.value = volume;
    src.connect(g).connect(ctx.destination);
    src.start();
}

function playMoveSound(san) {
    if (!san) return;
    if (san.includes('#')) playSound('gameEnd');
    else if (san.includes('+')) playSound('check');
    else if (san.includes('x')) playSound('capture');
    else playSound('move');
}

// ═══════════════════════════════════════════════════════════════════
// Status Bar
// ═══════════════════════════════════════════════════════════════════

function updateStatus(text) {
    document.getElementById('status-text').textContent = text;
}

// ═══════════════════════════════════════════════════════════════════
// Modals
// ═══════════════════════════════════════════════════════════════════

function showModal(id) { document.getElementById(id).classList.add('active'); }
function hideModal(id) { document.getElementById(id).classList.remove('active'); }

function showGameOverDialog(result, reason) {
    document.getElementById('game-over-title').textContent = reason;
    document.getElementById('game-over-detail').textContent = result;
    playSound('gameEnd');
    showModal('game-over-modal');
}

// ═══════════════════════════════════════════════════════════════════
// Button Group Helper
// ═══════════════════════════════════════════════════════════════════

function setupBtnGroup(groupId, cb) {
    const grp = document.getElementById(groupId);
    if (!grp) return;
    grp.addEventListener('click', (e) => {
        const btn = e.target.closest('.option-btn');
        if (!btn) return;
        grp.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        if (cb) cb(btn.dataset.value);
    });
}

// ═══════════════════════════════════════════════════════════════════
// Keyboard Shortcuts
// ═══════════════════════════════════════════════════════════════════

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
        if (selectedSquare) { selectedSquare = null; renderBoard(); }
    }
    if (e.key === 'f' && !e.ctrlKey && !e.metaKey && !e.target.closest('input')) {
        flipped = !flipped;
        renderBoard();
        renderCaptured();
    }
});

// ═══════════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    connect();

    // Button groups
    setupBtnGroup('color-select', (v) => settings.color = v);
    setupBtnGroup('strength-select', (v) => settings.strength = parseInt(v));
    setupBtnGroup('theme-select', (v) => {
        settings.theme = v;
        document.body.dataset.theme = v;
    });

    // New Game
    document.getElementById('new-game-btn').addEventListener('click', () => showModal('new-game-modal'));
    document.getElementById('start-game-btn').addEventListener('click', () => {
        hideModal('new-game-modal');
        send({ type: 'new_game', color: settings.color, strength: settings.strength });
    });

    // Controls
    document.getElementById('resign-btn').addEventListener('click', () => {
        if (confirm('Are you sure you want to resign?')) send({ type: 'resign' });
    });
    document.getElementById('undo-btn').addEventListener('click', () => send({ type: 'undo' }));
    document.getElementById('flip-btn').addEventListener('click', () => {
        flipped = !flipped;
        renderBoard();
        renderCaptured();
    });

    // Settings
    document.getElementById('settings-btn').addEventListener('click', () => showModal('settings-modal'));
    document.getElementById('close-settings').addEventListener('click', () => hideModal('settings-modal'));

    document.getElementById('sound-toggle').addEventListener('change', (e) => {
        settings.sound = e.target.checked;
    });
    document.getElementById('legal-toggle').addEventListener('change', (e) => {
        settings.showLegal = e.target.checked;
        renderBoard();
    });
    document.getElementById('coords-toggle').addEventListener('change', (e) => {
        settings.showCoords = e.target.checked;
        renderBoard();
    });
    document.getElementById('analysis-toggle').addEventListener('change', (e) => {
        settings.showAnalysis = e.target.checked;
        document.getElementById('analysis').classList.toggle('hidden', !e.target.checked);
        document.getElementById('eval-container').classList.toggle('hidden', !e.target.checked);
    });

    // Game over -> new game
    document.getElementById('game-over-new').addEventListener('click', () => {
        hideModal('game-over-modal');
        showModal('new-game-modal');
    });

    // Close modals on overlay click
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.classList.remove('active');
        });
    });

    // Show new game dialog on first load
    showModal('new-game-modal');
});
