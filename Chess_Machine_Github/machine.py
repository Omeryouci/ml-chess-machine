# -*- coding: utf-8 -*-
# Satranc Machine v5.2
# Kendini gelistiren satranc makinesi
#
# Calistirma:  pip install chess numpy pygame-ce   (veya pygame)
#              python satranc_machine.py
# Test:        python satranc_machine.py --test
# Ekran goruntusu: python satranc_machine.py --shots klasor

import chess, chess.polyglot, pygame, numpy as np
import json, os, time, threading, sys, random, math
from collections import deque
from datetime import datetime

# ===================== AYARLAR =====================
WIDTH, HEIGHT   = 1100, 720
BOARD_SIZE      = 640
SQUARE          = 80
EVAL_X, EVAL_W  = 646, 18
PANEL_X         = 672
PANEL_W         = WIDTH - PANEL_X
BOTTOM_Y        = 640
BOTTOM_H        = HEIGHT - BOTTOM_Y

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(DATA_DIR, "machine_v5.npz")
STATS_PATH = os.path.join(DATA_DIR, "machine_v5_stats.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history_v5.json")

PIECE_VALUES = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330,
                chess.ROOK:500, chess.QUEEN:900, chess.KING:0}

# konum tablolari (beyaz bakis acisi, a8 ilk sirada)
PST_PAWN = (
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0)
PST_KNIGHT = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50)
PST_BISHOP = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20)
PST_ROOK = (
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0)
PST_QUEEN = (
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20)
PST_KING_MID = (
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20)
PST_KING_END = (
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -50, -50)
PST = {chess.PAWN: PST_PAWN, chess.KNIGHT: PST_KNIGHT, chess.BISHOP: PST_BISHOP,
       chess.ROOK: PST_ROOK, chess.QUEEN: PST_QUEEN}


def classical_eval(board):
    """Materyal + konum tablosu ile degerlendirme (beyaz bakisi, cp)."""
    pieces = board.piece_map()
    np_material = sum(PIECE_VALUES[p.piece_type] for p in pieces.values()
                      if p.piece_type not in (chess.PAWN, chess.KING))
    endgame = np_material <= 1600
    score = 0
    for sq, p in pieces.items():
        t = PST_KING_END if (p.piece_type == chess.KING and endgame) \
            else PST.get(p.piece_type, PST_PAWN)
        if p.color == chess.WHITE:
            score += PIECE_VALUES[p.piece_type] + t[sq ^ 56]
        else:
            score -= PIECE_VALUES[p.piece_type] + t[sq]
    return float(score)


PIECE_SYMBOLS = {chess.PAWN:'♟', chess.KNIGHT:'♞', chess.BISHOP:'♝',
                 chess.ROOK:'♜', chess.QUEEN:'♛', chess.KING:'♚'}
PIECE_NAMES = {chess.PAWN: 'Piyon', chess.KNIGHT: 'At', chess.BISHOP: 'Fil',
               chess.ROOK: 'Kale', chess.QUEEN: 'Vezir', chess.KING: 'Sah'}

# Seviye esikleri
MACHINE_LEVELS = [
    (0,       "Ham Makine"),
    (300,     "Ilk Sinyaller"),
    (1500,    "Kalibre Ediliyor"),
    (4000,    "Ogrenen Makine"),
    (10000,   "Deneyimli Makine"),
    (25000,   "Gelismis Model"),
    (60000,   "Guclu Motor"),
    (150000,  "Ustalasan Makine"),
    (400000,  "Elit Model"),
    (1000000, "Super Makine"),
]

# Renkler
C_CRUST   = (16, 16, 26)
C_MANTLE  = (24, 24, 37)
C_SURFACE = (30, 30, 46)
C_CARD    = (36, 37, 54)
C_CARD_HI = (49, 50, 68)
C_BORDER  = (64, 66, 90)
C_SHADOW  = (8, 8, 16)
C_TEXT    = (205, 214, 244)
C_MUTED   = (166, 173, 200)
C_FAINT   = (117, 122, 148)
C_ACCENT  = (137, 180, 250)
C_ACCENT2 = (203, 166, 247)
C_GREEN   = (166, 227, 161)
C_YELLOW  = (249, 226, 175)
C_RED     = (243, 139, 168)
C_TEAL    = (148, 226, 213)
C_HINT    = (178, 183, 208)
C_GREEN_L = (185, 235, 180)
C_RED_L   = (250, 165, 190)

SQ_LIGHT  = (233, 218, 190)
SQ_DARK   = (168, 138, 108)


def lerp(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def machine_level_info(steps):
    name, idx = MACHINE_LEVELS[0][1], 0
    for i, (thr, nm) in enumerate(MACHINE_LEVELS):
        if steps >= thr:
            name, idx = nm, i
    nxt = MACHINE_LEVELS[idx + 1][0] if idx + 1 < len(MACHINE_LEVELS) else None
    return name, idx, nxt


# ===================== SINIR AGI =====================
class NeuralNetwork:
    """773 -> 256 -> 64 -> 1, ReLU + tanh, Adam."""
    N_IN, H1, H2 = 773, 256, 64

    def __init__(self):
        self.W1 = (np.random.randn(self.N_IN, self.H1) * np.sqrt(2.0 / self.N_IN)).astype(np.float32)
        self.b1 = np.zeros(self.H1, dtype=np.float32)
        self.W2 = (np.random.randn(self.H1, self.H2) * np.sqrt(2.0 / self.H1)).astype(np.float32)
        self.b2 = np.zeros(self.H2, dtype=np.float32)
        self.W3 = (np.random.randn(self.H2, 1) * np.sqrt(2.0 / self.H2)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)
        self.t = 0
        self.m = {k: np.zeros_like(getattr(self, k))
                  for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')}
        self.v = {k: np.zeros_like(getattr(self, k))
                  for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')}
        self.training_steps = 0
        self.loss_history = []

    def features(self, board):
        x = np.zeros(self.N_IN, dtype=np.float32)
        for sq, p in board.piece_map().items():
            idx = (0 if p.color == chess.WHITE else 6) + (p.piece_type - 1)
            x[idx * 64 + sq] = 1.0
        x[768] = 1.0 if board.turn == chess.WHITE else -1.0
        x[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        x[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        x[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        x[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        return x

    def forward(self, board):
        # sparse forward: sadece dolu karelerin W1 satirlari toplanir
        idx = [((0 if p.color == chess.WHITE else 6) + (p.piece_type - 1)) * 64 + sq
               for sq, p in board.piece_map().items()]
        h = (self.b1 + self.W1[idx].sum(axis=0)) if idx else self.b1.copy()
        if board.turn == chess.WHITE:
            h = h + self.W1[768]
        else:
            h = h - self.W1[768]
        if board.has_kingside_castling_rights(chess.WHITE):
            h = h + self.W1[769]
        if board.has_queenside_castling_rights(chess.WHITE):
            h = h + self.W1[770]
        if board.has_kingside_castling_rights(chess.BLACK):
            h = h + self.W1[771]
        if board.has_queenside_castling_rights(chess.BLACK):
            h = h + self.W1[772]
        a1 = np.maximum(0.0, h)
        a2 = np.maximum(0.0, a1 @ self.W2 + self.b2)
        return float(np.tanh(a2 @ self.W3 + self.b3)[0]) * 1000.0

    def train_batch(self, X, T, lr=0.002):
        N = X.shape[0]
        y_t = (T.astype(np.float32) / 1000.0).reshape(-1, 1)

        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0.0, Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = np.maximum(0.0, Z2)
        Z3 = A2 @ self.W3 + self.b3
        Y = np.tanh(Z3)

        loss = float(np.mean((Y - y_t) ** 2))

        dZ3 = (2.0 / N) * (Y - y_t) * (1.0 - Y * Y)
        dW3 = A2.T @ dZ3;              db3 = dZ3.sum(0)
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * (Z2 > 0)
        dW2 = A1.T @ dZ2;              db2 = dZ2.sum(0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1;               db1 = dZ1.sum(0)

        self.t += 1
        b1e, b2e, eps = 0.9, 0.999, 1e-8
        for name, g in (('W1', dW1), ('b1', db1), ('W2', dW2),
                        ('b2', db2), ('W3', dW3), ('b3', db3)):
            p = getattr(self, name)
            self.m[name] = b1e * self.m[name] + (1 - b1e) * g
            self.v[name] = b2e * self.v[name] + (1 - b2e) * (g * g)
            mhat = self.m[name] / (1 - b1e ** self.t)
            vhat = self.v[name] / (1 - b2e ** self.t)
            p -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)

        self.training_steps += 1
        return loss

    def save(self, path):
        np.savez_compressed(
            path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            mW1=self.m['W1'], mb1=self.m['b1'], mW2=self.m['W2'],
            mb2=self.m['b2'], mW3=self.m['W3'], mb3=self.m['b3'],
            vW1=self.v['W1'], vb1=self.v['b1'], vW2=self.v['W2'],
            vb2=self.v['b2'], vW3=self.v['W3'], vb3=self.v['b3'],
            t=self.t, training_steps=self.training_steps)

    @classmethod
    def load(cls, path):
        nn = cls()
        if os.path.exists(path):
            try:
                d = np.load(path)
                for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                    setattr(nn, k, d[k])
                for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                    nn.m[k] = d['m' + k]; nn.v[k] = d['v' + k]
                nn.t = int(d['t'])
                nn.training_steps = int(d['training_steps'])
            except Exception:
                pass
        return nn


# ===================== BEYIN =====================
NN_TRUST_STEPS = 25000


class Brain:
    def __init__(self, model_path=MODEL_PATH, stats_path=STATS_PATH):
        self.model_path = model_path
        self.stats_path = stats_path
        self.nn = NeuralNetwork()
        self.buffer = deque(maxlen=30000)
        self.games_played = 0
        self.wins = self.losses = self.draws = 0
        self.games_selfplay = 0

    def nn_weight(self):
        return min(1.0, self.nn.training_steps / NN_TRUST_STEPS)

    def eval_white(self, board):
        w = self.nn_weight()
        if w <= 0.0:
            return classical_eval(board)
        if w >= 1.0:
            return self.nn.forward(board)
        return (1.0 - w) * classical_eval(board) + w * self.nn.forward(board)

    def eval_stm(self, board):
        return self.eval_white(board) * (1.0 if board.turn == chess.WHITE else -1.0)

    def add_sample(self, fen, target):
        self.buffer.append((fen, float(max(-1000.0, min(1000.0, target)))))

    def train_from_buffer(self, steps, batch=64, lr=0.002):
        if len(self.buffer) < 16:
            return 0.0
        n = int(min(steps, max(8, len(self.buffer) // 4)))
        losses = []
        for _ in range(n):
            sample = random.sample(self.buffer, min(batch, len(self.buffer)))
            X = np.stack([self.nn.features(chess.Board(f)) for f, _ in sample])
            T = np.array([t for _, t in sample], dtype=np.float32)
            losses.append(self.nn.train_batch(X, T, lr))
        avg = float(np.mean(losses))
        self.nn.loss_history.append(avg)
        if len(self.nn.loss_history) > 300:
            self.nn.loss_history.pop(0)
        return avg

    def make_targets(self, samples, outcome, total_plies, gamma=0.97, w_search=0.5):
        for fen, s, ply in samples:
            r = max(0, total_plies - ply)
            disc_outcome = outcome * 1000.0 * (gamma ** r)
            target = w_search * s + (1.0 - w_search) * disc_outcome
            self.add_sample(fen, target)

    def save(self):
        try:
            self.nn.save(self.model_path)
            with open(self.stats_path, 'w') as f:
                json.dump({
                    'games_played': self.games_played,
                    'wins': self.wins, 'losses': self.losses, 'draws': self.draws,
                    'games_selfplay': self.games_selfplay,
                    'training_steps': self.nn.training_steps,
                    'loss_history': self.nn.loss_history,
                }, f)
        except Exception as e:
            print("Kaydetme hatasi:", e)

    @classmethod
    def load(cls, model_path=MODEL_PATH, stats_path=STATS_PATH):
        b = cls(model_path, stats_path)
        b.nn = NeuralNetwork.load(model_path)
        if os.path.exists(stats_path):
            try:
                with open(stats_path) as f:
                    d = json.load(f)
                b.games_played = d.get('games_played', 0)
                b.wins = d.get('wins', 0)
                b.losses = d.get('losses', 0)
                b.draws = d.get('draws', 0)
                b.games_selfplay = d.get('games_selfplay', 0)
                b.nn.loss_history = d.get('loss_history', [])
            except Exception:
                pass
        return b

    def level(self):
        return machine_level_info(self.nn.training_steps)


# ===================== ARAMA =====================
MATE = 100000
INF = 10 ** 9


class Engine:
    def __init__(self, brain, time_limit=5.0, max_depth=12):
        self.brain = brain
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.start_time = 0.0
        self.nodes = 0
        self.info = {'depth': 0, 'nodes': 0, 'score': 0.0}
        self.tt = {}
        self.killers = [[None, None] for _ in range(64)]
        self.history_h = {}
        self.eval_cache = {}

    def search(self, board):
        self.start_time = time.time()
        self.nodes = 0
        moves = list(board.legal_moves)
        if not moves:
            return None, 0.0, 0, 0
        if len(moves) == 1:
            return moves[0], 0.0, 0, 0

        self._order(board, moves, 0, None)
        best_mv, best_sc = moves[0], 0.0
        for depth in range(1, self.max_depth + 1):
            if time.time() - self.start_time > self.time_limit:
                break
            try:
                sc, mv = -INF, None
                alpha = -INF
                for m in moves:
                    board.push(m)
                    try:
                        s = -self._negamax(board, depth - 1, -INF, -alpha, 1)
                    finally:
                        board.pop()
                    if s > sc:
                        sc, mv = s, m
                    if s > alpha:
                        alpha = s
                if mv is not None:
                    best_mv, best_sc = mv, sc
                    moves.remove(mv)
                    moves.insert(0, mv)
                    self.info = {'depth': depth, 'nodes': self.nodes, 'score': sc}
                if abs(sc) > MATE - 1000:
                    break
            except TimeoutError:
                break
        return best_mv, best_sc, self.info['depth'], self.nodes

    def _negamax(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        if (self.nodes & 2047) == 0 and time.time() - self.start_time > self.time_limit:
            raise TimeoutError

        if board.is_repetition(2) or board.is_fifty_moves():
            return 0
        if board.is_insufficient_material():
            return 0

        key = chess.polyglot.zobrist_hash(board)
        tt_move = None
        entry = self.tt.get(key)
        if entry is not None:
            e_depth, e_flag, e_score, e_move = entry
            tt_move = e_move
            if e_depth >= depth:
                s = e_score
                if s > MATE - 1000:      s -= ply
                elif s < -(MATE - 1000): s += ply
                if e_flag == 0:                 return s
                if e_flag == 1 and s >= beta:   return s
                if e_flag == 2 and s <= alpha:  return s

        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply)

        in_check = board.is_check()
        moves = list(board.legal_moves)
        if not moves:
            return -(MATE - ply) if in_check else 0

        self._order(board, moves, ply, tt_move)

        best, best_mv = -INF, None
        alpha0 = alpha
        for m in moves:
            board.push(m)
            try:
                s = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if s > best:
                best, best_mv = s, m
            if best > alpha:
                alpha = best
            if alpha >= beta:
                if not board.is_capture(m):
                    if ply < 64:
                        k = self.killers[ply]
                        if m != k[0]:
                            k[1] = k[0]
                        k[0] = m
                    key2 = (m.from_square, m.to_square)
                    self.history_h[key2] = self.history_h.get(key2, 0) + depth * depth
                break

        if best >= beta:      flag = 1
        elif best <= alpha0:  flag = 2
        else:                 flag = 0
        s_store = best
        if s_store > MATE - 1000:      s_store += ply
        elif s_store < -(MATE - 1000): s_store -= ply
        self.tt[key] = (depth, flag, s_store, best_mv)
        if len(self.tt) > 400000:
            self.tt.clear()
        return best

    def _quiesce(self, board, alpha, beta, ply):
        self.nodes += 1
        if (self.nodes & 2047) == 0 and time.time() - self.start_time > self.time_limit:
            raise TimeoutError

        if board.is_check():
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE - ply)
            self._order(board, moves, ply, None)
            best = -INF
            for m in moves:
                board.push(m)
                try:
                    s = -self._quiesce(board, -beta, -alpha, ply + 1)
                finally:
                    board.pop()
                if s > best: best = s
                if best > alpha: alpha = best
                if alpha >= beta: return best
            return best

        key = chess.polyglot.zobrist_hash(board)
        stand = self.eval_cache.get(key)
        if stand is None:
            stand = self.brain.eval_stm(board)
            self.eval_cache[key] = stand
            if len(self.eval_cache) > 200000:
                self.eval_cache.clear()
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand

        moves = [m for m in board.generate_legal_moves()
                 if board.is_capture(m) or m.promotion]
        self._order(board, moves, ply, None)
        for m in moves:
            board.push(m)
            try:
                s = -self._quiesce(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if s >= beta:
                return s
            if s > alpha:
                alpha = s
        return alpha

    def _order(self, board, moves, ply, tt_move):
        killers = self.killers[ply] if ply < 64 else (None, None)
        hist = self.history_h

        def key(m):
            if tt_move is not None and m == tt_move:
                return 1 << 20
            s = 0
            victim = board.piece_at(m.to_square)
            if victim is not None:
                s = 100000 + PIECE_VALUES[victim.piece_type] * 10 \
                    - PIECE_VALUES.get(board.piece_type_at(m.from_square) or 0, 0)
            elif board.is_en_passant(m):
                s = 100000 + 900
            if m.promotion:
                s += 90000 + PIECE_VALUES.get(m.promotion, 0)
            if s == 0:
                if m == killers[0]:   s = 80000
                elif m == killers[1]: s = 79000
                else:                 s = hist.get((m.from_square, m.to_square), 0)
            return s

        moves.sort(key=key, reverse=True)


# ===================== SELF-PLAY =====================
def selfplay_worker(brain, history, stats, lock, stop, depth=3, time_limit=0.3,
                    max_plies=250, max_games=None, history_path=HISTORY_PATH):
    game = 0
    while not stop.is_set():
        if max_games is not None and game >= max_games:
            break
        game += 1
        engine = Engine(brain, time_limit, max_depth=depth)
        board = chess.Board()
        samples = []

        while not stop.is_set():
            if (board.is_game_over() or board.ply() >= max_plies
                    or board.is_repetition(3)):
                break
            if board.ply() < 4:
                mv = random.choice(list(board.legal_moves))
                board.push(mv)
                with lock:
                    stats['cur_len'] = board.ply()
                    stats['cur_move'] = mv.uci()
                continue
            mv, sc, dep, nodes = engine.search(board)
            if mv is None:
                break
            wsc = sc if board.turn == chess.WHITE else -sc
            samples.append((board.fen(), wsc, board.ply()))
            san = board.san(mv)
            board.push(mv)
            with lock:
                stats['cur_len'] = board.ply()
                stats['cur_move'] = san
                stats['cur_depth'] = dep

        if stop.is_set():
            break

        z = 0
        o = board.outcome()
        if o is not None and o.winner == chess.WHITE:
            z = 1
        elif o is not None and o.winner == chess.BLACK:
            z = -1
        total = board.ply()
        brain.make_targets(samples, z, total)
        loss = brain.train_from_buffer(steps=min(150, max(30, len(brain.buffer) // 8)))
        brain.games_selfplay += 1
        brain.save()

        with lock:
            stats['games'] += 1
            if z == 1:    stats['white_wins'] += 1
            elif z == -1: stats['black_wins'] += 1
            else:         stats['draws'] += 1
            stats['total_plies'] += total
            stats['avg_len'] = stats['total_plies'] / max(1, stats['games'])
            stats['last_loss'] = loss
            stats['steps'] = brain.nn.training_steps
            append_history(history, {
                'game_num': brain.games_selfplay,
                'mode': 'selfplay',
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'depth': depth,
                'result': 'Beyaz kazandi' if z == 1 else ('Siyah kazandi' if z == -1 else 'Berabere'),
                'result_val': z,
                'num_moves': total,
                'moves_san': None,
                'learning': loss,
            }, history_path)


# ===================== GECMIS =====================
MAX_HISTORY = 500


def load_history(path=HISTORY_PATH):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)[-MAX_HISTORY:]
        except Exception:
            pass
    return []


def save_history(h, path=HISTORY_PATH):
    try:
        with open(path, 'w') as f:
            json.dump(h, f, indent=1, ensure_ascii=False)
    except Exception as e:
        print("Gecmis kaydetme hatasi:", e)


def append_history(history, entry, path=HISTORY_PATH):
    history.append(entry)
    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]
    save_history(history, path)


# ===================== CIZIM YARDIMCILARI =====================
_BG_SURFACE = None


def get_background():
    global _BG_SURFACE
    if _BG_SURFACE is None:
        surf = pygame.Surface((WIDTH, HEIGHT))
        for y in range(HEIGHT):
            pygame.draw.line(surf, lerp(C_CRUST, C_MANTLE, y / HEIGHT),
                             (0, y), (WIDTH, y))
        for i in range(-HEIGHT, WIDTH, 90):
            pygame.draw.line(surf, (21, 21, 34), (i, HEIGHT), (i + HEIGHT, 0))
        _BG_SURFACE = surf
    return _BG_SURFACE


def text_at(screen, s, xy, font, color, anchor='topleft'):
    surf = font.render(s, True, color)
    r = surf.get_rect(**{anchor: xy})
    screen.blit(surf, r)
    return r


def draw_card(screen, rect, fill=C_CARD, border=C_BORDER, radius=12, shadow=True):
    if shadow:
        pygame.draw.rect(screen, C_SHADOW, rect.move(3, 4), border_radius=radius)
    pygame.draw.rect(screen, fill, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=radius)


def draw_button(screen, rect, label, font, base=C_CARD_HI, hover=None,
                text_color=C_TEXT, selected=False, radius=10):
    hov = rect.collidepoint(pygame.mouse.get_pos())
    if hover is None:
        hover = lerp(base, (255, 255, 255), 0.08)
    fill = C_ACCENT if selected else (hover if hov else base)
    pygame.draw.rect(screen, C_SHADOW, rect.move(2, 3), border_radius=radius)
    pygame.draw.rect(screen, fill, rect, border_radius=radius)
    if selected:
        border = lerp(C_ACCENT, (255, 255, 255), 0.35)
    else:
        border = lerp(C_BORDER, C_ACCENT, 0.45) if hov else C_BORDER
    pygame.draw.rect(screen, border, rect, width=2, border_radius=radius)
    surf = font.render(label, True, C_CRUST if selected else text_color)
    screen.blit(surf, surf.get_rect(center=rect.center))
    return hov


def draw_segmented(screen, rect, options, selected_idx, font, gap=8):
    n = len(options)
    seg_w = (rect.width - gap * (n - 1)) // n
    out = []
    for i, (label, val) in enumerate(options):
        r = pygame.Rect(rect.x + i * (seg_w + gap), rect.y, seg_w, rect.height)
        draw_button(screen, r, label, font, selected=(i == selected_idx))
        out.append((r, val))
    return out


def draw_progress(screen, rect, frac, fill=C_ACCENT):
    pygame.draw.rect(screen, (14, 14, 24), rect, border_radius=rect.height // 2)
    w = int(rect.width * max(0.0, min(1.0, frac)))
    if w > 0:
        pygame.draw.rect(screen, fill,
                         (rect.x, rect.y, w, rect.height),
                         border_radius=rect.height // 2)


def draw_chart(screen, rect, values, color=C_YELLOW, label=None, fsm=None):
    draw_card(screen, rect, fill=C_SURFACE)
    if label and fsm:
        text_at(screen, label, (rect.x + 14, rect.y + 10), fsm, C_MUTED)
    vals = [v for v in values[-140:] if v is not None]
    if len(vals) < 2:
        text_at(screen, "veri yok", (rect.centerx, rect.centery + 10),
                fsm or pygame.font.Font(None, 16), C_FAINT, 'center')
        return
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    pad_t, pad_b, pad_x = 44, 18, 16
    pts = []
    for i, v in enumerate(vals):
        x = rect.x + pad_x + (rect.width - 2 * pad_x) * i / (len(vals) - 1)
        y = rect.y + rect.height - pad_b - (rect.height - pad_t - pad_b) * (v - lo) / (hi - lo)
        pts.append((x, y))
    if len(pts) >= 2:
        rel = [(px - rect.x, py - rect.y) for px, py in pts]
        base = rect.height - pad_b
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.polygon(overlay, (249, 226, 175, 38),
                            rel + [(rel[-1][0], base), (rel[0][0], base)])
        screen.blit(overlay, rect.topleft)
        pygame.draw.lines(screen, color, False, pts, 2)
        pygame.draw.circle(screen, color, (int(pts[-1][0]), int(pts[-1][1])), 4)

    if fsm:
        for v, yy in ((hi, rect.y + 34), (lo, rect.bottom - 24)):
            s = fsm.render(f"{v:.5f}", True, C_MUTED)
            bg = pygame.Surface((s.get_width() + 10, s.get_height() + 4), pygame.SRCALPHA)
            bg.fill((24, 24, 37, 230))
            r = s.get_rect(topright=(rect.right - 14, yy))
            screen.blit(bg, (r.x - 5, r.y - 2))
            screen.blit(s, r)


def draw_quit_button(screen, font):
    rect = pygame.Rect(WIDTH - 54, 16, 36, 36)
    hov = rect.collidepoint(pygame.mouse.get_pos())
    fill = (190, 82, 95) if hov else (105, 54, 64)
    pygame.draw.rect(screen, C_SHADOW, rect.move(2, 3), border_radius=9)
    pygame.draw.rect(screen, fill, rect, border_radius=9)
    pygame.draw.rect(screen, lerp(fill, (255, 255, 255), 0.25), rect, 1, border_radius=9)
    t = font.render("X", True, (245, 235, 235))
    screen.blit(t, t.get_rect(center=rect.center))
    return rect


def piece_font(size):
    """Satranc sembollerini iceren ilk kullanilabilir fontu bulur."""
    for name in ('segoeuisymbol', 'applesymbols', 'notosanssymbols2',
                 'dejavusans', 'notosanssymbols', 'symbola'):
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


# ===================== TAHTA =====================
def sq_to_xy(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return f * SQUARE, (7 - r) * SQUARE


def xy_to_sq(x, y):
    col, row = x // SQUARE, y // SQUARE
    if not (0 <= col < 8 and 0 <= row < 8):
        return None
    return chess.square(col, 7 - row)


def is_promotion_move(board, frm, to):
    p = board.piece_at(frm)
    return (p is not None and p.piece_type == chess.PAWN
            and chess.square_rank(to) in (0, 7))


def draw_board(screen, board, sel, targets, last_moves, fpiece, fsm, hover_sq=None):
    for row in range(8):
        for col in range(8):
            c = SQ_LIGHT if (row + col) % 2 == 0 else SQ_DARK
            pygame.draw.rect(screen, c, (col * SQUARE, row * SQUARE, SQUARE, SQUARE))

    for mv in last_moves:
        for sq in (mv.from_square, mv.to_square):
            x, y = sq_to_xy(sq)
            s = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
            s.fill((255, 215, 0, 70))
            screen.blit(s, (x, y))

    if board.is_check():
        kq = board.king(board.turn)
        if kq is not None:
            x, y = sq_to_xy(kq)
            s = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
            s.fill((240, 60, 80, 110))
            screen.blit(s, (x, y))
            pygame.draw.rect(screen, (240, 60, 80), (x, y, SQUARE, SQUARE), 3)

    if hover_sq is not None:
        x, y = sq_to_xy(hover_sq)
        s = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
        s.fill((255, 255, 255, 30))
        screen.blit(s, (x, y))

    if sel is not None:
        x, y = sq_to_xy(sel)
        s = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
        s.fill((137, 180, 250, 70))
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, C_ACCENT, (x, y, SQUARE, SQUARE), 3)

    capturable = set()
    for t in targets:
        if board.piece_at(t) is not None:
            capturable.add(t)
    for t in targets:
        x, y = sq_to_xy(t)
        cx, cy = x + SQUARE // 2, y + SQUARE // 2
        if t in capturable:
            pygame.draw.circle(screen, (50, 120, 50), (cx, cy), SQUARE // 2 - 8, 6)
        else:
            pygame.draw.circle(screen, (50, 130, 50), (cx, cy), 11)

    for i, ch in enumerate('abcdefgh'):
        sh = fsm.render(ch, True, (30, 20, 10))
        screen.blit(sh, (i * SQUARE + 6, BOARD_SIZE - 20))
        text_at(screen, ch, (i * SQUARE + 5, BOARD_SIZE - 21), fsm, (250, 245, 235))

    for i in range(8):
        sh = fsm.render(str(8 - i), True, (30, 20, 10))
        screen.blit(sh, (6, i * SQUARE + 5))
        text_at(screen, str(8 - i), (5, i * SQUARE + 4), fsm, (250, 245, 235))

    for sq, p in board.piece_map().items():
        sym = PIECE_SYMBOLS[p.piece_type]
        col = (250, 250, 250) if p.color == chess.WHITE else (25, 25, 30)
        x, y = sq_to_xy(sq)
        outline = fpiece.render(sym, True, (10, 10, 14))
        colored = fpiece.render(sym, True, col)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            screen.blit(outline, (x + dx + 8, y + dy + 4))
        screen.blit(colored, (x + 8, y + 4))


def draw_eval_bar(screen, brain, board, fsm):
    score = brain.eval_white(board)
    frac = 0.5 + 0.5 * math.tanh(score / 600.0)
    x, y, w, h = EVAL_X, 14, EVAL_W, BOARD_SIZE - 28
    pygame.draw.rect(screen, C_SHADOW, (x + 2, y + 3, w, h), border_radius=9)
    pygame.draw.rect(screen, (35, 35, 52), (x, y, w, h), border_radius=9)
    wh = int(h * frac)
    if wh > 0:
        pygame.draw.rect(screen, (235, 236, 244),
                         (x + 2, y + h - wh, w - 4, max(0, wh - 4)), border_radius=7)
    pygame.draw.rect(screen, C_BORDER, (x, y, w, h), 1, border_radius=9)
    my = y + h // 2
    pygame.draw.line(screen, C_ACCENT, (x - 4, my), (x + w + 4, my), 2)
    return score


# ===================== OYUN EKRANI =====================
R_HDR  = pygame.Rect(PANEL_X + 12, 12, PANEL_W - 24, 52)
R_STAT = pygame.Rect(PANEL_X + 12, 74, PANEL_W - 24, 60)
R_INFO = pygame.Rect(PANEL_X + 12, 144, PANEL_W - 24, 96)
R_MOVES = pygame.Rect(PANEL_X + 12, 250, PANEL_W - 24, 254)
R_LAST = pygame.Rect(PANEL_X + 12, 512, PANEL_W - 24, 56)
R_BTN_UNDO = pygame.Rect(PANEL_X + 12, 578, 196, 46)
R_BTN_MENU = pygame.Rect(PANEL_X + 220, 578, 196, 46)


def draw_panel(screen, brain, move_records, player_color, game_num, board,
               thinking_flag, engine_info, fm, fsm, move_scroll):
    pygame.draw.rect(screen, C_MANTLE, (PANEL_X, 0, PANEL_W, BOARD_SIZE))

    draw_card(screen, R_HDR)
    text_at(screen, f"Oyun #{game_num}", (R_HDR.x + 14, R_HDR.y + 8), fm, C_ACCENT)
    side = 'BEYAZ' if player_color == chess.WHITE else 'SIYAH'
    text_at(screen, f"Sen: {side}", (R_HDR.right - 14, R_HDR.y + 8), fsm, C_MUTED, 'topright')
    text_at(screen, "vs MAKINE", (R_HDR.right - 14, R_HDR.y + 28), fsm, C_FAINT, 'topright')

    draw_card(screen, R_STAT)
    if board.is_game_over():
        turn_txt, dot_col = "Oyun bitti", C_FAINT
    elif board.turn == player_color:
        turn_txt, dot_col = "Sira sende", C_GREEN
    elif thinking_flag:
        dots = "." * (1 + int(time.time() * 2) % 3)
        turn_txt, dot_col = f"Makine dusunuyor{dots}", C_YELLOW
    else:
        turn_txt, dot_col = "Sira: Makine", C_RED
    pygame.draw.circle(screen, dot_col, (R_STAT.x + 22, R_STAT.y + 21), 6)
    text_at(screen, turn_txt, (R_STAT.x + 38, R_STAT.y + 11), fm, C_TEXT)
    sub = (f"derinlik={engine_info['depth']}  dugum={engine_info['nodes']:,}"
           if thinking_flag else "hamle yapmak icin tahtaya tikla")
    text_at(screen, sub, (R_STAT.x + 38, R_STAT.y + 36), fsm, C_MUTED)

    draw_card(screen, R_INFO)
    lvl_name, lvl_idx, nxt = brain.level()
    steps = brain.nn.training_steps
    text_at(screen, f"Seviye: {lvl_name}", (R_INFO.x + 14, R_INFO.y + 10), fm, C_ACCENT2)
    text_at(screen, f"{steps} adim", (R_INFO.right - 14, R_INFO.y + 14), fsm, C_MUTED, 'topright')
    if nxt is not None:
        cur_thr = MACHINE_LEVELS[lvl_idx][0]
        frac = (steps - cur_thr) / max(1, nxt - cur_thr)
    else:
        frac = 1.0
    draw_progress(screen, pygame.Rect(R_INFO.x + 14, R_INFO.y + 38, R_INFO.width - 28, 8),
                  frac, C_ACCENT2)
    if nxt is not None:
        text_at(screen, f"sonraki seviyeye {nxt - steps} adim",
                (R_INFO.x + 14, R_INFO.y + 52), fsm, C_HINT)
    else:
        text_at(screen, "en ust seviye!", (R_INFO.x + 14, R_INFO.y + 52), fsm, C_HINT)
    text_at(screen, f"oz-oyun: {brain.games_selfplay}   insana karsi: {brain.games_played} "
                    f"(W{brain.wins}/L{brain.losses}/D{brain.draws})",
            (R_INFO.x + 14, R_INFO.y + 74), fsm, C_MUTED)

    draw_card(screen, R_MOVES)
    text_at(screen, "HAMLELER", (R_MOVES.x + 14, R_MOVES.y + 10), fsm, C_MUTED)
    lines = []
    for i in range(0, len(move_records), 2):
        num = i // 2 + 1
        w = move_records[i]['san'] if i < len(move_records) else ''
        b = move_records[i + 1]['san'] if i + 1 < len(move_records) else ''
        lines.append((num, w, b))
    visible = 10
    max_scroll = max(0, len(lines) - visible)
    start = min(move_scroll[0], max_scroll)
    move_scroll[0] = start
    for j, (num, w, b) in enumerate(lines[start:start + visible]):
        ly = R_MOVES.y + 36 + j * 20
        is_last = (start + j) == len(lines) - 1
        text_at(screen, f"{num}.", (R_MOVES.x + 14, ly), fsm,
                C_ACCENT if is_last else C_FAINT)
        text_at(screen, w, (R_MOVES.x + 48, ly), fsm,
                C_TEXT if is_last else C_MUTED)
        text_at(screen, b, (R_MOVES.x + 130, ly), fsm,
                C_TEXT if is_last else C_MUTED)
    if max_scroll > 0:
        text_at(screen, "fare tekerlegi ile kaydir", (R_MOVES.x + 14, R_MOVES.bottom - 22),
                fsm, C_HINT)

    draw_card(screen, R_LAST)
    text_at(screen, "SON MAKINE HAMLESI", (R_LAST.x + 14, R_LAST.y + 8), fsm, C_FAINT)
    ai_side = 'white' if player_color == chess.BLACK else 'black'
    found = None
    for rec in reversed(move_records):
        if rec['side'] == ai_side and rec.get('thinking'):
            found = rec
            break
    if found:
        t = found['thinking']
        wsc = t['score'] if found['side'] == 'white' else -t['score']
        text_at(screen, found['san'], (R_LAST.x + 14, R_LAST.y + 26), fm, C_ACCENT)
        text_at(screen, f"derinlik {t['depth']}  deger {wsc:+.0f}  {t['nodes']:,} dugum",
                (R_LAST.right - 14, R_LAST.y + 30), fsm, C_MUTED, 'topright')
    else:
        text_at(screen, "(henuz yok)", (R_LAST.x + 14, R_LAST.y + 26), fm, C_FAINT)

    can_undo = (not board.is_game_over() and not thinking_flag
                and board.turn == player_color)
    draw_button(screen, R_BTN_UNDO, "Geri Al", fm,
                base=C_CARD if can_undo else (30, 31, 45),
                text_color=C_TEXT if can_undo else C_FAINT)
    draw_button(screen, R_BTN_MENU, "Menuye Don", fm, base=(70, 55, 60), hover=(95, 70, 75))


def draw_bottom_strip(screen, brain, board, thinking_flag, engine_info, fm, fsm):
    draw_card(screen, pygame.Rect(12, BOTTOM_Y + 8, WIDTH - 24, BOTTOM_H - 20))
    if thinking_flag:
        txt, color = (f"Makine dusunuyor...  derinlik={engine_info['depth']}"
                      f"  dugum={engine_info['nodes']:,}  deger={engine_info['score']:+.0f}"), C_YELLOW
    elif board.is_game_over():
        o = board.outcome()
        if o is not None and o.winner is not None:
            txt = f"Oyun bitti: {'Beyaz' if o.winner == chess.WHITE else 'Siyah'} kazandi"
        else:
            txt = "Oyun bitti: Berabere"
        color = C_MUTED
    else:
        txt, color = "Sira sende - tasina tikla, hedef kareye tikla", C_GREEN

    score = brain.eval_white(board)
    if abs(score) < 15:
        ev_txt, ev_col = "dengeli", C_MUTED
    elif score > 0:
        ev_txt, ev_col = f"beyaz ustun ({score:+.0f})", C_GREEN_L
    else:
        ev_txt, ev_col = f"siyah ustun ({score:+.0f})", C_RED_L
    nw = int(brain.nn_weight() * 100)
    ev_txt += f"   [degerlendirme: %{100 - nw} klasik + %{nw} ogrenilmis]"

    text_at(screen, txt, (28, BOTTOM_Y + 16), fm, color)
    text_at(screen, f"Makine degerlendirmesi: {ev_txt}", (28, BOTTOM_Y + 42), fsm, ev_col)
    bar = pygame.Rect(WIDTH - 300, BOTTOM_Y + 32, 190, 10)
    frac = 0.5 + 0.5 * math.tanh(score / 600.0)
    draw_progress(screen, bar, frac, C_TEAL)
    text_at(screen, "makine degeri", (bar.centerx, BOTTOM_Y + 12), fsm, C_HINT, 'midtop')
    text_at(screen, f"{score:+.0f}", (bar.right + 12, bar.centery - 8), fsm,
            C_MUTED, 'midleft')


def draw_game_frame(screen, brain, board, sel, targets, last_moves, move_records,
                    player_color, game_num, thinking_flag, engine_info,
                    fonts, move_scroll, hover_sq=None):
    fpiece, fm, fsm = fonts
    screen.blit(get_background(), (0, 0))
    draw_board(screen, board, sel, targets, last_moves, fpiece, fsm, hover_sq)
    draw_eval_bar(screen, brain, board, fsm)
    draw_panel(screen, brain, move_records, player_color, game_num, board,
               thinking_flag, engine_info, fm, fsm, move_scroll)
    draw_bottom_strip(screen, brain, board, thinking_flag, engine_info, fm, fsm)


# ===================== MENU =====================
MENU_TIMES = [(2, "Cok Hizli 2s"), (5, "Hizli 5s"), (20, "Normal 20s"), (60, "Derin 60s")]


def draw_menu_frame(screen, fonts, brain, sel_t):
    fb, fm, fsm = fonts
    screen.blit(get_background(), (0, 0))

    t1, t2 = "SATRANC ", "MACHINE"
    w1, w2 = fb.size(t1)[0], fb.size(t2)[0]
    x0 = WIDTH // 2 - (w1 + w2) // 2
    text_at(screen, t1, (x0, 56), fb, C_TEXT, 'midleft')
    text_at(screen, t2, (x0 + w1, 56), fb, C_ACCENT, 'midleft')
    text_at(screen, "kendini gelistiren satranc makinesi",
            (WIDTH // 2, 106), fm, C_MUTED, 'midtop')

    card = pygame.Rect(WIDTH // 2 - 240, 140, 480, 108)
    draw_card(screen, card)
    lvl_name, lvl_idx, nxt = brain.level()
    steps = brain.nn.training_steps
    text_at(screen, lvl_name, (card.centerx, card.y + 16), fm, C_ACCENT2, 'midtop')
    if nxt is not None:
        cur_thr = MACHINE_LEVELS[lvl_idx][0]
        frac = (steps - cur_thr) / max(1, nxt - cur_thr)
        text_at(screen, f"sonraki seviyeye {nxt - steps} adim",
                (card.centerx, card.y + 46), fsm, C_FAINT, 'midtop')
    else:
        frac = 1.0
    draw_progress(screen, pygame.Rect(card.x + 40, card.y + 66, card.width - 80, 10), frac, C_ACCENT2)
    text_at(screen, f"oz-oyun: {brain.games_selfplay}   insana karsi: {brain.games_played} "
                    f"(Makine W{brain.wins}/L{brain.losses}/D{brain.draws})",
            (card.centerx, card.y + 84), fsm, C_MUTED, 'midtop')

    bw = pygame.Rect(WIDTH // 2 - 240, 272, 232, 62)
    bb = pygame.Rect(WIDTH // 2 + 8, 272, 232, 62)
    draw_button(screen, bw, "BEYAZ OYNA", fm, base=(44, 82, 72), hover=(54, 102, 88))
    text_at(screen, "(once sen hamle yaparsin)", (bw.centerx, bw.bottom - 16), fsm, C_MUTED, 'midtop')
    draw_button(screen, bb, "SIYAH OYNA", fm, base=(88, 56, 66), hover=(108, 68, 80))
    text_at(screen, "(once makine hamle yapar)", (bb.centerx, bb.bottom - 16), fsm, C_MUTED, 'midtop')

    btrain = pygame.Rect(WIDTH // 2 - 240, 352, 480, 58)
    draw_button(screen, btrain, "KENDI KENDINE OGREN  (SELF-PLAY EGITIM)", fm,
                base=(46, 102, 86), hover=(56, 124, 104))

    binsp = pygame.Rect(WIDTH // 2 - 240, 426, 480, 58)
    draw_button(screen, binsp, "MAKINEYI INCELE  (istatistik + agirliklar)", fm,
                base=(72, 64, 110), hover=(88, 78, 132))

    text_at(screen, "Dusunme suresi (hamle basi):", (WIDTH // 2 - 240, 506), fm, C_MUTED)
    seg = draw_segmented(screen, pygame.Rect(WIDTH // 2 - 240, 538, 480, 44),
                         [(t, i) for i, (v, t) in enumerate(MENU_TIMES)], sel_t, fm)

    text_at(screen, "F11 = Tam Ekran    |    ESC = Cikis    |    "
                    "Makine her oyundan ogrenir", (WIDTH // 2, 640), fsm, C_HINT, 'midtop')

    draw_quit_button(screen, fb)
    return {'white': bw, 'black': bb, 'train': btrain, 'inspect': binsp,
            'times': seg, 'quit': pygame.Rect(WIDTH - 54, 16, 36, 36)}


def menu(screen, fonts, brain):
    sel_t = 2
    fb, fm, fsm = fonts
    while True:
        rects = draw_menu_frame(screen, (fb, fm, fsm), brain, sel_t)
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if e.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
            if e.type == pygame.MOUSEBUTTONDOWN:
                if rects['quit'].collidepoint(e.pos):
                    pygame.quit(); sys.exit()
                if rects['white'].collidepoint(e.pos):
                    return (chess.WHITE, MENU_TIMES[sel_t][0])
                if rects['black'].collidepoint(e.pos):
                    return (chess.BLACK, MENU_TIMES[sel_t][0])
                if rects['train'].collidepoint(e.pos):
                    return ('train', None)
                if rects['inspect'].collidepoint(e.pos):
                    return ('inspect', None)
                for r, val in rects['times']:
                    if r.collidepoint(e.pos):
                        sel_t = val


# ===================== SELF-PLAY EKRANI =====================
TRAIN_SPEEDS = [(2, 0.15, "HIZLI  d2"), (3, 0.4, "NORMAL  d3"), (4, 1.2, "DERIN  d4")]


def draw_train_frame(screen, fonts, brain, stats, sel):
    fb, fm, fsm = fonts
    screen.blit(get_background(), (0, 0))

    text_at(screen, "SELF-PLAY EGITIMI", (30, 26), fb, C_ACCENT)
    text_at(screen, "Makine kendi kendine oynuyor ve her oyundan ogreniyor...",
            (30, 78), fm, C_MUTED)

    lvl_name, _, _ = brain.level()

    c1 = pygame.Rect(30, 116, 470, 300)
    draw_card(screen, c1)
    y = c1.y + 16
    rows = [
        ("Tamamlanan oyun (oturum)", f"{stats['games']}  (toplam {brain.games_selfplay})"),
        ("Beyaz / Siyah / Berabere", f"{stats['white_wins']} / {stats['black_wins']} / {stats['draws']}"),
        ("Ortalama oyun uzunlugu", f"{stats['avg_len']:.0f} yarim hamle"),
        ("Son ogrenme hatasi", f"{stats['last_loss']:.6f}"),
        ("Toplam egitim adimi", f"{brain.nn.training_steps}"),
        ("Ogrenme tamponu", f"{len(brain.buffer)} pozisyon"),
        ("Makine seviyesi", lvl_name),
    ]
    for k, v in rows:
        text_at(screen, k, (c1.x + 18, y), fsm, C_MUTED)
        text_at(screen, v, (c1.right - 18, y), fsm, C_TEXT, 'topright')
        y += 26
        if k != rows[-1][0]:
            pygame.draw.line(screen, (46, 48, 68), (c1.x + 18, y - 6),
                             (c1.right - 18, y - 6))

    c2 = pygame.Rect(30, 430, 470, 120)
    draw_card(screen, c2)
    text_at(screen, "SU ANKI OYUN", (c2.x + 18, c2.y + 12), fsm, C_FAINT)
    text_at(screen, f"{stats['cur_len']} yarim hamle", (c2.x + 18, c2.y + 36), fm, C_TEXT)
    text_at(screen, f"son hamle: {stats['cur_move']}", (c2.x + 18, c2.y + 66), fm, C_ACCENT)
    text_at(screen, f"arama derinligi: {stats.get('cur_depth', 0)}",
            (c2.right - 18, c2.y + 40), fsm, C_MUTED, 'topright')

    c3 = pygame.Rect(520, 116, 550, 434)
    draw_chart(screen, c3, brain.nn.loss_history, C_YELLOW,
               "OGRENME HATASI (LOSS) EGRISI", fsm)

    seg = draw_segmented(screen, pygame.Rect(30, 572, 470, 46),
                         [(n, i) for i, (d, t, n) in enumerate(TRAIN_SPEEDS)], sel, fm)
    stop_btn = pygame.Rect(520, 572, 550, 46)
    draw_button(screen, stop_btn, "DURDUR ve Menuye Don", fm,
                base=(130, 62, 72), hover=(160, 76, 88))

    text_at(screen, "Hiz degistirilirse yeni ayarla devam edilir", (30, 632), fsm, C_HINT)
    draw_quit_button(screen, fb)
    return {'seg': seg, 'stop': stop_btn,
            'quit': pygame.Rect(WIDTH - 54, 16, 36, 36)}


def train_screen(screen, brain, history, fonts):
    fb, fm, fsm = fonts
    sel = 1
    stats = {'games': 0, 'white_wins': 0, 'black_wins': 0, 'draws': 0,
             'total_plies': 0, 'avg_len': 0.0, 'last_loss': 0.0, 'cur_len': 0,
             'cur_move': '-', 'cur_depth': 0, 'steps': brain.nn.training_steps}
    lock = threading.Lock()
    stop = threading.Event()
    thread = [None]

    def start_worker():
        d, tl, _ = TRAIN_SPEEDS[sel]
        stop.clear()
        with lock:
            stats.update({'games': 0, 'total_plies': 0,
                          'avg_len': 0.0, 'cur_len': 0, 'cur_move': '-',
                          'white_wins': 0, 'black_wins': 0, 'draws': 0,
                          'steps': brain.nn.training_steps})
        thread[0] = threading.Thread(
            target=selfplay_worker,
            args=(brain, history, stats, lock, stop, d, tl),
            daemon=True)
        thread[0].start()

    start_worker()

    while True:
        with lock:
            st = dict(stats)
        rects = draw_train_frame(screen, (fb, fm, fsm), brain, st, sel)
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                stop.set()
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    stop.set()
                    return
                if e.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
            if e.type == pygame.MOUSEBUTTONDOWN:
                if rects['quit'].collidepoint(e.pos):
                    stop.set()
                    pygame.quit(); sys.exit()
                if rects['stop'].collidepoint(e.pos):
                    stop.set()
                    return
                for r, val in rects['seg']:
                    if r.collidepoint(e.pos) and val != sel:
                        sel = val
                        stop.set()
                        old = thread[0]
                        if old is not None:
                            waited = 0.0
                            while old.is_alive() and waited < 60.0:
                                old.join(timeout=1.0)
                                waited += 1.0
                        start_worker()


# ===================== INCELEME EKRANI =====================
def draw_viewer_frame(screen, fonts, brain, history, tab, scroll, reset_armed):
    fb, fm, fsm = fonts
    screen.blit(get_background(), (0, 0))

    text_at(screen, "MAKINEYI INCELE", (30, 26), fb, C_ACCENT)
    lvl_name, _, _ = brain.level()
    text_at(screen, f"seviye: {lvl_name}", (WIDTH - 104, 49), fm, C_MUTED, 'midright')

    tabs = ["DURUM", "AGIRLIKLAR", "GECMIS", "SON OYUN"]
    tab_rects = draw_segmented(screen, pygame.Rect(30, 100, 660, 40),
                               [(t, i) for i, t in enumerate(tabs)], tab, fsm)

    if tab == 0:
        c = pygame.Rect(30, 156, 500, 320)
        draw_card(screen, c)
        recent = brain.nn.loss_history[-10:]
        avg_loss = float(np.mean(recent)) if recent else 0.0
        y = c.y + 18
        rows = [
            ("Egitim adimi", f"{brain.nn.training_steps}"),
            ("Son 10 oturum ort. hata", f"{avg_loss:.6f}"),
            ("Ogrenme tamponu", f"{len(brain.buffer)} pozisyon"),
            ("Self-play oyunlari", f"{brain.games_selfplay}"),
            ("Insana karsi oyunlar", f"{brain.games_played}"),
            ("Makine W / L / D", f"{brain.wins} / {brain.losses} / {brain.draws}"),
            ("Makine seviyesi", lvl_name),
        ]
        for k, v in rows:
            text_at(screen, k, (c.x + 18, y), fsm, C_MUTED)
            text_at(screen, v, (c.right - 18, y), fsm, C_TEXT, 'topright')
            y += 30
        reset = pygame.Rect(560, 452, 510, 52)
        draw_button(screen, reset,
                    "SIFIRLA - yeniden baslat" if not reset_armed
                    else "EMIN MISIN? TIKLA",
                    fm, base=(120, 52, 62), hover=(150, 64, 76))
        chart = pygame.Rect(560, 156, 510, 278)
        draw_chart(screen, chart, brain.nn.loss_history, C_YELLOW,
                   "OGRENME HATASI (LOSS)", fsm)

    elif tab == 1:
        c1 = pygame.Rect(30, 156, 500, 348)
        draw_card(screen, c1)
        text_at(screen, "GIZLI KATMAN -> CIKIS AGIRLIKLARI (W3, 64 noron)",
                (c1.x + 18, c1.y + 14), fsm, C_MUTED)
        w3 = brain.nn.W3.flatten()
        cw, ch, gap = 26, 26, 3
        grid_w = 16 * (cw + gap) - gap
        gx = c1.x + (c1.width - grid_w) // 2
        gy = c1.y + 64
        for i in range(64):
            val = float(w3[i])
            norm = max(-1.0, min(1.0, val * 5))
            if norm >= 0:
                col = (int(60 + norm * 140), int(60 + (1 - norm) * 60), 60)
            else:
                col = (int(60 + (1 + norm) * 60), int(60 + (-norm) * 140), 60)
            bx = gx + (i % 16) * (cw + gap)
            by = gy + (i // 16) * (ch + gap)
            pygame.draw.rect(screen, col, (bx, by, cw, ch), border_radius=5)
            pygame.draw.rect(screen, (80, 82, 105), (bx, by, cw, ch), 1, border_radius=5)
        text_at(screen, "yesil: pozitif etki    kirmizi: negatif etki",
                (c1.centerx, c1.bottom - 26), fsm, C_HINT, 'midtop')

        c2 = pygame.Rect(560, 156, 510, 348)
        draw_card(screen, c2)
        text_at(screen, "GIRDI KATMANI TAS AGIRLIK NORMLARI", (c2.x + 18, c2.y + 14), fsm, C_MUTED)
        for pt in range(12):
            col = brain.nn.W1[pt * 64:pt * 64 + 64]
            norm = float(np.mean(np.abs(col)))
            colr = (int(40 + min(1, norm * 4) * 180), 60, 90)
            bx = c2.x + 22 + (pt % 4) * 122
            by = c2.y + 48 + (pt // 4) * 98
            pygame.draw.rect(screen, colr, (bx, by, 110, 86), border_radius=8)
            pygame.draw.rect(screen, (80, 82, 105), (bx, by, 110, 86), 1, border_radius=8)
            nm = PIECE_NAMES.get(pt % 6 + 1, '?')
            side = 'B' if pt < 6 else 'S'
            text_at(screen, f"{side} {nm}", (bx + 10, by + 8), fsm, C_TEXT)
            text_at(screen, f"{norm:.3f}", (bx + 10, by + 60), fsm, C_MUTED)

    elif tab == 2:
        card = pygame.Rect(30, 156, WIDTH - 60, HEIGHT - 190)
        draw_card(screen, card)
        if not history:
            text_at(screen, "Henuz oynanmis oyun yok.", (WIDTH // 2, 380), fm, C_FAINT, 'center')
        else:
            text_at(screen, "NO      TARIH            MOD      SONUC            "
                            "HAMLE    OGRENME", (48, 168), fsm, C_HINT)
            pygame.draw.line(screen, (60, 62, 84), (48, 190), (card.right - 20, 190))
            rev = list(reversed(history))
            y = 200
            for i in range(scroll, min(scroll + 16, len(rev))):
                g = rev[i]
                mode = 'OZ-OYUN' if g.get('mode') == 'selfplay' else 'INSAN'
                line = (f"#{g['game_num']:<5d}  {g['timestamp'][:16]}  {mode:8s}"
                        f"{g['result']:14s} {g['num_moves']:4d} hamle  "
                        f"Loss={g.get('learning', 0):.5f}")
                if g.get('mode') == 'selfplay':
                    renk = C_MUTED
                else:
                    actual = g['result_val'] if g.get('player_color') == 'white' \
                        else -g['result_val']
                    renk = C_GREEN if actual == 1 else (C_RED if actual == -1 else C_YELLOW)
                text_at(screen, line, (48, y), fsm, renk)
                y += 27
            text_at(screen, "Yukari/Asagi ok ile kaydir",
                    (card.x + 18, card.bottom - 26), fsm, C_HINT)

    else:
        draw_card(screen, pygame.Rect(30, 156, WIDTH - 60, HEIGHT - 190))
        if not history:
            text_at(screen, "Henuz oyun yok.", (WIDTH // 2, 380), fm, C_FAINT, 'center')
        else:
            g = history[-1]
            y = 176
            text_at(screen, f"Oyun #{g['game_num']} - {g['result']} - {g['num_moves']} hamle",
                    (52, y), fm, C_ACCENT); y += 30
            text_at(screen, f"Ogrenme hatasi (Loss): {g.get('learning', 0):.6f}",
                    (52, y), fsm, C_GREEN); y += 26
            pygame.draw.line(screen, (60, 62, 84), (52, y), (WIDTH - 52, y)); y += 14
            details = g.get('move_details') or []
            if details:
                for i, det in enumerate(details[-26:]):
                    num = (g['num_moves'] - min(26, len(details)) + i) // 2 + 1
                    side = "B" if det['side'] == 'white' else "S"
                    th = det.get('thinking')
                    if th:
                        wsc = th['score'] if det['side'] == 'white' else -th['score']
                        line = f"{num:3d}.{side} {det['san']:7s} derinlik={th['depth']}  deger={wsc:+.0f}"
                    else:
                        line = f"{num:3d}.{side} {det['san']:7s}"
                    text_at(screen, line, (52, y), fsm, C_MUTED)
                    y += 20
                    if y > HEIGHT - 40:
                        break
            elif g.get('moves_san'):
                sans = g['moves_san']
                for i in range(0, len(sans), 2):
                    w = sans[i] if i < len(sans) else ''
                    b = sans[i + 1] if i + 1 < len(sans) else ''
                    text_at(screen, f"{i//2+1:3d}. {w:7s} {b:7s}", (52, y), fsm, C_MUTED)
                    y += 20
                    if y > HEIGHT - 40:
                        break

    text_at(screen, "1/2/3/4: sekme    |    F11: tam ekran    |    ESC: menu",
            (WIDTH // 2, HEIGHT - 14), fsm, C_HINT, 'midbottom')
    draw_quit_button(screen, fb)
    return {'tabs': tab_rects, 'reset': pygame.Rect(560, 452, 510, 52),
            'quit': pygame.Rect(WIDTH - 54, 16, 36, 36)}


def viewer(screen, brain, history, fonts):
    fb, fm, fsm = fonts
    tab = 0
    scroll = 0
    reset_armed = False
    while True:
        rects = draw_viewer_frame(screen, (fb, fm, fsm), brain, history,
                                  tab, scroll, reset_armed)
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return
                elif e.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_1: tab = 0
                elif e.key == pygame.K_2: tab = 1
                elif e.key == pygame.K_3: tab = 2; scroll = 0
                elif e.key == pygame.K_4: tab = 3
                elif e.key == pygame.K_UP: scroll = max(0, scroll - 1)
                elif e.key == pygame.K_DOWN:
                    scroll = min(max(0, len(history) - 17), scroll + 1)
            if e.type == pygame.MOUSEBUTTONDOWN:
                if rects['quit'].collidepoint(e.pos):
                    pygame.quit(); sys.exit()
                if tab == 0 and rects['reset'].collidepoint(e.pos):
                    if reset_armed:
                        for p in (brain.model_path, brain.stats_path):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                        brain.__init__(brain.model_path, brain.stats_path)
                        brain.save()
                        reset_armed = False
                    else:
                        reset_armed = True
                for r, val in rects['tabs']:
                    if r.collidepoint(e.pos):
                        tab = val
                        scroll = 0


# ===================== OGRENME =====================
def learn_from_human_game(brain, move_records, all_moves, out_val, player_color):
    ai_color = chess.WHITE if player_color == chess.BLACK else chess.BLACK
    ai_out = out_val if ai_color == chess.WHITE else -out_val
    brain.games_played += 1
    if ai_out == 1:
        brain.wins += 1
    elif ai_out == -1:
        brain.losses += 1
    else:
        brain.draws += 1

    board = chess.Board()
    samples = []
    for rec, mv in zip(move_records, all_moves):
        if rec.get('thinking'):
            s = rec['thinking']['score']
            wsc = s if board.turn == chess.WHITE else -s
            samples.append((board.fen(), wsc, board.ply()))
        board.push(mv)

    brain.make_targets(samples, out_val, len(all_moves))
    loss = brain.train_from_buffer(steps=min(150, max(30, len(brain.buffer) // 8)))
    brain.save()
    return loss


def draw_result_overlay(screen, brain, result_text, loss, fonts, learning):
    fm, fsm = fonts[1], fonts[2]

    shade = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
    shade.fill((10, 10, 22, 90))
    screen.blit(shade, (0, 0))

    card = pygame.Rect(44, 392, 552, 228)
    draw_card(screen, card, fill=C_SURFACE)
    if result_text == "Berabere":
        tcol = C_YELLOW
    elif result_text.startswith("Beyaz"):
        tcol = C_GREEN_L
    else:
        tcol = C_RED_L
    text_at(screen, result_text, (card.centerx, card.y + 16), fm, tcol, 'midtop')

    lvl_name, _, _ = brain.level()
    if learning:
        dots = "." * (1 + int(time.time() * 2) % 3)
        text_at(screen, f"Makine bu oyundan ogreniyor{dots}",
                (card.centerx, card.y + 54), fm, C_ACCENT, 'midtop')
        text_at(screen, "birkac saniye surebilir - tahtada son pozisyon duruyor",
                (card.centerx, card.y + 84), fsm, C_MUTED, 'midtop')
    else:
        text_at(screen, f"Loss: {loss:.6f}    egitim: {brain.nn.training_steps} adim    "
                        f"seviye: {lvl_name}",
                (card.centerx, card.y + 54), fsm, C_TEXT, 'midtop')
        text_at(screen, f"insana karsi: {brain.games_played}  "
                        f"(W{brain.wins}/L{brain.losses}/D{brain.draws})",
                (card.centerx, card.y + 78), fsm, C_MUTED, 'midtop')

    b_again = pygame.Rect(card.x + 24, card.bottom - 66, 240, 48)
    b_menu = pygame.Rect(card.right - 264, card.bottom - 66, 240, 48)
    if learning:
        draw_button(screen, b_again, "Yeni Oyun", fm, base=(40, 41, 58),
                    text_color=C_FAINT)
        draw_button(screen, b_menu, "Menuye Don", fm, base=(40, 41, 58),
                    text_color=C_FAINT)
    else:
        draw_button(screen, b_again, "YENI OYUN", fm,
                    base=(46, 102, 86), hover=(56, 124, 104))
        draw_button(screen, b_menu, "MENUYE DON", fm,
                    base=(100, 62, 74), hover=(125, 76, 90))
    return {'again': b_again, 'menu': b_menu}


def draw_promo_picker(screen, board, fonts):
    fpiece, fm, fsm = fonts
    shade = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
    shade.fill((10, 10, 22, 110))
    screen.blit(shade, (0, 0))

    card = pygame.Rect(BOARD_SIZE // 2 - 220, HEIGHT // 2 - 130, 440, 250)
    draw_card(screen, card, fill=C_SURFACE)
    text_at(screen, "PIYON TERFISI - TASINI SEC", (card.centerx, card.y + 16),
            fm, C_ACCENT, 'midtop')

    choices = [(chess.QUEEN, 'Vezir', '♛'), (chess.ROOK, 'Kale', '♜'),
               (chess.BISHOP, 'Fil', '♝'), (chess.KNIGHT, 'At', '♞')]
    bw, bh, gap = 92, 118, 14
    x0 = card.centerx - (4 * bw + 3 * gap) // 2
    rects = {}
    for i, (pt, nm, sym) in enumerate(choices):
        r = pygame.Rect(x0 + i * (bw + gap), card.y + 56, bw, bh)
        hov = r.collidepoint(pygame.mouse.get_pos())
        draw_card(screen, r, fill=C_CARD_HI if hov else C_CARD,
                  border=lerp(C_BORDER, C_ACCENT, 0.5) if hov else C_BORDER,
                  radius=10, shadow=False)
        col = (250, 250, 250) if board.turn == chess.WHITE else (25, 25, 30)
        outline = fpiece.render(sym, True, (10, 10, 14))
        colored = fpiece.render(sym, True, col)
        px = r.centerx - outline.get_width() // 2
        py = r.y + 12
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            screen.blit(outline, (px + dx, py + dy))
        screen.blit(colored, (px, py))
        text_at(screen, nm, (r.centerx, r.bottom - 30), fsm, C_TEXT, 'midtop')
        rects[pt] = r
    text_at(screen, "vazgecmek icin kartin disina tikla",
            (card.centerx, card.bottom - 26), fsm, C_FAINT, 'midtop')
    return rects


# ===================== OYUN =====================
def play_game(screen, brain, history, player_color, time_limit, fonts):
    fpiece, fm, fsm = fonts
    board = chess.Board()
    engine = Engine(brain, time_limit)
    game_num = brain.games_played + 1
    sel, targets = None, []
    last_moves = []
    move_records = []
    all_moves = []
    game_over = False
    thinking = False
    done = [True]
    ai_res = {'m': None, 'info': None}
    move_scroll = [0]
    learning_done = [False]
    learn_loss = [0.0]
    ov_rects = [None]
    promo = [None]
    promo_rects = [None]

    def worker(b):
        try:
            mv, sc, dep, nodes = engine.search(b)
            ai_res['m'] = mv
            ai_res['info'] = {'depth': dep, 'score': sc, 'nodes': nodes}
        except Exception:
            ai_res['m'] = None
            ai_res['info'] = None
        finally:
            done[0] = True

    def learn_worker(recs, mvs, out_val, pcol):
        try:
            loss = learn_from_human_game(brain, recs, mvs, out_val, pcol)
            append_history(history, {
                'game_num': brain.games_played,
                'mode': 'human',
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'player_color': 'white' if pcol == chess.WHITE else 'black',
                'time_limit': time_limit,
                'result': result_text_holder[0],
                'result_val': out_val,
                'num_moves': len(mvs),
                'moves_san': [r['san'] for r in recs],
                'move_details': recs,
                'learning': loss,
            })
            learn_loss[0] = loss
        except Exception as e:
            print("Ogrenme hatasi:", e)
        finally:
            learning_done[0] = True

    result_text_holder = [""]

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return
                if e.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
            if e.type == pygame.MOUSEBUTTONDOWN:
                if promo[0] is not None:
                    if promo_rects[0]:
                        for pt, r in promo_rects[0].items():
                            if r.collidepoint(e.pos):
                                frm, to = promo[0]
                                mv = chess.Move(frm, to, promotion=pt)
                                if mv in board.legal_moves:
                                    san = board.san(mv)
                                    board.push(mv); all_moves.append(mv)
                                    move_records.append({
                                        'san': san,
                                        'side': 'white' if player_color == chess.WHITE else 'black',
                                        'thinking': None})
                                    last_moves = [mv]
                                    move_scroll[0] = 999
                                break
                    promo[0] = None
                    continue
                if game_over and ov_rects[0] is not None and learning_done[0]:
                    if ov_rects[0]['again'].collidepoint(e.pos):
                        return 'again'
                    if ov_rects[0]['menu'].collidepoint(e.pos):
                        return
                if e.button in (4, 5) and e.pos[0] >= PANEL_X:
                    d = -1 if e.button == 4 else 1
                    move_scroll[0] = max(0, move_scroll[0] + d)
                    continue
                if e.pos[0] >= PANEL_X:
                    if R_BTN_UNDO.collidepoint(e.pos):
                        if (not game_over and not thinking
                                and board.turn == player_color and len(all_moves) >= 2):
                            board.pop(); board.pop()
                            all_moves.pop(); all_moves.pop()
                            move_records.pop(); move_records.pop()
                            last_moves = [all_moves[-1]] if all_moves else []
                            sel, targets = None, []
                            move_scroll[0] = 999
                        continue
                    if R_BTN_MENU.collidepoint(e.pos):
                        return
                    continue
                if pygame.Rect(WIDTH - 54, 16, 36, 36).collidepoint(e.pos):
                    return
                if game_over or thinking or board.turn != player_color:
                    continue
                if e.pos[0] >= BOARD_SIZE:
                    continue
                sq = xy_to_sq(*e.pos)
                if sq is None:
                    continue
                if sel is None:
                    p = board.piece_at(sq)
                    if p and p.color == player_color:
                        sel = sq
                        targets = [m.to_square for m in board.legal_moves
                                   if m.from_square == sq]
                else:
                    if (is_promotion_move(board, sel, sq)
                            and chess.Move(sel, sq, promotion=chess.QUEEN)
                            in board.legal_moves):
                        promo[0] = (sel, sq)
                        sel, targets = None, []
                        continue
                    mv = chess.Move(sel, sq)
                    if mv in board.legal_moves:
                        san = board.san(mv)
                        board.push(mv); all_moves.append(mv)
                        move_records.append({'san': san,
                                             'side': 'white' if player_color == chess.WHITE else 'black',
                                             'thinking': None})
                        last_moves = [mv]
                        sel, targets = None, []
                        move_scroll[0] = 999
                    else:
                        p = board.piece_at(sq)
                        if p and p.color == player_color:
                            sel = sq
                            targets = [m.to_square for m in board.legal_moves
                                       if m.from_square == sq]
                        else:
                            sel, targets = None, []

        if not game_over and board.is_game_over():
            game_over = True
            o = board.outcome()
            if o is None or o.winner is None:
                result_text, out_val = "Berabere", 0
            elif o.winner == chess.WHITE:
                result_text, out_val = "Beyaz kazandi!", 1
            else:
                result_text, out_val = "Siyah kazandi!", -1
            result_text_holder[0] = result_text
            threading.Thread(target=learn_worker,
                             args=(list(move_records), list(all_moves), out_val,
                                   player_color),
                             daemon=True).start()

        if not game_over and board.turn != player_color and not thinking and done[0]:
            thinking = True
            done[0] = False
            ai_res['m'] = None
            ai_res['info'] = None
            engine.info = {'depth': 0, 'nodes': 0, 'score': 0.0}
            threading.Thread(target=worker, args=(board.copy(),), daemon=True).start()

        if thinking and done[0]:
            m, info = ai_res['m'], ai_res['info']
            if m is None:
                legal = list(board.legal_moves)
                if legal:
                    m, info = random.choice(legal), None
            if m is not None:
                san = board.san(m)
                board.push(m); all_moves.append(m)
                move_records.append({'san': san,
                                     'side': 'white' if player_color == chess.BLACK else 'black',
                                     'thinking': info})
                last_moves = [m]
                move_scroll[0] = 999
            thinking = False

        hover_sq = None
        if (not game_over and not thinking and board.turn == player_color):
            mp = pygame.mouse.get_pos()
            if mp[0] < BOARD_SIZE:
                hover_sq = xy_to_sq(*mp)

        draw_game_frame(screen, brain, board, sel, targets, last_moves, move_records,
                        player_color, game_num, thinking, engine.info,
                        (fpiece, fm, fsm), move_scroll, hover_sq)

        if promo[0] is not None:
            promo_rects[0] = draw_promo_picker(screen, board, fonts)

        if game_over:
            ov_rects[0] = draw_result_overlay(screen, brain, result_text_holder[0],
                                              learn_loss[0], fonts,
                                              not learning_done[0])

        pygame.display.flip()


# ===================== ANA =====================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE | pygame.SCALED)
    pygame.display.set_caption("Satranc Machine v5.2")

    fb = pygame.font.SysFont('segoeui', 46, bold=True)
    fm = pygame.font.SysFont('segoeui', 22)
    fsm = pygame.font.SysFont('segoeui', 17)
    fpiece = piece_font(58)
    fonts = (fpiece, fm, fsm)

    brain = Brain.load()
    history = load_history()

    while True:
        choice = menu(screen, (fb, fm, fsm), brain)
        if choice is None:
            break
        if choice[0] == 'train':
            train_screen(screen, brain, history, fonts)
        elif choice[0] == 'inspect':
            viewer(screen, brain, history, (fb, fm, fsm))
        else:
            color, tlim = choice
            while play_game(screen, brain, history, color, tlim, fonts) == 'again':
                pass

    pygame.quit()
    sys.exit()


# ===================== EKRAN GORUNTULERI =====================
def render_screenshots(out_dir="shots"):
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    fb = pygame.font.SysFont('segoeui', 46, bold=True)
    fm = pygame.font.SysFont('segoeui', 22)
    fsm = pygame.font.SysFont('segoeui', 17)
    fpiece = piece_font(58)
    fonts3 = (fb, fm, fsm)

    brain = Brain.load()
    history = load_history()

    if not brain.nn.loss_history:
        brain.nn.loss_history = [0.02 * math.exp(-i / 40) + random.random() * 0.002
                                 for i in range(80)]
        brain.nn.training_steps = 4200
    if not history:
        history = [{
            'game_num': 1, 'mode': 'human', 'timestamp': '2026-11-07T12:00:00',
            'player_color': 'white', 'result': 'Beyaz kazandi!', 'result_val': 1,
            'num_moves': 24, 'moves_san': ['e4', 'e5', 'Nf3', 'Nc6'],
            'move_details': [], 'learning': 0.00421,
        }]

    draw_menu_frame(screen, fonts3, brain, 2)
    pygame.image.save(screen, os.path.join(out_dir, "1_menu.png"))

    board = chess.Board()
    sans = ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6', 'Ng5', 'd5', 'exd5', 'Nxd5']
    for san in sans:
        board.push_san(san)
    recs = [{'san': s, 'side': 'white' if i % 2 == 0 else 'black',
             'thinking': {'depth': 6, 'score': -40, 'nodes': 15000} if i % 2 == 1 else None}
            for i, s in enumerate(sans)]
    draw_game_frame(screen, brain, board, None, [], [], recs, chess.WHITE,
                    2, False, {'depth': 0, 'nodes': 0, 'score': 0.0},
                    (fpiece, fm, fsm), [0], None)
    pygame.image.save(screen, os.path.join(out_dir, "2_game.png"))

    sel_sq = chess.G1
    draw_game_frame(screen, brain, board, sel_sq,
                    [m.to_square for m in board.legal_moves if m.from_square == sel_sq],
                    [], recs, chess.WHITE, 2, False,
                    {'depth': 0, 'nodes': 0, 'score': 0.0}, (fpiece, fm, fsm), [0], sel_sq)
    pygame.image.save(screen, os.path.join(out_dir, "3_game_selected.png"))

    stats = {'games': 37, 'white_wins': 15, 'black_wins': 14, 'draws': 8,
             'total_plies': 2600, 'avg_len': 70.2, 'last_loss': 0.00312,
             'cur_len': 44, 'cur_move': 'Nf3', 'cur_depth': 3, 'steps': 4200}
    draw_train_frame(screen, fonts3, brain, stats, 1)
    pygame.image.save(screen, os.path.join(out_dir, "4_train.png"))

    for tab in range(4):
        draw_viewer_frame(screen, fonts3, brain, history, tab, 0, False)
        pygame.image.save(screen, os.path.join(out_dir, f"5_view{tab}.png"))

    mboard = chess.Board()
    for san in ('f3', 'e5', 'g4', 'Qh4'):
        mboard.push_san(san)
    draw_game_frame(screen, brain, mboard, None, [], [], recs, chess.WHITE,
                    2, False, {'depth': 0, 'nodes': 0, 'score': 0.0},
                    (fpiece, fm, fsm), [0], None)
    draw_result_overlay(screen, brain, "Siyah kazandi!", 0.00421,
                        (fpiece, fm, fsm), False)
    pygame.image.save(screen, os.path.join(out_dir, "6_gameover.png"))

    pboard = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    draw_game_frame(screen, brain, pboard, chess.A7,
                    [m.to_square for m in pboard.legal_moves
                     if m.from_square == chess.A7],
                    [], recs, chess.WHITE, 2, False,
                    {'depth': 0, 'nodes': 0, 'score': 0.0},
                    (fpiece, fm, fsm), [0], None)
    draw_promo_picker(screen, pboard, (fpiece, fm, fsm))
    pygame.image.save(screen, os.path.join(out_dir, "7_promotion.png"))

    pygame.quit()
    print(f"Ekran goruntuleri '{out_dir}' klasorune kaydedildi.")


# ===================== TEST =====================
def run_self_test():
    import tempfile
    ok_all = True

    def check(name, cond):
        nonlocal ok_all
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")
        ok_all = ok_all and cond

    tmp = tempfile.mkdtemp()
    mp = os.path.join(tmp, 'm.npz')
    sp = os.path.join(tmp, 's.json')

    nn = NeuralNetwork()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, NeuralNetwork.N_IN)).astype(np.float32)
    T = rng.uniform(-800, 800, 64).astype(np.float32)
    l0 = nn.train_batch(X, T)
    for _ in range(60):
        l1 = nn.train_batch(X, T)
    check("Sinir agi ogreniyor (loss dustu)", l1 < l0)

    b = chess.Board()
    for san in ('f3', 'e5', 'g4'):
        b.push_san(san)
    brain = Brain(mp, sp)
    eng = Engine(brain, time_limit=3.0)
    mv, sc, dep, nodes = eng.search(b)
    check("1 hamlede mat bulundu (Qh4#)", b.san(mv) == 'Qh4#' and sc > MATE - 1000)

    mv, sc, dep, nodes = Engine(brain, 1.0).search(chess.Board())
    check("Baslangictan yasal hamle", mv in chess.Board().legal_moves)

    brain.save()
    brain2 = Brain.load(mp, sp)
    check("Kaydet/yukle (agirliklar ayni)",
          np.allclose(brain.nn.W1, brain2.nn.W1) and
          np.allclose(brain.nn.W3, brain2.nn.W3) and
          brain2.nn.training_steps == brain.nn.training_steps)
    check("Kaydet/yukle (Adam m/v durumu ayni)",
          np.allclose(brain.nn.m['W1'], brain2.nn.m['W1']) and
          np.allclose(brain.nn.v['W3'], brain2.nn.v['W3']) and
          brain2.nn.t == brain.nn.t)

    nn_sparse = NeuralNetwork()
    maxdiff = 0.0
    for _ in range(10):
        rb = chess.Board()
        for _ in range(int(rng.integers(4, 30))):
            legal = list(rb.legal_moves)
            if not legal:
                break
            rb.push(random.choice(legal))
        x = nn_sparse.features(rb).reshape(1, -1)
        a1 = np.maximum(0.0, x @ nn_sparse.W1 + nn_sparse.b1)
        a2 = np.maximum(0.0, a1 @ nn_sparse.W2 + nn_sparse.b2)
        dense = float(np.tanh(a2 @ nn_sparse.W3 + nn_sparse.b3)[0, 0]) * 1000.0
        maxdiff = max(maxdiff, abs(dense - nn_sparse.forward(rb)))
    check(f"Seyrek forward = yogun forward (fark {maxdiff:.1e})", maxdiff < 1e-3)

    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    mv, sc, dep, nodes = Engine(brain, 1.0).search(stale)
    check("Pat pozisyonunda arama None doner", mv is None)

    kk = chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
    mv, sc, dep, nodes = Engine(brain, 1.0).search(kk)
    check("Yetersiz materyal: deger 0", mv is not None and abs(sc) < 1e-6)

    rm = chess.Board("3k4/7R/3K4/8/8/8/8/8 w - - 0 1")
    mv, sc, dep, nodes = Engine(brain, 2.0).search(rm)
    check("Kale ile 1 hamlede mat (Rh8#)",
          mv is not None and rm.san(mv) == 'Rh8#' and sc > MATE - 1000)

    epb = chess.Board()
    for s in ('e4', 'a6', 'e5', 'd5'):
        epb.push_san(s)
    ep_move = chess.Move(chess.E5, chess.D6)
    assert ep_move in epb.legal_moves
    mv, sc, dep, nodes = Engine(brain, 1.0).search(epb)
    check("En passant pozisyonunda arama calisiyor",
          mv is not None and mv in epb.legal_moves)

    b2 = chess.Board()
    b2.push_san('e4'); b2.push_san('e5')
    e1 = Engine(brain, time_limit=10.0, max_depth=3)
    m1, s1, _, _ = e1.search(b2.copy())
    e2 = Engine(brain, time_limit=10.0, max_depth=3)
    m2, s2, _, _ = e2.search(b2.copy())
    check("Arama belirleyici (TT/onbellek)", m1 == m2 and abs(s1 - s2) < 1e-6)

    mp2 = os.path.join(tmp, 'm2.npz')
    sp2 = os.path.join(tmp, 's2.json')
    bt = Brain(mp2, sp2)
    samples_t = [(chess.Board().fen(), 50.0, 10)]
    bt.make_targets(samples_t, 1, 20, gamma=0.97, w_search=0.5)
    expected = 0.5 * 50.0 + 0.5 * (1000.0 * 0.97 ** (20 - 10))
    got = bt.buffer[-1][1]
    check(f"make_targets formulu dogru ({got:.2f} = {expected:.2f})",
          abs(got - expected) < 1e-6)

    h_test = []
    for i in range(600):
        append_history(h_test, {'game_num': i}, os.path.join(tmp, 'h2.json'))
    check("Gecmis en fazla 500 kayit", len(h_test) == 500)

    start_pos = chess.Board()
    check("Klasik degerlendirme: baslangic dengeli (0)",
          abs(classical_eval(start_pos)) < 1e-9)
    qless = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    check("Klasik degerlendirme: vezir avantajini gorur",
          classical_eval(qless) > 500)
    fb = Brain(os.path.join(tmp, 'm3.npz'), os.path.join(tmp, 's3.json'))
    check("Hibrit: hic ogrenmemis makine tamamen klasik",
          abs(fb.eval_white(start_pos) - classical_eval(start_pos)) < 1e-9)
    fb.nn.training_steps = NN_TRUST_STEPS * 2
    check("Hibrit: olgun makine tamamen kendi degerlendirmesi",
          abs(fb.eval_white(start_pos) - fb.nn.forward(start_pos)) < 1e-9)

    pb = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    check("Terfi: vezir/kale/fil/at hamleleri yasal",
          chess.Move(chess.A7, chess.A8, chess.QUEEN) in pb.legal_moves and
          chess.Move(chess.A7, chess.A8, chess.KNIGHT) in pb.legal_moves)
    check("Terfi tetikleyici: piyon son siraya giderse",
          is_promotion_move(pb, chess.A7, chess.A8)
          and not is_promotion_move(pb, chess.H2, chess.H4))

    stats = {'games': 0, 'white_wins': 0, 'black_wins': 0, 'draws': 0,
             'total_plies': 0, 'avg_len': 0.0, 'last_loss': 0.0, 'cur_len': 0,
             'cur_move': '-', 'cur_depth': 0, 'steps': 0}
    lock = threading.Lock()
    stop = threading.Event()
    t0 = time.time()
    selfplay_worker(brain, [], stats, lock, stop, depth=2, time_limit=0.05,
                    max_plies=30, max_games=1, history_path=os.path.join(tmp, 'h.json'))
    check("Self-play oyunu + ogrenme tamamlandi",
          stats['games'] == 1 and len(brain.buffer) > 0 and stats['last_loss'] > 0)
    print(f"       (self-play {time.time()-t0:.1f}s, tampon: {len(brain.buffer)})")

    board = chess.Board()
    recs, mvs = [], []
    for san in ('e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6'):
        mv = board.parse_san(san)
        recs.append({'san': san, 'side': 'white' if board.turn == chess.WHITE else 'black',
                     'thinking': None})
        mvs.append(mv)
        board.push(mv)
    loss = learn_from_human_game(brain, recs, mvs, 1, chess.BLACK)
    check("Insan oyunundan ogrenme", loss > 0 and brain.games_played == 1)

    print("\n" + ("TUM TESTLER GECTI - Makine saglikli kuruldu." if ok_all
                  else "BAZI TESTLER BASARISIZ!"))
    return ok_all


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('--test', '-t'):
        sys.exit(0 if run_self_test() else 1)
    if len(sys.argv) > 1 and sys.argv[1] == '--shots':
        render_screenshots(sys.argv[2] if len(sys.argv) > 2 else 'shots')
        sys.exit(0)
    main()