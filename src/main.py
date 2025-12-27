"""
Dayakattai (4-player) pygame version
-----------------------------------
- Click "ROLL" to roll the Dayakattai dice (0/1/2/3 on each die).
- Rolls that score 1, 5, 6, or 12 add extra rolls to the queue.
- Click one of your coins to apply a selected move (you can use multiple rolls on the same coin).
- Click a move in the list to choose which roll to use.
- Rules are shown in the in-game Rules page.

Install/run:
  pip install pygame
  python dayakattai_pygame.py
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set

import pygame

# ---------------------------
# Rules / dice
# ---------------------------
FACES = (0, 1, 2, 3)

MOVE_TABLE: Dict[Tuple[int, int], int] = {
    (0, 0): 12,
    (0, 1): 1, (1, 0): 1,  # Daayam
    (0, 2): 2, (2, 0): 2,
    (0, 3): 3, (3, 0): 3,
    (1, 1): 2,
    (1, 2): 3, (2, 1): 3,
    (1, 3): 4, (3, 1): 4,
    (2, 2): 4,
    (2, 3): 5, (3, 2): 5,
    (3, 3): 6,
}

EXTRA_ROLL_ON = {1, 5, 6, 12}

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MENU_MUSIC_PATH = ASSETS_DIR / "Medieval Music  Cobblestone Village.mp3"
GAME_MUSIC_PATH = ASSETS_DIR / "Medieval Music  Wild Boar's Inn.mp3"


def roll_dayakattai() -> Tuple[int, int, int, bool]:
    a = random.choice(FACES)
    b = random.choice(FACES)
    move = MOVE_TABLE[(a, b)]
    is_daayam = (a, b) in {(0, 1), (1, 0)}
    return a, b, move, is_daayam


def load_sound(path: Path) -> Optional[pygame.mixer.Sound]:
    try:
        return pygame.mixer.Sound(str(path))
    except pygame.error:
        print(f"Sound disabled (failed to load): {path}")
        return None


# ---------------------------
# Board mapping (family spiral board)
# ---------------------------
# The family board is a *single spiral path* into the center.
# Coin.pos indexes into PATH. Finished = last cell of PATH (center).

GRID_SIZE = 9  # <- adjust to match your paper board grid (often 11 or 13)

def build_spiral_path(n: int) -> List[Tuple[int, int]]:
    """
    Build a square spiral path ending at center.
    This version starts at the OUTER bottom-left corner and goes:
      RIGHT (bottom edge) -> UP (right edge) -> LEFT (top edge) -> DOWN (left edge),
    then continues inward.

    If your real-life direction is opposite, you can flip PATH = PATH[::-1].
    """
    path: List[Tuple[int, int]] = []
    left = 0
    right = n - 1
    top = 0
    bottom = n - 1

    while left <= right and top <= bottom:
        # bottom row: left -> right
        for c in range(left, right + 1):
            path.append((bottom, c))
        bottom -= 1
        if top > bottom:
            break

        # right col: bottom -> top
        for r in range(bottom, top - 1, -1):
            path.append((r, right))
        right -= 1
        if left > right:
            break

        # top row: right -> left
        for c in range(right, left - 1, -1):
            path.append((top, c))
        top += 1
        if top > bottom:
            break

        # left col: top -> bottom
        for r in range(top, bottom + 1):
            path.append((r, left))
        left += 1

    # De-dupe while preserving order (defensive; spiral generation can repeat at center)
    seen = set()
    out = []
    for rc in path:
        if rc not in seen:
            out.append(rc)
            seen.add(rc)
    return out

PATH: List[Tuple[int, int]] = build_spiral_path(GRID_SIZE)

# If the spiral direction is backwards compared to your family arrows, toggle this:
REVERSE_PATH = False
if REVERSE_PATH:
    PATH = PATH[::-1]

TRACK_LEN = len(PATH)
CENTER_CELL = PATH[-1]

# Start squares (edit these to match your paper)
# These MUST be cells that exist in PATH.
# Order is clockwise from bottom.
START_CELL = {
    0: (8, 4),  # bottom mountain X
    1: (4, 0),  # left mountain X
    2: (0, 4),  # top mountain X
    3: (4, 8),  # right mountain X
}
START_INDEX = {pi: PATH.index(rc) for pi, rc in START_CELL.items()}

# Safe squares (X cells) — edit to match your board.
# Seeded with side-midpoints + center, plus explicit PATH indices.
SAFE_INDEXES = {32, 38, 44, 50, 56, 60, 64, 68}
SAFE_CELLS = {
    (0, 4),  # top mountain
    (4, 8),  # right mountain
    (8, 4),  # bottom mountain
    (4, 0),  # left mountain
    CENTER_CELL,
}
for idx in SAFE_INDEXES:
    if 0 <= idx < len(PATH):
        SAFE_CELLS.add(PATH[idx])


def ring_index(rc: Tuple[int, int]) -> int:
    r, c = rc
    return min(r, c, GRID_SIZE - 1 - r, GRID_SIZE - 1 - c)


OUTER_RING_LEN = sum(1 for rc in PATH if ring_index(rc) == 0)


def enters_inner_layer(current_pos: Optional[int], target_pos: int) -> bool:
    if current_pos is None:
        return False
    return ring_index(PATH[current_pos]) == 0 and ring_index(PATH[target_pos]) > 0


# ---------------------------
# Game state / engine
# ---------------------------
TOTAL_PLAYERS = 4
COINS_PER_PLAYER = 6


@dataclass
class Coin:
    pos: Optional[int] = None
    outer_steps: int = 0

    @property
    def is_home(self) -> bool:
        return self.pos is None

    @property
    def is_finished(self) -> bool:
        return self.pos == (TRACK_LEN - 1)  # center


@dataclass
class Player:
    name: str
    idx: int
    coins: List[Coin]
    has_cut: bool = False
    has_opened_home: bool = False
    has_full_lap: bool = False


def all_finished(p: Player) -> bool:
    return all(c.is_finished for c in p.coins)


def can_exit_home(player: Player, move: int) -> bool:
    if move == 1:
        return True
    if move == 5 and player.has_opened_home:
        return True
    return False


def is_outer_ring_pos(pos: int) -> bool:
    return 0 <= pos < OUTER_RING_LEN


def can_enter_inner(player: Player, coin: Coin, move: int) -> bool:
    if coin.pos is None or not is_outer_ring_pos(coin.pos):
        return True
    if not player.has_cut:
        return False
    total = coin.outer_steps + move
    return player.has_full_lap or total >= OUTER_RING_LEN


def legal_move_coin(player: Player, coin: Coin, move: int) -> bool:
    if coin.is_home:
        return can_exit_home(player, move)
    if coin.is_finished:
        return False
    target = target_pos_for_move(player, coin, move)
    return target is not None


def occupies(players: List[Player], path_pos: int) -> List[Tuple[int, int]]:
    hits = []
    for pi, pl in enumerate(players):
        for ci, coin in enumerate(pl.coins):
            if coin.pos == path_pos and coin.pos is not None and 0 <= coin.pos < TRACK_LEN:
                hits.append((pi, ci))
    return hits


def can_land_on(players: List[Player], player_idx: int, target_pos: int) -> bool:
    cell = PATH[target_pos]
    if cell in SAFE_CELLS:
        return True
    for opi, _ in occupies(players, target_pos):
        if opi == player_idx:
            return False
    return True


def target_pos_for_move(player: Player, coin: Coin, move: int) -> Optional[int]:
    if coin.is_home:
        return START_INDEX[player.idx]
    if coin.pos is None:
        return None
    new_pos = coin.pos + move
    if is_outer_ring_pos(coin.pos):
        if new_pos < OUTER_RING_LEN:
            return new_pos
        if can_enter_inner(player, coin, move):
            return new_pos if new_pos <= (TRACK_LEN - 1) else None
        return (coin.pos + move) % OUTER_RING_LEN
    if 0 <= new_pos < TRACK_LEN:
        return new_pos
    return None


def would_cut(players: List[Player], current_player_idx: int, coin_idx: int, move: int) -> bool:
    p = players[current_player_idx]
    coin = p.coins[coin_idx]
    target = target_pos_for_move(p, coin, move)
    if target is None:
        return False
    cell = PATH[target]
    if cell in SAFE_CELLS:
        return False
    if not can_land_on(players, current_player_idx, target):
        return False
    for opi, _ in occupies(players, target):
        if opi != current_player_idx:
            return True
    return False


def is_move_legal(players: List[Player], player_idx: int, coin_idx: int, move: int) -> bool:
    player = players[player_idx]
    coin = player.coins[coin_idx]
    if not legal_move_coin(player, coin, move):
        return False
    target = target_pos_for_move(player, coin, move)
    if target is None:
        return False
    if not can_land_on(players, player_idx, target):
        return False
    return True


def apply_move(
    players: List[Player],
    current_player_idx: int,
    coin_idx: int,
    move: int,
) -> Tuple[bool, bool]:
    p = players[current_player_idx]
    coin = p.coins[coin_idx]

    if not is_move_legal(players, current_player_idx, coin_idx, move):
        return False, False

    did_cut = False
    target = target_pos_for_move(p, coin, move)
    if target is None:
        return False, False

    # Enter from home
    if coin.is_home:
        coin.pos = START_INDEX[p.idx]
        coin.outer_steps = 0
        p.has_opened_home = True
    else:
        assert coin.pos is not None
        if is_outer_ring_pos(coin.pos):
            total = coin.outer_steps + move
            if total >= OUTER_RING_LEN:
                p.has_full_lap = True
            coin.outer_steps = total % OUTER_RING_LEN
        coin.pos = target

    # Cutting
    assert coin.pos is not None
    cell = PATH[coin.pos]
    if cell not in SAFE_CELLS:
        occ = occupies(players, coin.pos)
        for (opi, oci) in occ:
            if opi != current_player_idx:
                players[opi].coins[oci].pos = None
                players[opi].coins[oci].outer_steps = 0
                did_cut = True
        if did_cut:
            p.has_cut = True

    return True, did_cut


# ---------------------------
# pygame UI
# ---------------------------
W, H = 1400, 860

CELL = 80  # adjust for GRID_SIZE; 60 fits 11 pretty well
BOARD_ORIGIN = (80, 60)

PANEL_X = 860
PANEL_Y = 80
PANEL_W = 320

COIN_RADIUS = 14
MOVES_HEADER_H = 28
MOVES_LINE_H = 22
MOVES_W = PANEL_W - 40
MOVES_PANEL_Y = PANEL_Y + 230

MENU_BTN_W = 260
MENU_BTN_H = 54

HOME_ANCHORS = [
    (PANEL_X, 470),
    (PANEL_X + 140, 470),
    (PANEL_X, 570),
    (PANEL_X + 140, 570),
]


def cell_center(rc: Tuple[int, int]) -> Tuple[int, int]:
    r, c = rc
    x0, y0 = BOARD_ORIGIN
    cx = x0 + c * CELL + CELL // 2
    cy = y0 + r * CELL + CELL // 2
    return cx, cy


def draw_x(surface: pygame.Surface, rect: pygame.Rect, color: pygame.Color, width: int = 3) -> None:
    pygame.draw.line(surface, color, rect.topleft, rect.bottomright, width)
    pygame.draw.line(surface, color, rect.topright, rect.bottomleft, width)


def draw_dayakattai_die(surface: pygame.Surface, rect: pygame.Rect, value: int) -> None:
    brass = pygame.Color(191, 158, 95)
    brass_dark = pygame.Color(120, 90, 40)
    brass_light = pygame.Color(220, 190, 120)
    pip = pygame.Color(45, 35, 18)

    pygame.draw.rect(surface, brass, rect, border_radius=10)
    pygame.draw.rect(surface, brass_dark, rect, 2, border_radius=10)

    pygame.draw.line(surface, brass_light, (rect.left + 3, rect.top + 3), (rect.right - 3, rect.top + 3), 3)
    pygame.draw.line(surface, brass_light, (rect.left + 3, rect.top + 3), (rect.left + 3, rect.bottom - 3), 3)
    pygame.draw.line(surface, brass_dark, (rect.left + 3, rect.bottom - 3), (rect.right - 3, rect.bottom - 3), 3)
    pygame.draw.line(surface, brass_dark, (rect.right - 3, rect.top + 3), (rect.right - 3, rect.bottom - 3), 3)

    cx, cy = rect.center
    dx = rect.width // 5
    dy = rect.height // 4
    positions = {"center": (cx, cy), "tl": (cx - dx, cy - dy), "br": (cx + dx, cy + dy)}

    if value == 0:
        dots = []
    elif value == 1:
        dots = [positions["center"]]
    elif value == 2:
        dots = [positions["tl"], positions["br"]]
    else:
        dots = [positions["tl"], positions["center"], positions["br"]]

    pip_r = max(3, min(rect.width, rect.height) // 10)
    for x, y in dots:
        pygame.draw.circle(surface, pip, (x, y), pip_r)


def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    font: pygame.font.Font,
    fill: pygame.Color,
    outline: pygame.Color,
    text_color: pygame.Color,
) -> None:
    pygame.draw.rect(surface, fill, rect, border_radius=10)
    pygame.draw.rect(surface, outline, rect, 2, border_radius=10)
    img = font.render(label, True, text_color)
    surface.blit(img, (rect.centerx - img.get_width() // 2, rect.centery - img.get_height() // 2))


def coin_draw_positions_at_cell(cell_rc: Tuple[int, int], coins_here: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cx, cy = cell_center(cell_rc)
    n = len(coins_here)
    if n == 1:
        return [(cx, cy)]
    pts = []
    spread = 18
    for i in range(n):
        angle = (i / n) * 6.283185307
        v = pygame.math.Vector2(1, 0).rotate_rad(angle)
        pts.append((int(cx + spread * v.x), int(cy + spread * v.y)))
    return pts


def home_anchor(player_idx: int) -> Tuple[int, int]:
    if 0 <= player_idx < len(HOME_ANCHORS):
        return HOME_ANCHORS[player_idx]
    return (PANEL_X, 520)


def render_text(surface: pygame.Surface, text: str, pos: Tuple[int, int], font: pygame.font.Font, color=(25, 25, 25)) -> None:
    img = font.render(text, True, color)
    surface.blit(img, pos)


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        test = f"{current} {word}"
        if font.size(test)[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


async def main() -> None:
    pygame.init()
    try:
        pygame.mixer.init()
    except pygame.error:
        print("Sound disabled (mixer init failed).")
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Dayakatt.AI (4P) - pygame")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 18)
    font_big = pygame.font.SysFont("arial", 24, bold=True)
    font_small = pygame.font.SysFont("arial", 16)
    font_title = pygame.font.SysFont("arial", 36, bold=True)

    bg = pygame.Color(245, 245, 245)
    grid_line = pygame.Color(180, 180, 180)
    black = pygame.Color(25, 25, 25)
    safe_fill = pygame.Color(230, 230, 230)

    player_defs = [
        ("Red", pygame.Color(220, 60, 60)),
        ("Green", pygame.Color(60, 160, 80)),
        ("Yellow", pygame.Color(230, 200, 30)),
        ("Blue", pygame.Color(70, 110, 210)),
    ]
    player_colors = [color for _, color in player_defs]

    sounds: Dict[str, Optional[pygame.mixer.Sound]] = {}
    mixer_ready = pygame.mixer.get_init() is not None
    sfx_enabled = mixer_ready
    bgm_enabled = mixer_ready
    if mixer_ready:
        sounds = {
            "roll": load_sound(ASSETS_DIR / "Dayakattai Sound FX.wav"),
            "move": load_sound(ASSETS_DIR / "Coin Placing FX.wav"),
        }
    current_music: Optional[Path] = None
    music_failed: Set[Path] = set()

    def play_sound(key: str) -> None:
        if not sfx_enabled:
            return
        snd = sounds.get(key)
        if snd is not None:
            snd.play()

    def sync_music(phase_name: str) -> None:
        nonlocal current_music
        if not bgm_enabled or not pygame.mixer.get_init():
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            current_music = None
            return
        target = MENU_MUSIC_PATH if phase_name in {"MENU", "RULES"} else GAME_MUSIC_PATH
        if target in music_failed:
            pygame.mixer.music.stop()
            current_music = None
            return
        if current_music == target:
            return
        try:
            pygame.mixer.music.load(str(target))
            pygame.mixer.music.play(-1)
            current_music = target
        except pygame.error:
            print(f"Sound disabled (failed to load music): {target}")
            music_failed.add(target)
            current_music = None

    def toggle_bgm() -> None:
        nonlocal bgm_enabled
        if not pygame.mixer.get_init():
            bgm_enabled = False
            return
        bgm_enabled = not bgm_enabled
        sync_music(phase)

    def toggle_sfx() -> None:
        nonlocal sfx_enabled
        if not pygame.mixer.get_init():
            sfx_enabled = False
            return
        sfx_enabled = not sfx_enabled

    human_players = 1

    def make_players(humans: int) -> List[Player]:
        humans = max(1, min(TOTAL_PLAYERS, humans))
        players_out: List[Player] = []
        for i, (color_name, _) in enumerate(player_defs):
            role = "Human" if i < humans else "CPU"
            players_out.append(
                Player(
                    f"P{i + 1} ({color_name}, {role})",
                    idx=i,
                    coins=[Coin() for _ in range(COINS_PER_PLAYER)],
                )
            )
        return players_out

    rules_text = [
        "Roll the dice on your turn. If the value is 1 (thayam), 5, 6, or 12, you move a coin and get another roll.",
        "At the start, a player must roll 1 (thayam) to place the first coin on their home. After that, cut coins can re-enter on 1 or 5.",
        "On a normal square (non-safe), only one coin can stand. Safe squares (marked with a cross) can stack.",
        "You can cut an opponent's coin if it is not in a safe square. The cut coin goes out and re-enters per rule 2. Cutting grants an extra roll.",
        "You must cut at least one opponent coin and complete a full outer lap to enter the inner layer.",
    ]

    players: List[Player] = []
    turn = 0
    moves_queue: List[Tuple[int, bool, Tuple[int, int]]] = []
    current_move_i = 0
    phase = "MENU"

    roll_btn = pygame.Rect(PANEL_X, PANEL_Y + 40, 180, 44)
    undo_btn = pygame.Rect(PANEL_X, roll_btn.bottom + 8, 180, 34)

    selected_coin_idx: Optional[int] = None
    last_roll_info = ""
    undo_stack: List[Dict[str, object]] = []

    rolling = False
    roll_end_ms = 0
    anim_a, anim_b = 0, 0

    ai_players: Set[int] = set()
    ai_delay_ms = 900
    ai_next_action_ms = 0

    cut_message = ""
    cut_message_until = 0
    undo_flash_until = 0

    menu_x = W // 2 - MENU_BTN_W // 2
    players_btn = pygame.Rect(menu_x, 250, MENU_BTN_W, MENU_BTN_H)
    start_btn = pygame.Rect(menu_x, 320, MENU_BTN_W, MENU_BTN_H)
    rules_btn = pygame.Rect(menu_x, start_btn.bottom + 16, MENU_BTN_W, MENU_BTN_H)
    quit_btn = pygame.Rect(menu_x, rules_btn.bottom + 16, MENU_BTN_W, MENU_BTN_H)
    back_btn = pygame.Rect(60, H - 90, 160, 48)
    bgm_btn = pygame.Rect(W - 190, 24, 160, 36)
    sfx_btn = pygame.Rect(W - 190, bgm_btn.bottom + 8, 160, 36)

    def update_player_config() -> None:
        nonlocal players, ai_players
        players = make_players(human_players)
        ai_players = set(range(human_players, TOTAL_PLAYERS))

    def reset_game() -> None:
        nonlocal players, turn, moves_queue, current_move_i, phase
        nonlocal selected_coin_idx, last_roll_info, undo_stack
        nonlocal rolling, roll_end_ms, anim_a, anim_b
        nonlocal cut_message, cut_message_until
        update_player_config()
        turn = 0
        moves_queue = []
        current_move_i = 0
        phase = "WAIT_ROLL"
        selected_coin_idx = None
        last_roll_info = ""
        undo_stack = []
        rolling = False
        roll_end_ms = 0
        anim_a, anim_b = 0, 0
        cut_message = ""
        cut_message_until = 0

    update_player_config()

    def current_player_idx() -> int:
        return turn % len(players)

    def append_rolls_with_extras() -> None:
        nonlocal moves_queue, last_roll_info
        play_sound("roll")
        while True:
            a, b, mv, is_daayam = roll_dayakattai()
            moves_queue.append((mv, is_daayam, (a, b)))
            last_roll_info = f"Rolled {a},{b} => {mv}" + (" (DAAYAM)" if is_daayam else "")
            if mv in EXTRA_ROLL_ON:
                continue
            break

    def enqueue_rolls() -> None:
        nonlocal moves_queue, last_roll_info, rolling, roll_end_ms, anim_a, anim_b
        moves_queue = []
        append_rolls_with_extras()

        rolling = True
        roll_end_ms = pygame.time.get_ticks() + 450
        anim_a, anim_b = random.choice(FACES), random.choice(FACES)

    def coin_screen_pos(pi: int, ci: int) -> Tuple[int, int]:
        coin = players[pi].coins[ci]
        if coin.is_home:
            hx, hy = home_anchor(pi)
            row = ci // 3
            col = ci % 3
            return (hx + col * 34, hy + row * 34)
        if coin.is_finished:
            return cell_center(CENTER_CELL)
        assert coin.pos is not None
        return cell_center(PATH[coin.pos])

    def move_allowed(pi: int, ci: int, mv: int, _is_daayam: bool) -> bool:
        return is_move_legal(players, pi, ci, mv)

    def any_legal_move_for_coin(pi: int, ci: int) -> bool:
        if not moves_queue:
            return False
        for mv, is_daayam, _ in moves_queue:
            if move_allowed(pi, ci, mv, is_daayam):
                return True
        return False

    def any_legal_move(pi: int, mv: int, is_daayam: bool) -> bool:
        return any(move_allowed(pi, ci, mv, is_daayam) for ci in range(COINS_PER_PLAYER))

    def any_legal_move_for_any_roll(pi: int) -> bool:
        return any(any_legal_move(pi, mv, is_daayam) for (mv, is_daayam, _) in moves_queue)

    def best_roll_index_for_coin(pi: int, ci: int) -> Optional[int]:
        if not moves_queue:
            return None
        if 0 <= current_move_i < len(moves_queue):
            mv, is_daayam, _ = moves_queue[current_move_i]
            if move_allowed(pi, ci, mv, is_daayam):
                return current_move_i
        for idx, (mv, is_daayam, _) in enumerate(moves_queue):
            if move_allowed(pi, ci, mv, is_daayam):
                return idx
        return None

    def can_pick(pi: int, ci: int) -> bool:
        return any_legal_move_for_coin(pi, ci)

    def advance_if_no_moves() -> None:
        nonlocal current_move_i, moves_queue, phase, turn, selected_coin_idx
        if phase != "WAIT_PICK":
            return
        pi = current_player_idx()
        if any_legal_move_for_any_roll(pi):
            return
        turn += 1
        phase = "WAIT_ROLL"
        moves_queue = []
        current_move_i = 0
        selected_coin_idx = None

    def perform_move(pi: int, roll_idx: int, coin_idx: int) -> None:
        nonlocal phase, turn, moves_queue, current_move_i, selected_coin_idx, cut_message, cut_message_until
        mv, is_daayam, _ = moves_queue[roll_idx]
        ok, did_cut = apply_move(players, pi, coin_idx, mv)
        selected_coin_idx = coin_idx if ok else None
        if not ok:
            return

        play_sound("move")

        if did_cut:
            cut_message = "CUT!"
            cut_message_until = pygame.time.get_ticks() + 1200

        moves_queue.pop(roll_idx)

        if all_finished(players[pi]):
            phase = "GAME_OVER"
            moves_queue = []
            current_move_i = 0
            selected_coin_idx = None
            return

        if did_cut:
            append_rolls_with_extras()

        if not moves_queue:
            turn += 1
            phase = "WAIT_ROLL"
            current_move_i = 0
            selected_coin_idx = None
            return

        current_move_i = min(roll_idx, len(moves_queue) - 1)
        advance_if_no_moves()

    def choose_ai_move(pi: int) -> Optional[Tuple[int, int]]:
        for idx, (mv, is_daayam, _) in enumerate(moves_queue):
            for ci in range(COINS_PER_PLAYER):
                if move_allowed(pi, ci, mv, is_daayam) and would_cut(players, pi, ci, mv):
                    return idx, ci
        for idx, (mv, is_daayam, _) in enumerate(moves_queue):
            for ci in range(COINS_PER_PLAYER):
                if move_allowed(pi, ci, mv, is_daayam):
                    return idx, ci
        return None

    def schedule_ai() -> None:
        nonlocal ai_next_action_ms
        ai_next_action_ms = pygame.time.get_ticks() + ai_delay_ms

    running = True
    while running:
        clock.tick(60)

        now = pygame.time.get_ticks()
        if rolling:
            if now >= roll_end_ms:
                rolling = False
            else:
                if now % 60 < 10:
                    anim_a = random.choice(FACES)
                    anim_b = random.choice(FACES)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    toggle_sfx()
                elif event.key == pygame.K_b:
                    toggle_bgm()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if bgm_btn.collidepoint(mx, my):
                    toggle_bgm()
                    continue
                if sfx_btn.collidepoint(mx, my):
                    toggle_sfx()
                    continue
                if phase not in {"MENU", "RULES"} and current_player_idx() in ai_players:
                    continue

                if phase == "MENU":
                    if players_btn.collidepoint(mx, my):
                        human_players = (human_players % TOTAL_PLAYERS) + 1
                        update_player_config()
                    elif start_btn.collidepoint(mx, my):
                        reset_game()
                    elif rules_btn.collidepoint(mx, my):
                        phase = "RULES"
                    elif quit_btn.collidepoint(mx, my):
                        running = False

                elif phase == "RULES":
                    if back_btn.collidepoint(mx, my):
                        phase = "MENU"

                elif phase == "WAIT_ROLL":
                    if roll_btn.collidepoint(mx, my):
                        enqueue_rolls()
                        current_move_i = 0
                        phase = "WAIT_PICK"
                        selected_coin_idx = None
                        advance_if_no_moves()

                elif phase == "WAIT_PICK":
                    pi = current_player_idx()

                    if moves_queue and PANEL_X <= mx <= PANEL_X + MOVES_W:
                        first_y = MOVES_PANEL_Y + MOVES_HEADER_H
                        if first_y <= my < first_y + len(moves_queue) * MOVES_LINE_H:
                            idx = (my - first_y) // MOVES_LINE_H
                            if 0 <= idx < len(moves_queue):
                                current_move_i = int(idx)
                                advance_if_no_moves()
                                continue

                    clicked_coin = None
                    for ci in range(COINS_PER_PLAYER):
                        x, y = coin_screen_pos(pi, ci)
                        if (mx - x) ** 2 + (my - y) ** 2 <= (COIN_RADIUS + 6) ** 2:
                            clicked_coin = ci
                            break

                    if clicked_coin is not None:
                        roll_idx = best_roll_index_for_coin(pi, clicked_coin)
                        if roll_idx is None:
                            continue
                        current_move_i = roll_idx
                        perform_move(pi, roll_idx, clicked_coin)

        pi = current_player_idx()
        if phase not in {"MENU", "RULES", "GAME_OVER"} and pi in ai_players and pygame.time.get_ticks() >= ai_next_action_ms:
            if phase == "WAIT_ROLL":
                enqueue_rolls()
                current_move_i = 0
                phase = "WAIT_PICK"
                selected_coin_idx = None
                advance_if_no_moves()
                schedule_ai()
            elif phase == "WAIT_PICK":
                if not moves_queue:
                    advance_if_no_moves()
                else:
                    choice = choose_ai_move(pi)
                    if choice is None:
                        advance_if_no_moves()
                    else:
                        perform_move(pi, choice[0], choice[1])
                schedule_ai()

        sync_music(phase)

        screen.fill(bg)

        if phase == "MENU":
            title = font_title.render("DAYAKATT.AI", True, pygame.Color(40, 40, 40))
            screen.blit(title, (W // 2 - title.get_width() // 2, 210))
            draw_button(
                screen,
                players_btn,
                f"Humans: {human_players}  AI: {TOTAL_PLAYERS - human_players}",
                font_small,
                pygame.Color(235, 235, 235),
                pygame.Color(120, 120, 120),
                pygame.Color(20, 20, 20),
            )
            draw_button(screen, start_btn, "Start Game", font_big, pygame.Color(235, 235, 235), pygame.Color(120, 120, 120), pygame.Color(20, 20, 20))
            draw_button(screen, rules_btn, "Rules", font_big, pygame.Color(235, 235, 235), pygame.Color(120, 120, 120), pygame.Color(20, 20, 20))
            draw_button(screen, quit_btn, "Quit", font_big, pygame.Color(235, 235, 235), pygame.Color(120, 120, 120), pygame.Color(20, 20, 20))
            draw_button(
                screen,
                bgm_btn,
                f"BGM: {'On' if bgm_enabled else 'Off'}",
                font_small,
                pygame.Color(235, 235, 235),
                pygame.Color(120, 120, 120),
                pygame.Color(20, 20, 20),
            )
            draw_button(
                screen,
                sfx_btn,
                f"SFX: {'On' if sfx_enabled else 'Off'}",
                font_small,
                pygame.Color(235, 235, 235),
                pygame.Color(120, 120, 120),
                pygame.Color(20, 20, 20),
            )
            pygame.display.flip()
            await asyncio.sleep(0)
            continue

        if phase == "RULES":
            title = font_title.render("RULES", True, pygame.Color(40, 40, 40))
            screen.blit(title, (W // 2 - title.get_width() // 2, 70))
            text_x = 120
            text_y = 150
            max_width = W - 2 * text_x
            for i, rule in enumerate(rules_text, start=1):
                prefix = f"{i}) "
                wrapped = wrap_text(rule, font, max_width - font.size(prefix)[0])
                render_text(screen, prefix + wrapped[0], (text_x, text_y), font)
                text_y += 26
                for line in wrapped[1:]:
                    render_text(screen, " " * len(prefix) + line, (text_x, text_y), font)
                    text_y += 24
                text_y += 10
            draw_button(screen, back_btn, "Back", font_big, pygame.Color(235, 235, 235), pygame.Color(120, 120, 120), pygame.Color(20, 20, 20))
            draw_button(
                screen,
                bgm_btn,
                f"BGM: {'On' if bgm_enabled else 'Off'}",
                font_small,
                pygame.Color(235, 235, 235),
                pygame.Color(120, 120, 120),
                pygame.Color(20, 20, 20),
            )
            draw_button(
                screen,
                sfx_btn,
                f"SFX: {'On' if sfx_enabled else 'Off'}",
                font_small,
                pygame.Color(235, 235, 235),
                pygame.Color(120, 120, 120),
                pygame.Color(20, 20, 20),
            )
            pygame.display.flip()
            await asyncio.sleep(0)
            continue

        board_w = GRID_SIZE * CELL
        board_h = GRID_SIZE * CELL
        board_rect = pygame.Rect(BOARD_ORIGIN[0], BOARD_ORIGIN[1], board_w, board_h)
        pygame.draw.rect(screen, pygame.Color(255, 255, 255), board_rect, border_radius=8)
        pygame.draw.rect(screen, pygame.Color(160, 160, 160), board_rect, 2, border_radius=8)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = pygame.Rect(BOARD_ORIGIN[0] + c * CELL, BOARD_ORIGIN[1] + r * CELL, CELL, CELL)
                pygame.draw.rect(screen, grid_line, rect, 1)
                if (r, c) in SAFE_CELLS:
                    pygame.draw.rect(screen, safe_fill, rect)
                    draw_x(screen, rect.inflate(-16, -16), pygame.Color(140, 140, 140), width=3)

        # PATH indices
        for i, rc in enumerate(PATH):
            cx, cy = cell_center(rc)
            img = font.render(str(i), True, pygame.Color(120, 120, 120))
            screen.blit(img, (cx - 10, cy - 10))

        if cut_message and now < cut_message_until:
            msg = font_big.render(cut_message, True, pygame.Color(200, 50, 50))
            msg_x = BOARD_ORIGIN[0] + (board_w // 2) - (msg.get_width() // 2)
            msg_y = BOARD_ORIGIN[1] - 34
            screen.blit(msg, (msg_x, msg_y))

        panel_rect = pygame.Rect(PANEL_X - 20, 50, PANEL_W, 640)
        pygame.draw.rect(screen, pygame.Color(255, 255, 255), panel_rect, border_radius=12)
        pygame.draw.rect(screen, pygame.Color(210, 210, 210), panel_rect, 2, border_radius=12)

        pi = current_player_idx()
        render_text(screen, "DAYAKATT.AI", (PANEL_X, PANEL_Y - 20), font_big)
        render_text(screen, f"Turn: {players[pi].name}", (PANEL_X, PANEL_Y + 10), font_big, color=player_colors[pi])
        draw_button(
            screen,
            bgm_btn,
            f"BGM: {'On' if bgm_enabled else 'Off'}",
            font_small,
            pygame.Color(235, 235, 235),
            pygame.Color(120, 120, 120),
            pygame.Color(20, 20, 20),
        )
        draw_button(
            screen,
            sfx_btn,
            f"SFX: {'On' if sfx_enabled else 'Off'}",
            font_small,
            pygame.Color(235, 235, 235),
            pygame.Color(120, 120, 120),
            pygame.Color(20, 20, 20),
        )

        pygame.draw.rect(screen, pygame.Color(235, 235, 235), roll_btn, border_radius=10)
        pygame.draw.rect(screen, pygame.Color(120, 120, 120), roll_btn, 2, border_radius=10)
        render_text(screen, "ROLL", (roll_btn.x + 66, roll_btn.y + 12), font_big)

        dice_label_y = roll_btn.bottom + 16
        dice_y = dice_label_y + 22
        die_w, die_h = 90, 44
        dice_gap = 12

        if rolling:
            show_a, show_b = anim_a, anim_b
        elif moves_queue and phase == "WAIT_PICK" and current_move_i < len(moves_queue):
            show_a, show_b = moves_queue[current_move_i][2]
        else:
            show_a, show_b = 0, 0

        draw_dayakattai_die(screen, pygame.Rect(PANEL_X, dice_y, die_w, die_h), show_a)
        draw_dayakattai_die(screen, pygame.Rect(PANEL_X + die_w + dice_gap, dice_y, die_w, die_h), show_b)

        rolls_left = len(moves_queue) if phase == "WAIT_PICK" else 0
        rolls_y = dice_y + die_h + 8
        render_text(screen, f"Rolls left: {rolls_left}", (PANEL_X, rolls_y), font)

        finished_counts = [sum(c.is_finished for c in pl.coins) for pl in players]
        lead = max(finished_counts)
        leaders = [i for i, cnt in enumerate(finished_counts) if cnt == lead]
        if len(leaders) > 1:
            leader_text = f"Leader: Tie ({lead}/{COINS_PER_PLAYER})"
        else:
            leader_text = f"Leader: {players[leaders[0]].name} ({lead}/{COINS_PER_PLAYER})"
        render_text(screen, leader_text, (PANEL_X, MOVES_PANEL_Y - 24), font)

        y = MOVES_PANEL_Y
        if moves_queue:
            render_text(screen, "Moves:", (PANEL_X, y), font_big)
            y += MOVES_HEADER_H
            for idx, (mv, is_daayam, (a, b)) in enumerate(moves_queue):
                marker = "▶" if idx == current_move_i and phase == "WAIT_PICK" else " "
                render_text(screen, f"{marker} {a},{b} → {mv}" + (" D" if is_daayam else ""), (PANEL_X, y), font)
                y += MOVES_LINE_H
        else:
            render_text(screen, "Moves: (roll to start)", (PANEL_X, y), font)

        help_y = min(y + 16, PANEL_Y + 340)
        # render_text(screen, "Click a move, then a coin.", (PANEL_X, help_y), font_small)
        # render_text(screen, "Extra rolls add to queue; use all to end turn.", (PANEL_X, help_y + 20), font_small)
        # render_text(screen, "See Rules in the main menu.", (PANEL_X, help_y + 40), font_small)

        for pi in range(len(players)):
            hx, hy = home_anchor(pi)
            label = ""
            render_text(screen, label, (hx, hy - 20), font, color=player_colors[pi])

        # stack coins on same PATH cell so you can see them
        stacks: Dict[int, List[Tuple[int, int]]] = {}
        for ppi, pl in enumerate(players):
            for ci, coin in enumerate(pl.coins):
                if coin.pos is not None and 0 <= coin.pos < TRACK_LEN:
                    stacks.setdefault(coin.pos, []).append((ppi, ci))

        drawn_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for pos, plist in stacks.items():
            pts = coin_draw_positions_at_cell(PATH[pos], plist)
            for (pii, cii), (x, y2) in zip(plist, pts):
                drawn_positions[(pii, cii)] = (x, y2)

        for ppi, pl in enumerate(players):
            for ci, coin in enumerate(pl.coins):
                x, y2 = drawn_positions.get((ppi, ci), coin_screen_pos(ppi, ci))
                if phase == "WAIT_PICK" and ppi == current_player_idx() and can_pick(ppi, ci):
                    pygame.draw.circle(screen, pygame.Color(40, 140, 60), (x, y2), COIN_RADIUS + 6, 3)
                pygame.draw.circle(screen, player_colors[ppi], (x, y2), COIN_RADIUS)
                pygame.draw.circle(screen, black, (x, y2), COIN_RADIUS, 2)
                lbl = font.render(str(ci), True, pygame.Color(255, 255, 255))
                screen.blit(lbl, (x - 6, y2 - 10))

        if phase == "GAME_OVER":
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            win_text = f"🏆 {players[current_player_idx()].name} wins!"
            img = font_big.render(win_text, True, pygame.Color(255, 255, 255))
            screen.blit(img, (W // 2 - img.get_width() // 2, H // 2 - 20))
            img2 = font.render("Close the window to exit.", True, pygame.Color(255, 255, 255))
            screen.blit(img2, (W // 2 - img2.get_width() // 2, H // 2 + 18))

        pygame.display.flip()
        await asyncio.sleep(0)

    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())
