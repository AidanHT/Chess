# main.py

import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

# Load images
def load_images():
    pieces = ['B', 'K', 'N', 'P', 'Q', 'R']  # Bishop, King, Knight, Pawn, Queen, Rook
    colors = ['b', 'w']  # Black, White
    images = {}
    for color in colors:
        for piece in pieces:
            images[f"{color}{piece}"] = pygame.transform.scale(
                pygame.image.load(f"images/{color}{piece}.png"), 
                (SQUARE_SIZE, SQUARE_SIZE)
            )
    return images

# Draw the board
def draw_board(win):
    colors = [WHITE, GRAY]
    for row in range(ROWS):
        for col in range(COLS):
            color = colors[(row + col) % 2]
            pygame.draw.rect(win, color, (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Initialize board with pieces
def initialize_board():
    board = [[None for _ in range(COLS)] for _ in range(ROWS)]
    # Place pawns
    for col in range(COLS):
        board[1][col] = 'bP'
        board[6][col] = 'wP'
    # Place other pieces
    order = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for i, piece in enumerate(order):
        board[0][i] = f'b{piece}'
        board[7][i] = f'w{piece}'
    return board

# Draw pieces on the board
def draw_pieces(win, board, images):
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece:
                win.blit(images[piece], (col*SQUARE_SIZE, row*SQUARE_SIZE))

# Check if a move is valid
def is_valid_move(board, piece, start_pos, end_pos, current_turn):
    start_row, start_col = start_pos
    end_row, end_col = end_pos
    if piece is None or piece[0] != current_turn:
        return False

    # Check if the destination square contains a piece of the same color
    destination_piece = board[end_row][end_col]
    if destination_piece and destination_piece[0] == current_turn:
        return False

    piece_type = piece[1]
    color = piece[0]

    # Pawn movement
    if piece_type == 'P':
        direction = -1 if color == 'w' else 1
        start_row_pawn = 6 if color == 'w' else 1
        if start_col == end_col:  # Move forward
            if (end_row == start_row + direction and board[end_row][end_col] is None) or \
               (end_row == start_row + 2 * direction and start_row == start_row_pawn and board[end_row][end_col] is None and board[start_row + direction][end_col] is None):
                return True
        elif abs(start_col - end_col) == 1 and end_row == start_row + direction:  # Capture
            if board[end_row][end_col] is not None and board[end_row][end_col][0] != color:
                return True

    # Rook movement
    elif piece_type == 'R':
        if start_row == end_row or start_col == end_col:
            if not any(board[r][c] for r, c in get_path(start_pos, end_pos)):
                return True

    # Knight movement
    elif piece_type == 'N':
        if (abs(start_row - end_row), abs(start_col - end_col)) in [(2, 1), (1, 2)]:
            return True

    # Bishop movement
    elif piece_type == 'B':
        if abs(start_row - end_row) == abs(start_col - end_col):
            if not any(board[r][c] for r, c in get_path(start_pos, end_pos)):
                return True

    # Queen movement
    elif piece_type == 'Q':
        if start_row == end_row or start_col == end_col or abs(start_row - end_row) == abs(start_col - end_col):
            if not any(board[r][c] for r, c in get_path(start_pos, end_pos)):
                return True

    # King movement
    elif piece_type == 'K':
        if max(abs(start_row - end_row), abs(start_col - end_col)) == 1:
            return True

    return False

# Get path between two positions (excluding start and end)
def get_path(start_pos, end_pos):
    path = []
    start_row, start_col = start_pos
    end_row, end_col = end_pos
    row_step = (end_row - start_row) // max(1, abs(end_row - start_row))
    col_step = (end_col - start_col) // max(1, abs(end_col - start_col))
    current_row, current_col = start_row + row_step, start_col + col_step
    while (current_row, current_col) != (end_row, end_col):
        path.append((current_row, current_col))
        current_row += row_step
        current_col += col_step
    return path

# Main function
def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess')
    images = load_images()
    font = pygame.font.SysFont(None, 55)

    def reset_game():
        return initialize_board(), 'w', None, None

    board, current_turn, selected_piece, selected_pos = reset_game()

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                row, col = y // SQUARE_SIZE, x // SQUARE_SIZE
                if selected_piece:
                    # Attempt to move piece
                    if is_valid_move(board, selected_piece, selected_pos, (row, col), current_turn):
                        # Check if the move captures a king
                        if board[row][col] and board[row][col][1] == 'K':
                            winner = 'White' if current_turn == 'w' else 'Black'
                            text = font.render(f'{winner} wins!', True, (0, 0, 0))
                            win.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
                            pygame.display.flip()
                            pygame.time.wait(3000)
                            board, current_turn, selected_piece, selected_pos = reset_game()
                        else:
                            board[row][col] = selected_piece
                            board[selected_pos[0]][selected_pos[1]] = None
                            # Switch turns
                            current_turn = 'b' if current_turn == 'w' else 'w'
                    selected_piece = None
                else:
                    # Select piece
                    if board[row][col] and board[row][col][0] == current_turn:
                        selected_piece = board[row][col]
                        selected_pos = (row, col)

        draw_board(win)
        draw_pieces(win, board, images)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()