import pygame

# Colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

def draw_board(win, square_size):
    colors = [WHITE, GRAY]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(win, color, (col * square_size, row * square_size, square_size, square_size))

def initialize_board():
    board = [[None for _ in range(8)] for _ in range(8)]
    # Place pawns
    for col in range(8):
        board[1][col] = 'bP'
        board[6][col] = 'wP'
    # Place other pieces
    order = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for i, piece in enumerate(order):
        board[0][i] = f'b{piece}'
        board[7][i] = f'w{piece}'
    return board

def draw_pieces(win, board, images, square_size):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece:
                win.blit(images[piece], (col * square_size, row * square_size)) 