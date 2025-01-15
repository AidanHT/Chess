# main.py

import pygame
import sys
from board import draw_board, initialize_board, draw_pieces
from pieces import is_valid_move
from utils import load_images

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
SQUARE_SIZE = WIDTH // 8

def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess')
    images = load_images(SQUARE_SIZE)
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

        draw_board(win, SQUARE_SIZE)
        draw_pieces(win, board, images, SQUARE_SIZE)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()