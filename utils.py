import pygame

def load_images(square_size):
    pieces = ['B', 'K', 'N', 'P', 'Q', 'R']  # Bishop, King, Knight, Pawn, Queen, Rook
    colors = ['b', 'w']  # Black, White
    images = {}
    for color in colors:
        for piece in pieces:
            images[f"{color}{piece}"] = pygame.transform.scale(
                pygame.image.load(f"images/{color}{piece}.png"), 
                (square_size, square_size)
            )
    return images 