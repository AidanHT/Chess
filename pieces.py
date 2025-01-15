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