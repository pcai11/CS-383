BLANK_CHAR = '0'


class EightPuzzleBoard:
    """Class representing a single state of an 8-puzzle board.

    In general, the board positions are set when an object is created and should not be
    manipulated.  The successor functions generate reachable states from the current board.

    The tiles themselves are represented by a list of digits internally, and manipulated
    using (x, y) coordinates.
    """

    def __init__(self, board_string, mods=None):
        """Constructor for 8-puzzle board.

        Args:
            board_string: nine-digit string describing the board, with '0' representing the blank
            mods: optional list of (x, y, value) tuples that are applied to the board_string
                immediately after creation
        """
        self._board = list(board_string)
        if mods:
            for x, y, val in mods:
                self._set_tile(x, y, val)

    def _get_tile(self, x, y):  # return an individual tile value
        return self._board[6 - y * 3 + x]

    def _set_tile(self, x, y, val):  # set an individual tile value
        self._board[6 - y * 3 + x] = val

    def _create_successor(self, delta_x, delta_y):  # create a successor object (or None if invalid)
        pos = self._board.index(BLANK_CHAR)
        blank_x = pos % 3
        blank_y = 2 - int(pos / 3)
        move_x = blank_x + delta_x
        move_y = blank_y + delta_y
        if (move_x < 0) or (move_x > 2) or (move_y < 0) or (move_y > 2):
            return None
        else:
            mods = [(blank_x, blank_y, self._get_tile(move_x, move_y)),
                    (move_x, move_y, self._get_tile(blank_x, blank_y))]
            succ = EightPuzzleBoard("".join(self._board), mods)
            return succ

    def success_up(self):
        """Generate the board resulting from moving a tile up into the blank space.

        Returns: an EightPuzzleBoard object representing the successor state of this one, or None
            if up is not a valid move for this board
        """
        return self._create_successor(0, -1)

    def success_down(self):
        """Generate the board resulting from moving a tile down into the blank space.

        Returns: an EightPuzzleBoard object representing the successor state of this one, or None
            if down is not a valid move for this board
        """
        return self._create_successor(0, 1)

    def success_right(self):
        """Generate the board resulting from moving a tile right into the blank space.

        Returns: an EightPuzzleBoard object representing the successor state of this one, or None
            if right is not a valid move for this board
        """
        return self._create_successor(-1, 0)

    def success_left(self):
        """Generate the board resulting from moving a tile left into the blank space.

        Returns: an EightPuzzleBoard object representing the successor state of this one, or None
            if left is not a valid move for this board
        """
        return self._create_successor(1, 0)

    def successors(self):
        """Generates all successors of this board.

        Returns: a dictionary mapping moves to EightPuzzleBoard objects representing the results of
            each valid move move for this board
        """
        u = self.success_up()
        d = self.success_down()
        l = self.success_left()
        r = self.success_right()
        succs = {}
        if u:
            succs["up"] = u
        if d: 
            succs["down"] = d
        if l: 
            succs["left"] = l
        if r:
            succs["right"] = r
        return succs

    def find(self, c):
        """Return the coordinates of a given tile.
        
        Returns: a tuple containing x, y coordinates of c
        """
        pos = self._board.index(c)
        x = pos % 3
        y = 2 - int(pos/3)
        return x, y

    def __str__(self):
        return "".join(self._board)

    def __repr__(self):
        return "".join(self._board)

    def __hash__(self):
        return hash("".join(self._board))

    def __eq__(self, other):
        return "".join(self._board) == "".join(other._board)
        
    def pretty(self):
        """Pretty-print the board.

        Returns: a readable three-line representation of the board
        """
        brd_str = " ".join(self._board).replace(BLANK_CHAR, ".", 1)
        return "{}\n{}\n{}".format(brd_str[:6], brd_str[6:12], brd_str[12:])


    
