import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = [[False for _ in range(width)] for _ in range(height)]

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randint(0, height - 1)
            j = random.randint(0, width - 1)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation of the board.
        """
        for i in range(self.height):
            print(" ".join(["X" if self.board[i][j] else "." for j in range(self.width)]))
        print()

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are within one row and column of a given cell.
        """
        count = 0
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if (i, j) == cell:
                    continue
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1
        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return set(self.cells)
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return set(self.cells)
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width

        # Keep track of cells that have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the AI is told that a given safe cell has a specific
        number of neighboring mines.
        """
        self.moves_made.add(cell)
        self.mark_safe(cell)

        # Add a new sentence based on the cell's neighbors
        new_sentence_cells = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if (i, j) == cell:
                    continue
                if 0 <= i < self.height and 0 <= j < self.width:
                    if (i, j) not in self.safes and (i, j) not in self.mines:
                        new_sentence_cells.add((i, j))
                    elif (i, j) in self.mines:
                        count -= 1

        if new_sentence_cells:
            self.knowledge.append(Sentence(new_sentence_cells, count))

        # Update knowledge
        self.update_knowledge()

    def update_knowledge(self):
        """
        Updates the knowledge base, marking cells as safe or mines
        and adding any new sentences inferred from existing ones.
        """
        knowledge_changed = True
        while knowledge_changed:
            knowledge_changed = False

            safes = set()
            mines = set()

            # Identify all safes and mines
            for sentence in self.knowledge:
                safes.update(sentence.known_safes())
                mines.update(sentence.known_mines())

            # Mark all identified safes and mines
            for safe in safes:
                if safe not in self.safes:
                    self.mark_safe(safe)
                    knowledge_changed = True

            for mine in mines:
                if mine not in self.mines:
                    self.mark_mine(mine)
                    knowledge_changed = True

            # Remove empty sentences
            self.knowledge = [s for s in self.knowledge if s.cells]

            # Infer new sentences
            new_sentences = []
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1 != s2 and s1.cells.issubset(s2.cells):
                        inferred = Sentence(s2.cells - s1.cells, s2.count - s1.count)
                        if inferred not in self.knowledge and inferred not in new_sentences:
                            new_sentences.append(inferred)

            if new_sentences:
                self.knowledge.extend(new_sentences)
                knowledge_changed = True

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        """
        for cell in self.safes:
            if cell not in self.moves_made:
                return cell
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        """
        choices = [
            (i, j)
            for i in range(self.height)
            for j in range(self.width)
            if (i, j) not in self.moves_made and (i, j) not in self.mines
        ]
        if choices:
            return random.choice(choices)
        return None
