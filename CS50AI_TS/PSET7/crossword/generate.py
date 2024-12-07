import sys
from crossword import *


class CrosswordCreator:
    def __init__(self, crossword):
        """
        Create new CSP crossword generator using a different approach.
        """
        self.crossword = crossword
        self.domains = {var: self.crossword.words.copy() for var in self.crossword.variables}

    def letter_grid(self, assignment):
        """
        Returns a 2D array representing the crossword grid based on the current assignment.
        """
        grid = [[None for _ in range(self.crossword.width)] for _ in range(self.crossword.height)]
        for var, word in assignment.items():
            direction = var.direction
            for k in range(len(word)):
                i = var.i + (k if direction == Variable.DOWN else 0)
                j = var.j + (k if direction == Variable.ACROSS else 0)
                grid[i][j] = word[k]
        return grid

    def print(self, assignment):
        """
        Prints the crossword grid to the terminal.
        """
        grid = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(grid[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file using a more efficient method.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 80
        cell_border = 3
        interior_size = cell_size - 2 * cell_border
        grid = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new("RGBA", (self.crossword.width * cell_size,
                        self.crossword.height * cell_size), "black")
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 60)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                rect = [
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if grid[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), grid[i][j], font=font)
                        draw.text(
                            (rect[0][0] + (interior_size - w) / 2,
                             rect[0][1] + (interior_size - h) / 2 - 10),
                            grid[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Solve the crossword puzzle using a more efficient backtracking approach.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack({})

    def enforce_node_consistency(self):
        """
        Update `self.domains` so that each variable is node-consistent.
        This is done by removing any words that do not match the length of the variable.
        """
        for var in self.crossword.variables:
            self.domains[var] = {word for word in self.domains[var] if len(word) == var.length}

    def revise(self, x, y):
        """
        Makes variable `x` arc-consistent with variable `y` using an optimized method.
        """
        overlap = self.crossword.overlaps.get((x, y))
        if overlap is None:
            return False

        revised = False
        i, j = overlap
        for word_x in list(self.domains[x]):
            if not any(word_x[i] == word_y[j] for word_y in self.domains[y]):
                self.domains[x].remove(word_x)
                revised = True
        return revised

    def ac3(self, arcs=None):
        """
        Enforces arc consistency using an optimized method.
        """
        if arcs is None:
            arcs = [(x, y) for x in self.crossword.variables for y in self.crossword.neighbors(x)]

        queue = arcs
        while queue:
            x, y = queue.pop(0)  # Use pop(0) to implement a queue (FIFO)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False  # If any domain becomes empty, no solution is possible
                for neighbor in self.crossword.neighbors(x) - {y}:  # Exclude y from its neighbors
                    queue.append((neighbor, x))  # Add arc (neighbor, x) to queue
        return True

    def assignment_complete(self, assignment):
        """
        Check if the assignment is complete.
        """
        return all(var in assignment for var in self.crossword.variables)

    def consistent(self, assignment):
        """
        Check if the assignment is consistent with no conflicts.
        """
        assigned_words = set()
        for var in self.crossword.variables:
            if var not in assignment:
                continue
            word = assignment[var]
            if len(word) != var.length:
                return False
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[(var, neighbor)]
                    if assignment[var][i] != assignment[neighbor][j]:
                        return False
            if word in assigned_words:
                return False
            assigned_words.add(word)
        return True

    def order_domain_values(self, var, assignment):
        """
        Order the domain values by the least constraining value heuristic.
        """
        # Return words ordered by how many words they eliminate from the domain of neighbors
        return sorted(self.domains[var], key=lambda word: sum(1 for neighbor in self.crossword.neighbors(var) if word in self.domains[neighbor]))

    def select_unassigned_variable(self, assignment):
        """
        Select the unassigned variable with the fewest remaining values in its domain.
        """
        unassigned_vars = [(var, len(self.domains[var]))
                           for var in self.crossword.variables if var not in assignment]
        return min(unassigned_vars, key=lambda x: x[1])[0]

    def backtrack(self, assignment):
        """
        Backtracking algorithm to solve the crossword puzzle.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result:
                    return result
            assignment.pop(var)
        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
