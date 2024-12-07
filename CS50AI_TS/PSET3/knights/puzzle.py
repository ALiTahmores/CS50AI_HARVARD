from logic import *

# Define symbols for each character being a knight or a knave
AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")
BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")
CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# General knowledge: A person is either a knight or a knave, but not both
general_knowledge = And(
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),
    Or(CKnight, CKnave),
    Not(And(CKnight, CKnave))
)

# Puzzle 0: A says "I am both a knight and a knave."
knowledge0 = And(
    general_knowledge,
    # A is truthful if a knight
    Implication(AKnight, And(AKnight, AKnave)),
    # A is lying if a knave
    Implication(AKnave, Not(And(AKnight, AKnave)))
)

# Puzzle 1:
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    general_knowledge,
    # If A is a knight, they are truthful
    Implication(AKnight, And(AKnave, BKnave)),
    # If A is a knave, the statement is false
    Implication(AKnave, Not(And(AKnave, BKnave)))
)

# Puzzle 2:
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    general_knowledge,
    # A's statement logic
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    Implication(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))),
    # B's statement logic
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    Implication(BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight))))
)

# Puzzle 3:
# A says either "I am a knight." or "I am a knave."
# B says "A said 'I am a knave'." and "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    general_knowledge,
    # A says "I am a knight." or "I am a knave."
    Or(
        And(AKnight, Or(AKnight, AKnave)),
        And(AKnave, Not(Or(AKnight, AKnave)))
    ),
    # B's statements
    Implication(BKnight, And(CKnave, AKnave)),
    Implication(BKnave, Not(And(CKnave, AKnave))),
    # C's statement
    Implication(CKnight, AKnight),
    Implication(CKnave, Not(AKnight))
)

# Main function to evaluate each puzzle


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
