import nltk
import sys
import re

# TERMINALS: Expanding vocabulary for various parts of speech
TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red" | "big" | "small" | "beautiful" | "quick"
Adv -> "down" | "here" | "never" | "quickly" | "carefully" | "always"
Conj -> "and" | "but" | "or" | "until"
Det -> "a" | "an" | "his" | "my" | "the" | "some" | "many" | "few"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself" | "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "smile" | "thursday" | "walk" | "we" | "word" | "cat" | "dog" | "house"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "from"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat" | "smiled" | "tell" | "were" | "is" | "are"
"""

# NONTERMINALS: Expanding grammar rules to cover more sentence structures
NONTERMINALS = """
S -> NP VP | NP VP Conj NP VP | NP VP Conj VP | S Conj S | VP Conj VP
NP -> N | Det N | Det AP N | P NP | NP P NP | Det N P NP
VP -> V | Adv VP | V Adv | VP NP | V NP Adv | V Adv NP | VP PP
AP -> Adj | AP Adj
PP -> P NP
"""

# Create the grammar with expanded vocabulary and rules
grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    # Handle file input or user input
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    else:
        s = input("Sentence: ")

    # Preprocess sentence to tokenize and filter
    s = preprocess(s)

    # Try to parse the sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print parsed trees with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks:")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    test = re.compile('[a-zA-Z]')
    tokens = nltk.word_tokenize(sentence)
    return [entry.lower() for entry in tokens if test.match(entry)]


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []
    ptree = nltk.tree.ParentedTree.convert(tree)

    # Iterate over all subtrees
    for subtree in ptree.subtrees():
        # Check if the current subtree is a noun phrase (NP)
        if subtree.label() == "NP":
            # Ensure the NP does not contain another NP as a subtree
            if not any(ancestor.label() == "NP" for ancestor in subtree.subtrees(lambda t: t != subtree)):
                chunks.append(subtree)

    # Return noun phrase chunks as a list of their flattened forms
    return [chunk.flatten() for chunk in chunks]


if __name__ == "__main__":
    main()
