import csv
import itertools

# Constants for gene and trait probabilities
PROBS = {
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },
    "trait": {
        2: {True: 0.65, False: 0.35},
        1: {True: 0.56, False: 0.44},
        0: {True: 0.01, False: 0.99}
    },
    "mutation": 0.01
}


def load_data(filename):
    """
    Load data from a CSV file and return it as a dictionary.
    """
    data = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["name"]] = {
                "name": row["name"],
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all subsets of set s.
    """
    return [
        set(subset) for subset in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def gene_inheritance(parent, parent_genes):
    """
    Calculate the probability of a child inheriting a gene from a parent.
    """
    if parent_genes == 2:
        return 1 - PROBS["mutation"]
    elif parent_genes == 1:
        return 0.5
    else:
        return PROBS["mutation"]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute the joint probability of the given set of genes and traits.
    """
    probability = 1

    for person in people:
        genes = (2 if person in two_genes else
                 1 if person in one_gene else 0)
        has_trait = person in have_trait

        mother = people[person]["mother"]
        father = people[person]["father"]

        # Independent gene probability
        if mother is None and father is None:
            gene_prob = PROBS["gene"][genes]
        else:
            mom_genes = (2 if mother in two_genes else
                         1 if mother in one_gene else 0)
            dad_genes = (2 if father in two_genes else
                         1 if father in one_gene else 0)

            if genes == 2:
                gene_prob = gene_inheritance(mother, mom_genes) * \
                    gene_inheritance(father, dad_genes)
            elif genes == 1:
                gene_prob = (
                    gene_inheritance(mother, mom_genes) * (1 - gene_inheritance(father, dad_genes)) +
                    (1 - gene_inheritance(mother, mom_genes)) * gene_inheritance(father, dad_genes)
                )
            else:
                gene_prob = (1 - gene_inheritance(mother, mom_genes)) * \
                    (1 - gene_inheritance(father, dad_genes))

        # Trait probability
        trait_prob = PROBS["trait"][genes][has_trait]

        # Multiply into total probability
        probability *= gene_prob * trait_prob

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Update the probabilities dictionary with new joint probability `p`.
    """
    for person in probabilities:
        genes = (2 if person in two_genes else
                 1 if person in one_gene else 0)
        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Normalize probabilities so that they sum to 1.
    """
    for person in probabilities:
        for field in probabilities[person]:
            total = sum(probabilities[person][field].values())
            for key in probabilities[person][field]:
                probabilities[person][field][key] /= total


def heredity(filename):
    """
    Perform full heredity probability calculation.
    """
    people = load_data(filename)

    probabilities = {
        person: {
            "gene": {0: 0, 1: 0, 2: 0},
            "trait": {True: 0, False: 0}
        }
        for person in people
    }

    names = set(people)
    for have_trait in powerset(names):
        for one_gene in powerset(names - have_trait):
            for two_genes in powerset(names - one_gene - have_trait):
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    normalize(probabilities)

    return probabilities
