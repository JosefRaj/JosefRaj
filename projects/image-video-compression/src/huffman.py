import numpy as np
from .helpers import show_huffman_table, show_huffman_tree


# Session 2: Huffman coding
def create_huffman_table_from_signal(input_signal: np.ndarray) -> dict:
    """
    Create a Huffman table from the given input signal.
    
    :param input_signal: The input signal to create the Huffman table from.
    :type input_signal: np.ndarray
    :return: A dictionary mapping each unique value in the input signal to its corresponding Huffman code.
    :rtype: dict
    """
    symbols, counts = np.unique(
        input_signal,
        axis=0,
        return_counts=True
    )

    probabilities = counts / np.sum(counts)

    frequencies = []

    for symbol, probability in zip(symbols, probabilities):
        frequencies.append((tuple(symbol), float(probability)))

    return create_huffman_table_from_probability(frequencies)    
    

def create_huffman_table_from_probability(frequencies: list) -> dict:
    """"
    Create a Huffman table from the given frequencies.
    
    :param frequencies: A list of tuples (symbol, frequency).
    :type frequencies: list
    :return: A dictionary mapping each unique value to its corresponding Huffman code.
    :rtype: dict
    """   
    sorted_frequencies = sorted(
        frequencies,
        key=lambda entry: entry[1]
    )

    huffman_table = {}

    for symbol, probability in sorted_frequencies:
        huffman_table[symbol] = ""

    huffman_tree = []

    for symbol, probability in sorted_frequencies:
        huffman_tree.append((probability, [symbol]))
        
    while len(huffman_tree) > 1:
        least_prob = huffman_tree.pop(0)
        second_least_prob = huffman_tree.pop(0)

        for symbol in least_prob[1]:
            huffman_table[symbol] = "1" + huffman_table[symbol]

        for symbol in second_least_prob[1]:
            huffman_table[symbol] = "0" + huffman_table[symbol]

        new_probability = least_prob[0] + second_least_prob[0]
        new_symbols = least_prob[1] + second_least_prob[1]

        huffman_tree.insert(0, (new_probability, new_symbols))
        huffman_tree.sort(key=lambda entry: entry[0])

    
    return huffman_table