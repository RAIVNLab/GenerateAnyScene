import json
import nltk
from nltk.corpus import wordnet as wn

def construct_flatten_wordnet_noun():
    """Constructs a dictionary of all noun synsets in WordNet.

    Returns:
        dictionary: A dictionary containing the noun synsets in WordNet.
    """
    noun_dict = {}
    for synset in wn.all_synsets(pos=wn.NOUN):
        details = {
            "definition": synset.definition(),
            "hypernyms": extract_hypernyms(synset),
            "level": synset.min_depth()
        }
        noun_dict[synset.name()] = details
    return noun_dict

# nltk.download('wordnet')
def extract_hypernyms(synset):
    """Extracts the hypernyms of a synset.

    Args:
        synset: A WordNet synset.

    Returns:
        A list containing the first hypernyms of the synset.
    """
    hypernyms = synset.hypernyms()
    if not hypernyms:
        return []
    else:
        # get the first hypernym
        return [hypernyms[0].name()]
    


def extract_hyponyms(synset, level):
    """
    Extracts the hyponyms of a synset and their hyponyms recursively.
    Args:
        synset: A WordNet synset.
        level: The level of the current synset in the WordNet tree.
    Returns:
        A dictionary containing the hyponyms of the synset.
    """
    hyponyms = {}
    for hyponym in synset.hyponyms():
        # get the definition and hyponyms of the hyponym
        hyponyms[hyponym.name()] = {
            "definition": hyponym.definition(),
            "level": level + 1,
            "hyponyms": extract_hyponyms(hyponym, level + 1)
        }
    return hyponyms

def construct_word_tree(root="object.n.01", root_level=2):
    """
    Constructs a tree of hyponyms for a given root synset.
    Args:
        root: The name of the root synset.
        root_level: The level of the root synset in the WordNet tree.
    Returns:
        data: A dictionary containing the tree of hyponyms.
        count: The number of objects in the tree.
    """
    word_synset = wn.synset(root)
    # living_thing = wn.all_synsets(pos=wn.ADJ)
    data = {
        "definition": word_synset.definition(),
        "level": root_level, 
        "hyponyms": extract_hyponyms(word_synset, root_level+1)
    }
    
    # walk through all the tree and count how many object are there in the tree
    count = 0
    def count_hyponyms(hyponyms):
        nonlocal count
        for hyponym in hyponyms:
            count += 1
            count_hyponyms(hyponyms[hyponym]['hyponyms'])
    count_hyponyms(data['hyponyms'])

    return data, count


def extract_polysemous_words(word_tree):
    """ Extracts the polysemous words from the WordNet tree.
    
    Args:
        word_tree: A dictionary containing the tree of hyponyms.
    
    Returns:
        polymeous_word: A dictionary containing the polysemous words and their definitions.
        {
            word_name: {
                word_name.x.xx: definition,
                word_name.x.xx: definition, 
            },
            ...
        }
    """
     
    noun_data = construct_flatten_wordnet_noun()
    word_polysemy_pairs = {} # get all the polysemy of each word
    def process_hyponyms(hyponyms):
        for hyponym in hyponyms:
            # split the word name and pos, num
            if len(hyponym.split('.')) != 3:
                continue
            word, pos, num = hyponym.split('.')
            if word not in word_polysemy_pairs:
                word_polysemy_pairs[word] = [hyponym]
            else:
                word_polysemy_pairs[word].append(hyponym)
            process_hyponyms(hyponyms[hyponym]['hyponyms'])
    process_hyponyms(word_tree['hyponyms'])

    ## 
    # clean the poly_words, no duplicate polysemous words for each original word
    for word_name in word_polysemy_pairs: 
        word_polysemy_pairs[word_name] = list(set(word_polysemy_pairs[word_name]))
    polysemous_word = {}
    for word_name in word_polysemy_pairs:
        if len(word_polysemy_pairs[word_name]) > 1: # is a polysemous word
            polysemous_word[word_name] = {each_polysemy: noun_data[each_polysemy]["definition"] for each_polysemy in word_polysemy_pairs[word_name]}
    print("total polymeous_word_name is:", len(word_polysemy_pairs.keys()))
    # uotput the polysemous words and their fathers
    return polysemous_word

def extract_cofather_polysemous_words(word_tree):
    """ Extracts the polysemous words that share the same father.

    Args:
        word_tree: A dictionary containing the tree of hyponyms.

    Returns:
        cofather_polysemy: A dictionary containing the polysemous words that share the same father.
        {
            word_name: {
                father: {
                    word_name.x.xx: definition,
                    word_name.x.xx: definition,
                },
                father: {
                    word_name.x.xx: definition,
                    word_name.x.xx: definition,
                },
            },
            ...
        }
    """
    
    # get the polysemous words' fathers of each original word
    polysemous_word = extract_polysemous_words(word_tree) 
    noun_data = construct_flatten_wordnet_noun()
    polysemy_father_pairs = {}
    for word_name in polysemous_word:
        polysemy_father_pairs[word_name] = {}
        for every_polysemy in polysemous_word[word_name]:
            polysemy_father_pairs[word_name][every_polysemy] = noun_data[every_polysemy]["hypernyms"]
    # output the polysemous words' fathers
    
    cofather_polysemy = {}
    from collections import defaultdict

    for word_name, polysemy_fathers in polysemy_father_pairs.items():
        father_dict = defaultdict(list)
        for polysemy, fathers in polysemy_fathers.items():
            for father in fathers:
                father_dict[father].append(polysemy)
        filtered_father_dict = {father: {sense: noun_data[sense]["definition"] for sense in sense_list} for father, sense_list in father_dict.items() if len(sense_list) > 1}
        if filtered_father_dict:
            cofather_polysemy[word_name] = filtered_father_dict
    print("totally same father polysemous words:", len(cofather_polysemy.keys()))
    return cofather_polysemy



def rename_word_tree(word_tree):
    """Renames the keys of the word tree to include the part of speech and number.

    Args:
        word_tree: A dictionary containing the tree of hyponyms.

    Returns:
        A dictionary containing the tree of hyponyms with renamed keys.
    """
    polysemous_words = extract_polysemous_words(word_tree)
    noun_data = construct_flatten_wordnet_noun()
    polysemous_word_list = [word for sublist in polysemous_words.values() for word in sublist.keys()]

    def rename_hyponyms(hyponyms):
        if hyponyms is None:
            return None

        new_hyponyms = {}
        for hyponym, details in hyponyms.items():
            if hyponym not in polysemous_word_list:
                word_items = hyponym.split('.')
                new_key = '.'.join(word_items[:-2]) if len(word_items) > 3 else word_items[0]
            else:
                word_items = hyponym.split('.')
                word = '.'.join(word_items[:-2]) if len(word_items) > 3 else word_items[0]
                
                father_of_this_word = noun_data[hyponym]["hypernyms"][0].split('.')
                father_of_this_word = '.'.join(father_of_this_word[:-2]) if len(father_of_this_word) > 3 else father_of_this_word[0]
                
                new_key = f"{word} ({father_of_this_word})"

            new_hyponyms[new_key] = {
                "definition": details["definition"],
                "level": details["level"],
                "hyponyms": rename_hyponyms(details["hyponyms"])
            }
        return new_hyponyms

    return {"hyponyms": rename_hyponyms(word_tree.get("hyponyms"))}







