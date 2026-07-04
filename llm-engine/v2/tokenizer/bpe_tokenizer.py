from collections import defaultdict


class BPETokenizer:
    def __init__(self):
        # token -> id
        self.vocab = {}

        # (token1, token2) -> merged_token
        self.merges = {}

    def train(self, text: str, num_merges: int):
        pass

    def encode(self, text: str):
        pass

    def decode(self, ids):
        pass

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass

    def _build_initial_corpus(self, text: str):
        """
        Converts text into a list of words,
        where every word is represented as characters.

        Example:

        "low lower"

        becomes

        [
            ["l","o","w"],
            ["l","o","w","e","r"]
        ]
        """

        corpus = []

        for word in text.split():
            corpus.append(list(word))

        return corpus

    def _count_pairs(self, corpus):
        """
        Counts every adjacent token pair.
        """

        pair_counts = defaultdict(int)

        for word in corpus:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += 1

        return dict(pair_counts)

    def _merge_pair(self, corpus, pair):
        """
        Replaces every occurrence of a token pair with a merged token.

        Example:

        pair = ("l", "o")

        ["l","o","w"]

        becomes

        ["lo","w"]
        """

        merged_token = "".join(pair)

        new_corpus = []

        for word in corpus:
            new_word = []

            i = 0

            while i < len(word):

                if (
                    i < len(word) - 1
                    and word[i] == pair[0]
                    and word[i + 1] == pair[1]
                ):
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_corpus.append(new_word)

        return new_corpus


if __name__ == "__main__":

    tokenizer = BPETokenizer()

    corpus = tokenizer._build_initial_corpus(
        "low low lower lowest below swallow"
    )

    print("Initial Corpus")
    print(corpus)

    pairs = tokenizer._count_pairs(corpus)

    print("\nPair Counts")
    for pair, count in sorted(pairs.items()):
        print(pair, count)

    best_pair = max(pairs, key=pairs.get)

    print(f"\nBest Pair: {best_pair}")

    corpus = tokenizer._merge_pair(corpus, best_pair)

    print("\nAfter Merge")
    print(corpus)