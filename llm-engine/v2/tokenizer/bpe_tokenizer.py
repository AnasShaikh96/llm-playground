from collections import defaultdict

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}

    def train(self, text: str, num_merges: int):

        corpus = self._build_initial_corpus(text)

        initial_corpus = [word[:] for word in corpus]

        for i in range(num_merges):

            pair_counts = self._count_pairs(corpus)

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)

            merged_token = "".join(best_pair)

            self.merges[best_pair] = merged_token

            corpus = self._merge_pair(corpus, best_pair)

            print(
                f"Merge {i+1}: {best_pair} -> {merged_token}"
            )

        self._build_vocab(initial_corpus)

        return corpus
    
    def _merge_pair(self, corpus, pair):
        """
        Replace every occurrence of `pair` in the corpus with the merged token.

        Example:

        pair = ("l", "o")

        [
            ["l", "o", "w"],
            ["l", "o", "w", "e", "r"]
        ]

        becomes

        [
            ["lo", "w"],
            ["lo", "w", "e", "r"]
        ]
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

    def _build_vocab(self, initial_corpus):
        """
        Build the vocabulary from:
        - all original characters
        - every merged token
        """

        tokens = set()

        for word in initial_corpus:
            tokens.update(word)

        tokens.update(self.merges.values())

        tokens = sorted(tokens)

        self.vocab = {
            token: idx
            for idx, token in enumerate(tokens)
        }

        self.inverse_vocab = {
            idx: token
            for token, idx in self.vocab.items()
        }

    def encode(self, text):

        tokens = list(text)

        for pair in self.merges.keys():
            tokens = self._apply_merge(tokens, pair)

        return [
            self.vocab[token]
            for token in tokens
        ]

    def decode(self, ids):

        tokens = [
            self.inverse_vocab[token_id]
            for token_id in ids
        ]

        return "".join(tokens)

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

    def _apply_merge(self, tokens, pair):

        merged = "".join(pair)

        result = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                result.append(merged)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        return result


if __name__ == "__main__":

    tokenizer = BPETokenizer()

    tokenizer.train(
        "low low lower lowest",
        num_merges=10
    )

    print("\nVocabulary")
    print(tokenizer.vocab)

    encoded = tokenizer.encode("lower")

    print("\nEncoded")
    print(encoded)

    decoded = tokenizer.decode(encoded)

    print("\nDecoded")
    print(decoded)