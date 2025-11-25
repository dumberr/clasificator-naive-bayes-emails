import csv
import math
import random


def tokenize(text):
    text = text.lower()
    new = []
    for ch in text:
        if ch.isalnum() or ch == " ":
            new.append(ch)
        else:
            new.append(" ")
    clean = "".join(new)
    words = clean.split()
    return words


def load_dataset(csv_path):
    data = []
    with open(csv_path, "r", encoding = "utf-8") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if first:
                first = False
                continue
            text = row[0].strip()
            label = row[1].strip()
            data.append((text, label))
    return data


def train_test_split(data, ratio=0.3, seed=42):
    random.seed(seed)
    random.shuffle(data)
    n_total = len(data)
    n_test = int(n_total * ratio)
    test_data = data[:n_test]
    train_data = data[n_test:]
    return train_data, test_data


class BayesMultinomial:
    def __init__(self):
        self.classes = []
        self.class_priors = {}
        self.word_counts = {}
        self.total_tokens = {}
        self.vocab = set()
        self.vocab_size = 0
        self.class_doc_counts = {}
        self.n_docs = 0

    def fit(self, texts, labels):
        self.classes = sorted(list(set(labels)))
        self.n_docs = len(labels)

        for c in self.classes:
            self.class_doc_counts[c] = 0
            self.word_counts[c] = {}
            self.total_tokens[c] = 0

        for text, label in zip(texts, labels):
            self.class_doc_counts[label] += 1
            words = tokenize(text)
            for w in words:
                self.vocab.add(w)
                if w not in self.word_counts[label]:
                    self.word_counts[label][w] = 0
                self.word_counts[label][w] += 1
                self.total_tokens[label] += 1

        self.vocab_size = len(self.vocab)

        for c in self.classes:
            self.class_priors[c] = self.class_doc_counts[c] / self.n_docs

    def word_prob(self, word, c):
        if word in self.word_counts[c]:
            count_wc = self.word_counts[c][word]
        else:
            count_wc = 0
        numarator = count_wc + 1.0
        numitor = self.total_tokens[c] + self.vocab_size
        return numarator / numitor

    def predict_one(self, text):
        words = tokenize(text)
        best_class = None
        best_log_prob = None

        for c in self.classes:
            log_prob_c = math.log(self.class_priors[c])
            for w in words:
                p_wc = self.word_prob(w, c)
                log_prob_c += math.log(p_wc)
            if (best_log_prob is None) or (log_prob_c > best_log_prob):
                best_log_prob = log_prob_c
                best_class = c

        return best_class

    def predict(self, texts):
        results = []
        for t in texts:
            cl = self.predict_one(t)
            results.append(cl)
        return results


def accuracy_score(y_true, y_pred):
    corect = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            corect += 1
    if len(y_true) == 0:
        return 0.0
    else:
        return corect / len(y_true)


def main():
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "emails.csv" 

    print("incarc setul de date din:", csv_path)
    data = load_dataset(csv_path)

    if not data:
        print("fisierul nu are date.")
        return

    train_data, test_data = train_test_split(data, ratio=0.3, seed=42)

    train_texts = [t for (t, lbl) in train_data]
    train_labels = [lbl for (t, lbl) in train_data]

    test_texts = [t for (t, lbl) in test_data]
    test_labels = [lbl for (t, lbl) in test_data]

    print("nr exemple train:", len(train_data))
    print("nr exemple test:", len(test_data))

    model = BayesMultinomial()
    model.fit(train_texts, train_labels)

    preds = model.predict(test_texts)
    acc = accuracy_score(test_labels, preds)
    print(f"accuracy pe test: {acc * 100:.2f}%")

    #test
    for i in range(min(5, len(test_texts))):
        print("txt:", test_texts[i])
        print("real:", test_labels[i])
        print("prediction:", preds[i])

    #test
    print("input:")
    while True:
        line = input("> ")
        line = line.strip()
        if line == "":
            break
        predicted = model.predict_one(line)
        print("prediction:", predicted)



if __name__ == "__main__":
    main()
