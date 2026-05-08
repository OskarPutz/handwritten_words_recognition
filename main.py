import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter, defaultdict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from skimage.transform import resize
import seaborn as sns

DATASET_DIR = "HPT_handwritten_polish_text_dataset"
NUM_AUTHORS = 8
NUM_CLASSES = 10
IMG_SIZE = (32, 64)   # (height, width) — words are wider than tall
MAX_DEPTH = 4
N_ESTIMATORS = 100
LEARNING_RATE = 1.0
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def select_top_words(dataset_dir, n=NUM_CLASSES):
    """Return top-N most frequent words that appear in every author's data."""
    author_words = defaultdict(set)
    word_counts = Counter()

    for author_id in range(1, NUM_AUTHORS + 1):
        txt_path = os.path.join(dataset_dir, f"author{author_id}", "word_places.txt")
        with open(txt_path, encoding="windows-1250") as f:
            for line in f:
                if line.startswith("%") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    word = " ".join(parts[1:-4]).lower()
                    if word:
                        author_words[author_id].add(word)
                        word_counts[word] += 1

    common = set.intersection(*author_words.values())
    top_n = [w for w, _ in word_counts.most_common() if w in common][:n]
    return top_n


def load_annotations(dataset_dir, target_words):
    """Parse all word_places.txt files and return list of (img_path, word, r1, c1, r2, c2)."""
    target_set = set(target_words)
    records = []

    for author_id in range(1, NUM_AUTHORS + 1):
        author_dir = os.path.join(dataset_dir, f"author{author_id}")
        txt_path = os.path.join(author_dir, "word_places.txt")

        with open(txt_path, encoding="windows-1250") as f:
            for line in f:
                if line.startswith("%") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                img_rel = parts[0].strip('"').replace("\\", os.sep)
                r1, c1, r2, c2 = int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1])
                word = " ".join(parts[1:-4]).lower()
                if not word or word not in target_set:
                    continue
                img_path = os.path.join(author_dir, img_rel)
                records.append((img_path, word, r1, c1, r2, c2))

    return records


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(crop):
    """Convert crop to fixed-size grayscale, then extract HOG features."""
    if crop.ndim == 3:
        gray = np.mean(crop, axis=2)
    else:
        gray = crop.astype(float)

    img = resize(gray, IMG_SIZE, anti_aliasing=True)
    rng = img.max() - img.min()
    img = (img - img.min()) / (rng + 1e-8)

    features = hog(
        img,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )
    return features


def build_feature_matrix(records):
    """Load images (cached), crop words, extract HOG features."""
    image_cache = {}
    X, y = [], []

    for img_path, word, r1, c1, r2, c2 in records:
        if img_path not in image_cache:
            image_cache[img_path] = np.array(Image.open(img_path))
        crop = image_cache[img_path][r1:r2, c1:c2]
        if crop.size == 0:
            continue
        X.append(extract_features(crop))
        y.append(word)

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_class_distribution(y, classes, save_path="class_distribution.png"):
    counts = [np.sum(y == w) for w in classes]
    plt.figure(figsize=(10, 4))
    plt.bar(classes, counts, color="steelblue")
    plt.xlabel("Word")
    plt.ylabel("Count")
    plt.title("Class distribution (top 10 words, all authors)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — AdaBoost SAMME")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_learning_curve(clf, X_train, y_train, X_test, y_test,
                        save_path="learning_curve.png"):
    """Use staged_predict — no retraining needed."""
    train_accs, test_accs = [], []

    for y_tr, y_te in zip(clf.staged_predict(X_train), clf.staged_predict(X_test)):
        train_accs.append(accuracy_score(y_train, y_tr))
        test_accs.append(accuracy_score(y_test, y_te))

    estimators = range(1, len(train_accs) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(estimators, train_accs, label="Train")
    plt.plot(estimators, test_accs, label="Test")
    plt.xlabel("Number of estimators")
    plt.ylabel("Accuracy")
    plt.title("AdaBoost SAMME — Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Select classes
    print("Selecting top 10 words common to all authors...")
    top_words = select_top_words(DATASET_DIR)
    print(f"Classes: {top_words}\n")

    # 2. Load and featurise
    print("Loading annotations...")
    records = load_annotations(DATASET_DIR, top_words)
    print(f"Annotated samples for selected classes: {len(records)}")

    print("Extracting HOG features (this may take ~1 min)...")
    X, y = build_feature_matrix(records)
    print(f"Feature matrix shape: {X.shape}\n")

    # 3. Encode labels + stratified split
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # 4. Class distribution
    plot_class_distribution(y, top_words)

    # 5. Train AdaBoost SAMME
    print(f"Training AdaBoost SAMME (n_estimators={N_ESTIMATORS})...")
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH),
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    print("Training done.\n")

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 7. Confusion matrix
    plot_confusion_matrix(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        top_words,
    )

    # 8. Learning curve
    print("Generating learning curve...")
    plot_learning_curve(clf, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
