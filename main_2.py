import os
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from skimage.transform import resize

DATA_DIR = "HPT_handwritten_polish_text_dataset"
NUM_AUTHORS = 8 # Total writers
NUM_CLASSES = 10 # Word count
IMG_SIZE = (32, 64) # Image size
MAX_DEPTH = 4 # Tree depth
N_ESTIMATORS = 100 # Tree count
LEARNING_RATE = 1.0 # Learning rate
TEST_SIZE = 0.2 # Test ratio
RANDOM_STATE = 42 # Random seed

# Data loading

def select_top_words(DATA_DIR, n=NUM_CLASSES):
    author_words = defaultdict(set)
    word_counts = Counter()

    for author_id in range(1, NUM_AUTHORS+1):
        txt_path = os.path.join(DATA_DIR, f"author{author_id}", "word_places.txt")
        with open(txt_path) as f:
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


def load_annotations(DATA_DIR, target_words):
    target_set = set(target_words)
    records = []

    for author_id in range(1, NUM_AUTHORS + 1):
        author_dir = os.path.join(DATA_DIR, f"author{author_id}")
        txt_path = os.path.join(author_dir, "word_places.txt")

        with open(txt_path) as f:
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

# Feature extraction

def extract_features(crop):
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


# Visualisation
# TO DO

def main():
    # Select classes
    print("Selecting top 10 words common to all authors")
    top_words = select_top_words(DATA_DIR)
    print(f"Classes: {top_words}\n")

    # Load and featurise
    records = load_annotations(DATA_DIR, top_words)
    print(f"Annotated samples for selected classes: {len(records)}")

    print("Extracting HOG features")
    X, y = build_feature_matrix(records)
    print(f"Feature matrix shape: {X.shape}\n")

    # Encode labels + stratified split
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train AdaBoost SAMME
    print(f"training AdaBoost SAMME n_estimators={N_ESTIMATORS}")
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH),
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    print("training done\n")

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix TO DO

if __name__ == "__main__":
    main()
