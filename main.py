import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from text_processor import TextProcessor
from bert_embeddings import BertEmbeddings
from sklearn.neural_network import MLPClassifier
from skmultilearn.problem_transform import LabelPowerset

def main():
    # Initialize classes
    article_folder_path = 'target_4_December_release/EN/raw-documents'
    file_path = 'target_4_December_release/EN/subtask-1-annotations.txt'

    text_processor = TextProcessor(article_folder_path)
    bert_embeddings = BertEmbeddings()

    # Step 1: Read and Process Data
    data = text_processor.read_custom_file(file_path, "train")
    data = text_processor.add_article_text(data)
    data = text_processor.add_context(data, window_size=50)
    data = text_processor.add_entity_passage_column(data)

    # Process labels
    mlb = MultiLabelBinarizer()
    train_label = mlb.fit_transform(data["fine_grained_roles"])
    train_label_df = pd.DataFrame(train_label, columns=mlb.classes_)

    data = pd.concat([data, train_label_df], axis=1)

    # Step 2: Split Data into Train and Test sets
    X_train, X_test = train_test_split(data, test_size=0.1, random_state=42, stratify=data["main_role"])

    # Step 3: Get BERT embeddings
    X_train_e = np.array([bert_embeddings.get_bert_embedding(doc, embedding_strategy='mean') for doc in X_train["entity_passage"]])
    X_test_e = np.array([bert_embeddings.get_bert_embedding(doc, embedding_strategy='mean') for doc in X_test["entity_passage"]])

    # Step 4: Normalize embeddings
    scaler = StandardScaler()
    X_train_e = scaler.fit_transform(X_train_e)
    X_test_e = scaler.transform(X_test_e)

    # Step 5: Prepare labels
    y_train = X_train.drop(['article_id', 'entity_mention', 'start_offset', 'end_offset',
           'main_role', 'fine_grained_roles', 'article_text', 'context',
           'entity_passage'], axis=1)

    y_test = X_test.drop(['article_id', 'entity_mention', 'start_offset', 'end_offset',
           'main_role', 'fine_grained_roles', 'article_text', 'context',
           'entity_passage'], axis=1)

    # Step 6: Create and train the model
    lp = LabelPowerset(MLPClassifier(hidden_layer_sizes=(200, 20), max_iter=500, random_state=42))
    lp.fit(X_train_e, y_train)

    # Step 7: Make predictions and evaluate
    y_pred_sparse = lp.predict(X_test_e)
    y_pred = y_pred_sparse.toarray()

    # Step 8: Calculate metrics
    exact_match = np.mean((y_test == y_pred).all(axis=1))
    print(f"Exact Match Ratio: {exact_match:.5f}")

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.5f}")
    print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.5f}")

if __name__ == "__main__":
    main()
