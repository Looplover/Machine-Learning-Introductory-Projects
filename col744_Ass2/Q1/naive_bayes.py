import numpy as np
import pandas as pd
from tqdm.auto import tqdm

class NaiveBayes:
    def __init__(self):
        """Initializes all the necessary model parameters."""
        self.class_priors = None
        self.word_lhoods = None
        self.vocab = None
        self.smoothening = None
        self.classes_ = None
        self.class_map_ = None
        self.num_classes = None
        self.vocab_index = None
        
    def fit(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        """
        Learn the parameters of the model from the training data.
        Classes are 1-indexed.

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        self.smoothening = smoothening
        
        # Create a mapping from class labels to 0-based indices
        self.classes_ = np.sort(df[class_col].unique())
        self.num_classes = len(self.classes_)
        self.class_map_ = {label: i for i, label in enumerate(self.classes_)}
        
        # Calculate class priors for value_counts
        class_cnts = df[class_col].value_counts().reindex(self.classes_).fillna(0)
        self.class_priors = np.log(class_cnts / len(df))
        
        # Build the vocabulary
        print("Building vocabulary...")
        self.vocab = set(word for tokens_list in df[text_col] for word in tokens_list)
        self.vocab = sorted(list(self.vocab), key=str) # key=str handles bigrams (tuples)
        self.vocab_index = {word: i for i, word in enumerate(self.vocab)}
        vocab_size = len(self.vocab)
        
        # Tally word counts for each class
        word_counts_per_class = np.zeros((self.num_classes, vocab_size))
        total_words_per_class = np.zeros(self.num_classes)
        
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fitting Model"):
            label = row[class_col]
            class_idx = self.class_map_[label]
            tokens = row[text_col]
            
            total_words_per_class[class_idx] += len(tokens)
            for token in tokens:
                if token in self.vocab_index:
                    token_idx = self.vocab_index[token]
                    word_counts_per_class[class_idx, token_idx] += 1
        
        # Calculate smoothed log-likelihoods
        numerator = word_counts_per_class + self.smoothening
        denominator = total_words_per_class + self.smoothening * vocab_size
        self.word_lhoods = np.log(numerator / denominator[:, np.newaxis])

    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        predictions = []
        
        for tokens in tqdm(df[text_col], desc="Predicting Classes"):
            # Start with the log priors for each class
            log_posterior = self.class_priors.copy()
            
            # Add the log likelihoods for each token
            for token in tokens:
                if token in self.vocab_index:
                    token_idx = self.vocab_index[token]
                    log_posterior += self.word_lhoods[:, token_idx]
            
            # Find the index of the class with the highest probability
            predicted_index = np.argmax(log_posterior)
            # Map the index back to the original class label
            predictions.append(self.classes_[predicted_index])
        
        df[predicted_col] = predictions
        
    def accuracy(self, df, class_col="Class Index", predicted_col="Predicted"):
        """
        Computes the accuracy of the predictions.
        """
        correct = (df[class_col] == df[predicted_col]).sum()
        total = len(df)
        return correct / total if total > 0 else 0