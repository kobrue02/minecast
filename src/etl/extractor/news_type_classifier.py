from abc import ABC
import requests
import time
import evaluate
import random
import pandas as pd
import numpy as np
import re
import pickle
import torch
import csv
import os

from src.etl.extractor.types import NewsType

from bs4 import BeautifulSoup
from fastembed import TextEmbedding
from torch import nn, Tensor
from word2vec import word2vec
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from tqdm import tqdm

import warnings
warnings.filterwarnings("error")

SUPPORTED_TOPICS = [
        "Drilling Commenced",
        "PEAs & Feasibility Studies",
        "Financial Results & Earnings"
    ]

class NewsTypeClassifier(ABC):
    _LabelsToNewsType: dict[str, NewsType] = {
        "Drilling Commenced": NewsType.DRILLING,
        "PEAs & Feasibility Studies": NewsType.PEA,
        "Financial Results & Earnings": NewsType.FINANCIAL,
        "10-K Report": NewsType.R_10K,
        "10-Q Report": NewsType.R_10Q,
        "PFS Report": NewsType.PFS,
        "FS Report": NewsType.FS,
        "PEA Report": NewsType.PEA,
    }
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.train_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.label2id: dict = None
        self.id2label: dict = None
        self.accuracy = evaluate.load("accuracy")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        if not os.path.exists("data/news_reports_labeled.csv"):
            scraper = NewsReportScraper()
            _ = scraper.scrape(output_file="data/news_reports_labeled.csv")
        self.load_data("data/news_reports_labeled.csv")

    def load_data(self, file_path):
        # Load and preprocess the data
        df = pd.read_csv(file_path)
        df['label'] = df['topic_name'].astype('category').cat.codes
        df['text'] = df['article_title'].astype(str)
        # Create label mappings
        categories = df['topic_name'].astype('category').cat.categories
        self.id2label = {i: label for i, label in enumerate(categories)}
        self.label2id = {label: i for i, label in self.id2label.items()}
        X_train, X_test, y_train, y_test = train_test_split(
            df[['text']], df['label'], test_size=0.2, random_state=42
        )
        train_dataset = Dataset.from_dict({
            'text': X_train['text'].tolist(),
            'label': y_train.tolist()
        })
        test_dataset = Dataset.from_dict({
            'text': X_test['text'].tolist(),
            'label': y_test.tolist()
        })
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def use_subset(self, n_labels: int = None, labels: list = None):
        """
        Use a subset of the dataset with n_labels number of labels.
        
        Args:
            n_labels (int): Number of labels to use
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset is not loaded.")
        
        def _filter_n_labels(n_labels):
            # Filter the dataset to only include the specified number of labels
            filtered_train_dataset = self.train_dataset.filter(
                lambda x: x['label'] < n_labels
            )
            # Update the test dataset as well
            filtered_test_dataset = self.test_dataset.filter(
                lambda x: x['label'] in filtered_train_dataset['label']
            )
            self.train_dataset = filtered_train_dataset
            self.test_dataset = filtered_test_dataset
            # Update label mappings
            self.id2label = {i: label for i, label in enumerate(self.id2label.values()) if i < n_labels}
            self.label2id = {label: i for i, label in self.id2label.items()}
        
        def _filter_label_list(labels):
            allowed_ids = {label: idx for idx, label in enumerate(labels)}
            # Remap function
            def remap(example):
                label_name = self.id2label[example['label']]
                new_label = allowed_ids[label_name]
                return {'label': new_label}

            filtered_train_dataset = self.train_dataset.filter(
                lambda x: self.id2label[x['label']] in labels
            ).map(remap)

            filtered_test_dataset = self.test_dataset.filter(
                lambda x: self.id2label[x['label']] in labels
            ).map(remap)

            self.train_dataset = filtered_train_dataset
            self.test_dataset = filtered_test_dataset
            self.id2label = {i: label for i, label in enumerate(labels)}
            self.label2id = {label: i for i, label in self.id2label.items()}

        
        if n_labels is not None:
            if n_labels > len(self.id2label):
                raise ValueError(f"n_labels {n_labels} exceeds the number of available labels.")
            _filter_n_labels(n_labels)
        elif labels is not None:
            if not all(label in self.id2label.values() for label in labels):
                raise ValueError("Some labels are not present in the dataset.")
            _filter_label_list(labels)
        else:
            raise ValueError("Either n_labels or labels must be provided.")
        
    def get_embeddings(self, text):
        """
        Get embeddings for the input text.
        
        Args:
            text (str): Input text to get embeddings for
            
        Returns:
            Tensor: Embeddings for the input text
        """
        raise NotImplementedError("Get embeddings method not implemented in base class.")

    def train(self, *args):
        raise NotImplementedError("Train method not implemented in base class.")
    
    def predict(self, *args):
        raise NotImplementedError("Predict method not implemented in base class.")
    
    def evaluate(self):
        """Evaluate the model using accuracy and loss."""
        if self.test_dataset is None:
            raise ValueError("Test dataset is not loaded.")
        
        references = np.array(self.test_dataset['label'])
        predictions = [
            self.predict(text) for text in tqdm(self.test_dataset['text'])
        ]
        predictions = np.array([self.label2id[pred] for pred in predictions])
        # Calculate accuracy
        accuracy = self.accuracy.compute(predictions=predictions, references=references)
        print(f"Accuracy: {accuracy['accuracy']:.4f}")
        # Generate classification report
        report = classification_report(references, predictions, target_names=self.id2label.values(), zero_division=0)
        print("Classification Report:")
        print(report)
    
    def save_model(self, model_path):
        raise NotImplementedError("Save model method not implemented in base class.")
    
    def load_model(self, model_path):
        raise NotImplementedError("Load model method not implemented in base class.")

class NewsReportScraper:

    def __init__(self, base_url="https://www.juniorminingnetwork.com"):
        self.base_url = base_url
        self.topics_url = f"{base_url}/mining-topics/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.results = []
    
    def _get_page_content(self, url):
        """Helper method to get the content of a page."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None
    
    def _extract_topics(self, html_content):
        """Extract all topic names and IDs from the HTML content."""
        topics = {}
        if not html_content:
            return topics
        
        soup = BeautifulSoup(html_content, 'html.parser')
        tags_ul = soup.find('ul', class_='tags')
        
        if not tags_ul:
            print("Could not find the tags list.")
            return topics
        
        for li in tags_ul.find_all('li'):
            a_tag = li.find('a')
            if a_tag and a_tag.get('href'):
                topic_name = a_tag.text.strip()
                href = a_tag.get('href')
                
                # Extract the topic ID from the href (after the # symbol)
                match = re.search(r'#([^#]+)$', href)
                if match:
                    topic_id = match.group(1)
                    topics[topic_name] = topic_id
        
        return topics
    
    def _extract_article_titles(self, html_content):
        """Extract all article titles from the HTML content."""
        article_titles = []
        if not html_content:
            return article_titles
        
        soup = BeautifulSoup(html_content, 'html.parser')
        for article in soup.find_all(class_='article-title'):
            title = article.text.strip()
            article_titles.append(title)
        
        return article_titles
    
    def save_to_csv(self, filename="mining_articles.csv"):
        """Save the results to a CSV file with just topic names and article titles."""
        df = pd.DataFrame(data=self.results)
        df.reset_index(inplace=True)
        print(df.head())
        # Save to CSV
        df.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of the scraped data."""
        print("\nSummary of Scraped Data:")
        print("-----------------------")
        total_articles = 0
        
        for item in self.results:
            topic_name = item['topic_name']
            article_title = item['article_title']
            print(f"Topic: {topic_name}, Article Title: {article_title}")
            total_articles += 1
        
        print("-----------------------")
        print(f"Total topics: {len(self.results)}")
        print(f"Total articles: {total_articles}")
    
    def scrape(self, delay=1, output_file="mining_articles.csv"):
        """
        Main driver method to scrape mining topics and article titles.
        
        Parameters:
        - delay (float): Time in seconds to wait between requests to avoid overloading the server
        - output_file (str): Filename to save the results as CSV
        
        Returns:
        - dict: The scraped data containing topics and article titles
        """
        print("Starting the scraping process...")
        
        # Step 1: Get all mining topics
        print("Fetching mining topics...")
        html_content = self._get_page_content(self.topics_url)
        topics = self._extract_topics(html_content)
        
        if not topics:
            print("No topics found. Exiting.")
            return {}
        
        print(f"Found {len(topics)} topics.")
        
        # Step 2: Get article titles for each topic
        for topic_name, topic_id in topics.items():
            if not topic_name in SUPPORTED_TOPICS:
                print(f"Skipping unsupported topic: {topic_name}")
                continue
            print(f"Fetching articles for topic: {topic_name} (ID: {topic_id})...")
            topic_url = f"{self.topics_url}#{topic_id}"
            
            html_content = self._get_page_content(topic_url)
            articles = self._extract_article_titles(html_content)
            
            for article in articles:
                # Clean the article title
                article = re.sub(r'\s+', ' ', article).strip()
                match topic_name:
                    case "PEAs & Feasibility Studies":
                        if any(
                            term.lower() in article.lower() for term in [" PEA", "Feasibility Study", "Preliminary Economic Assessment", " FS", " PFS"]
                        ):
                            self.results.append({
                                'topic_name': topic_name,
                                'article_title': article,
                            })
                    case "Drilling Commenced":
                        if any(
                            term.lower() in article.lower() for term in ["drilling commenced", "drilling program", "drilling results"]
                        ):
                            self.results.append({
                                'topic_name': topic_name,
                                'article_title': article,
                            })
                    case "Financial Results & Earnings":
                        if any(
                            term.lower() in article.lower() for term in ["financial", "earnings", "quarterly results"]
                        ):
                            self.results.append({
                                'topic_name': topic_name,
                                'article_title': article,
                            })
                    case "Project Update":
                        if any(
                            term.lower() in article.lower() for term in ["project update", "update"]
                        ):
                            self.results.append({
                                'topic_name': topic_name,
                                'article_title': article,
                            })
                    case _:
                        continue

            print(f"  Found {len(articles)} articles.")
            # Sleep to avoid overwhelming the server
            time.sleep(delay)
        self.print_summary()
        # Step 3: Save results as CSV and print summary
        if output_file:
            self.save_to_csv(output_file)
        return self.results
    
class JinaNewsTypeClassifier(NewsTypeClassifier):
    def __init__(self, model_path = None):
        super().__init__()
        self.embeddings = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, verbose=True)
    
    def get_embeddings(self, text):
        """Get embeddings for the input text."""
        embedding = self.embeddings.encode([text], task="classification")
        return embedding[0]
    
    def train(self):
        """Train the model."""
        if self.train_dataset is None:
            raise ValueError("Training dataset is not loaded.")
        
        # Get embeddings for the training data
        X_train = np.array(
            [self.get_embeddings(text) for text in tqdm(self.train_dataset['text'])]
            )
        y_train = np.array(self.train_dataset['label'])
        
        # Train the model
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict(self, text):
        """Predict the topic for the input text."""
        if self.model is None:
            raise ValueError("Model is not trained.")
        
        # Get embeddings for the input text
        embedding = self.get_embeddings(text)
        embedding = np.array(embedding).reshape(1, -1)
        
        # Predict the topic
        prediction = self.model.predict(embedding)
        predicted_topic = self.id2label[prediction[0]]
        return predicted_topic
    
class FastEmbedNewsTypeClassifier(NewsTypeClassifier):
    def __init__(self, model = None):
        super().__init__()
        self.embeddings = TextEmbedding()
        if model:
            self.model = model
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=1000, verbose=True
                )
    
    @classmethod
    def load_model(cls, model_path):
        """Load the model from a file."""
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} does not exist.")
        
        with open(model_path, 'rb') as f:
            attrs = pickle.load(f)
        instance = cls()
        instance.model = attrs['model']
        instance.embeddings = TextEmbedding()
        instance.id2label = attrs['id2label']
        instance.label2id = attrs['label2id']
        instance.train_dataset = attrs['train_dataset']
        instance.test_dataset = attrs['test_dataset']
        return instance
    
    def get_embeddings(self, text):
        embedding = self.embeddings.embed(text)
        return next(embedding)
    
    def train(self, save_to = None):
        """Train the model."""
        if self.train_dataset is None:
            raise ValueError("Training dataset is not loaded.")
        
        # Get embeddings for the training data
        X_train = np.array(
            [self.get_embeddings(text) for text in tqdm(self.train_dataset['text'])]
            )
        y_train = np.array(self.train_dataset['label'])
        
        # Train the model
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")
        if save_to:
            self.save_model(save_to)
            print("Model saved successfully.")

    def save_model(self, model_path):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("Model is not trained.")
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'id2label': self.id2label,
                'label2id': self.label2id,
                'train_dataset': self.train_dataset,
                'test_dataset': self.test_dataset
            }, f)
        print(f"Model saved to {model_path}")
    
    def predict(self, text):
        """Predict the topic for the input text."""
        if self.model is None:
            raise ValueError("Model is not trained.")
        
        # Get embeddings for the input text
        embedding = self.get_embeddings(text)
        embedding = np.array(embedding).reshape(1, -1)
        
        # Predict the topic
        prediction = self.model.predict(embedding)
        predicted_topic = self.id2label[prediction[0]]
        return predicted_topic
    
    def inference(self, text):
        """Get the probability of each class for the input text."""
        if self.model is None:
            raise ValueError("Model is not trained.")
        
        prediction = self.predict(text)
        news_type = self._LabelsToNewsType[prediction]
        return news_type


# Example usage
if __name__ == "__main__":
    model = SVC(kernel="sigmoid", probability=True)
    classifier = FastEmbedNewsTypeClassifier(model=model)
    classifier.use_subset(labels=SUPPORTED_TOPICS)  # Use a subset of the dataset
    classifier.train("models/fastEmbedModel.pkl")
    text = "NI 43-101 Technical Report and Prefeasibility Study for the Madsen Mine, Ontario, Canada Report prepared for West Red Lake Gold Mines Ltd. Prepared by SRK Consulting (Canada) Inc. CAPR003299 February 2025"
    predicted_topic = classifier.inference(text)
    print(f"Predicted topic: {predicted_topic}")
    #classifier.evaluate()
    FastEmbedNewsTypeClassifier.load_model("models/fastEmbedModel.pkl")
    # Example prediction
    text = "NI 43-101 Technical Report and Prefeasibility Study for the Madsen Mine, Ontario, Canada Report prepared for West Red Lake Gold Mines Ltd. Prepared by SRK Consulting (Canada) Inc. CAPR003299 February 2025"
    predicted_topic = classifier.inference(text)
    print(f"Predicted topic: {predicted_topic}")