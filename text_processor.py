import pandas as pd
import os
import re
import html

class TextProcessor:
    def __init__(self, article_folder_path):
        self.article_folder_path = article_folder_path

    def read_custom_file(self, file_path, data_type):
        """
        Read custom formatted file with 6 columns
        """
        if data_type == "train":
            cleaned_data = []
            with open(file_path, "r") as file:
                for line in file:
                    parts = line.strip().split("\t")
                    fixed_columns = parts[:5]
                    fine_grained_roles = parts[5:]  # The remaining parts are the fine-grained roles
                    fixed_columns.append(fine_grained_roles)  # Store as a list, not as a string
                    cleaned_data.append(fixed_columns)

            return pd.DataFrame(cleaned_data, columns=["article_id", "entity_mention", "start_offset", "end_offset", 
                                                       "main_role", "fine_grained_roles"])
        else:
            cleaned_data = []
            with open(file_path, "r") as file:
                for line in file:
                    parts = line.strip().split("\t")
                    fixed_columns = parts[:4]
                    cleaned_data.append(fixed_columns)

            return pd.DataFrame(cleaned_data, columns=["article_id", "entity_mention", "start_offset", "end_offset"])

    def add_article_text(self, data):
        """
        Add article text to the dataframe based on article_id
        """
        def extract_text(row):
            article_path = os.path.join(self.article_folder_path, row["article_id"])
            if os.path.exists(article_path):
                with open(article_path, "r", encoding="utf-8") as file:
                    return file.read()
            return None

        data["article_text"] = data.apply(extract_text, axis=1)
        return data

    def extract_context(self, row, window_size=50):
        """
        Extract context around entity mention
        """
        if pd.isna(row["article_text"]):
            return None

        start_offset = int(row["start_offset"]) if isinstance(row["start_offset"], str) else row["start_offset"]
        end_offset = int(row["end_offset"]) if isinstance(row["end_offset"], str) else row["end_offset"]

        text = row["article_text"]
        start = max(0, start_offset - window_size)
        end = min(len(text), end_offset + window_size)
        return self.clean_text(text[start:end])

    def add_context(self, data, window_size=50):
        """
        Add context to the dataframe
        """
        data["context"] = data.apply(self.extract_context, axis=1, window_size=window_size)
        return data

    def clean_text(self, text):
        """
        Comprehensive text cleaning function
        """
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = html.unescape(text)  # Decode HTML entities
        text = re.sub(r'https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters, digits, and punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces, tabs, and newlines
        text = text.strip()  # Remove leading/trailing whitespace
        text = text.lower()  # Convert to lowercase

        return text

    def extract_paragraph_by_keyword(self, article_text, keyword):
        """
        Extract the paragraph containing the keyword from the article text.
        """
        if not isinstance(article_text, str):
            return "No passage found."
        paragraphs = article_text.split('\n\n')
        for paragraph in paragraphs:
            if keyword in paragraph:
                return paragraph.strip()
        return "No passage found."

    def extract_paragraph_by_entity(self, article_text, entity_mention):
        """
        Extract the paragraph containing the entity_mention from the article text.
        """
        if not isinstance(article_text, str) or not isinstance(entity_mention, str):
            return "No passage found."
        paragraphs = article_text.split('\n\n')
        for paragraph in paragraphs:
            if entity_mention in paragraph:
                return paragraph.strip()
        return "No passage found."


    def add_entity_passage_column(self, data):
        """
        Add a column to the dataframe containing the extracted and cleaned passage with the entity mention.
        """
        # Apply the extraction function and clean the text before adding it to the dataframe
        data["entity_passage"] = data.apply(
            lambda row: self.clean_text(self.extract_paragraph_by_entity(row["article_text"], row["entity_mention"])), axis=1
        )
        return data
