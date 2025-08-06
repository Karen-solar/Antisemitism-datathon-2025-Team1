import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load annotated data (ensure path is correct)
df = pd.read_csv('../annotations/Datathon2025_team1_annotation.csv')

# Filter for labeled data only
df = df[df['label'].notna()]

# Prepare text and labels (adapt column names as needed)
texts = df['text'].tolist()
labels = df['label'].astype(int).tolist()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)

# Tokenization
encodings = tokenizer(texts, truncation=True, padding=True)

# Dataset class
class TweetDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

dataset = TweetDataset(encodings, labels)

# Train-test split
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
eval_dataset = dataset[train_size:]

# Trainer setup
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()