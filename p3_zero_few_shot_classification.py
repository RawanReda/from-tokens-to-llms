from p2_feature_and_classification import read_data, print_f1_score_and_classification_model
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import numpy as np
import torch

file_path = r'..\cw-pack-2026\cw-pack-2026\texts\hansard500.csv'
df = read_data(file_path)


X = df['speech']
y = df['party']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=26)

train_dataset = Dataset.from_dict({
    "text": X_train.tolist(),
    "label": y_train.tolist()
})

test_dataset = Dataset.from_dict({
    "text": X_test.tolist(),
    "label": y_test.tolist()
})


# Get unique labels for classification
unique_labels = df['party'].unique()
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(unique_labels)
candidate_labels=list(id2label.values())


# zeroshot
zero_shot_model= "knowledgator/comprehend_it-base"
classifier = pipeline(
    "zero-shot-classification",
    model=zero_shot_model)
y_pred = []
for text in X_test.tolist():
    y_pred.append(classifier(text, candidate_labels)["labels"][0])

print_f1_score_and_classification_model(y_test, y_pred, "zero shot")



## few shot 

model = 'microsoft/Phi-4-mini-instruct'
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
   'text-generation',
   model=model,
   tokenizer=tokenizer,
   torch_dtype=torch.bfloat16,
   device_map='auto',
)
 
num_examples =5 
examples_indices = [0,6,8]

for idx in example_indices:
    speech = X_train.iloc[idx]
    label = y_train.iloc[idx]
    few_shot_examples += f"Speech: {speech}\nParty: {label}\n\n"

y_pred = []
for speech in X_test.tolist():
    prompt=f""""
    Classify parliament speeches by party. 

    Examples:
    {few_shot_examples}

    Now classify this speech:
    Speech: {speech}

    Party: 
    """

    response = pipeline(prompt, max_length=100)
    for party in candidate_labels:
        if party in result:
            y_pred.append(party)
            break

print("few shot ypred", y_pred)
print_f1_score_and_classification_model(y_test, y_pred, "few shot")