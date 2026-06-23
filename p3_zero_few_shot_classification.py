import p2_feature_and_classification
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import numpy as np
import torch

file_path = "cw-pack-2026/texts/hansard500.csv"
df = p2_feature_and_classification.read_data(file_path)


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

def print_f1_score_and_classification_model(y_test, y_pred, model_name=""):
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"{model_name} Macro F1 Score: ", f1)
    print(f"{model_name} Classification Report: \n", report)

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

model = 'HuggingFaceTB/SmolLM2-135M-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model)
generator = pipeline(
   'text-generation',
   model=model,
   tokenizer=tokenizer,
   torch_dtype=torch.bfloat16,
   device_map='auto',
)

generator.model.generation_config.max_length = None

example_indices = [0,6,8]
few_shot_examples = ""
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

    response = generator(prompt, max_new_tokens=4, max_length=None, do_sample=False)
    generated_text = response[0]["generated_text"]
    generated_answer = generated_text[len(prompt):].strip()
    predicted = "Unknown"

    for party in candidate_labels:
        if party.lower() in generated_answer.lower():
            predicted = party
            break

    y_pred.append(predicted)


print("few shot ypred", y_pred)
print_f1_score_and_classification_model(y_test, y_pred, "few shot")