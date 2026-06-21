import p2_feature_and_classification
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import numpy as np


file_path = r'..\cw-pack-2026\cw-pack-2026\texts\hansard500.csv'
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
for text in X_train.tolist():
    y_pred.append(classifier(text, candidate_labels)["labels"][0])

print_f1_score_and_classification_model(y_pred, y_test)



## few shot 

model = 'microsoft/Phi-4-mini-instruct'
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
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
print_f1_score_and_classification_model(y_test, y_pred)


# model_name = "arnir0/Tiny-LLM"
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# def tokenize_input(model_inputs):
#     tokenized = tokenizer(model_inputs["text"], padding="max_length", truncation=True, max_length=512)
#     tokenized["label"] = [label2id[label] for label in model_inputs["label"]]
#     return tokenized

# tokenized_train_dataset = train_dataset.map(tokenize_input, batched=True, remove_columns=train_dataset.column_names)
# tokenized_test_dataset = test_dataset.map(tokenize_input, batched=True, remove_columns=test_dataset.column_names)


# training_args = TrainingArguments(
#     output_dir="trained_model_args",
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=8,
#     gradient_checkpointing=True,
#     learning_rate=2e-5,
#     logging_steps=10,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_test_dataset,
#     processing_class=tokenizer,
#     data_collator=DataCollatorWithPadding(tokenizer),
# )

# trainer.train()

# model.save_pretrained("./fine-tuned-model")
# tokenizer.save_pretrained("./fine-tuned-model")

# # Make predictions on test set
# test_encodings = tokenizer(X_test.tolist(), padding="max_length", truncation=True, max_length=512, return_tensors="pt")
# test_predictions = model(**test_encodings)
# predicted_labels = np.argmax(test_predictions.logits.detach().numpy(), axis=1)

# print(test_predictions)
# print(predicted_labels)
# print("Predictions sample:", [id2label[pred] for pred in predicted_labels[:5]])

# # Convert predicted indices to labels
# y_pred = [id2label[pred] for pred in predicted_labels]

# # Calculate metrics
# f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
# report = classification_report(y_test, y_pred, zero_division=0)
# print(f"\nF1 Score: {f1}")
# print("\nClassification Report:")
# print(report)





    # mode for zero-shot classification https://huggingface.co/cross-encoder/nli-distilroberta-base