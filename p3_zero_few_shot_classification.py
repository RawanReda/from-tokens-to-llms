import p2_feature_and_classification
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from textwrap import dedent
from sklearn.metrics import f1_score, classification_report
import numpy as np
import torch
from transformers.utils import logging

logging.set_verbosity_error()

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

parties = df['party'].value_counts().index.tolist()

def print_f1_score_and_classification_model(y_test, y_pred, model_name=""):
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"{model_name} Macro F1 Score: ", f1)
    print(f"{model_name} Classification Report: \n", report)

def print_with_lines(text):
    print("=" * 100)
    print(text)
    print("=" * 100)

# Prompt builders
# Candidate labels help constrain the set of valid outputs.
# The second instruction forces the model to answer using only one of the party names,
# which reduces open-ended continuation.

def build_zero_shot_prompt(text, candidate_labels):
    return "\n".join([
        f"Classify this speech into one of these parties: {', '.join(candidate_labels)}",
        "Answer only with one of these party names.",
        "",
        f"Speech: {text}",
        "",
        "Party:"
    ]).strip()


def build_few_shot_prompt(speech, candidate_labels, few_shot_examples_text):
    return "\n".join([
        f"Classify this speech into one of these parties: {', '.join(candidate_labels)}",
        "Answer only with one of these party names.",
        "",
        "Examples:",
        few_shot_examples_text,
        "",
        "Now classify this speech:",
        f"Speech: {speech}",
        "",
        "Party:"
    ]).strip()


def extract_generated_answer(response, prompt):
    generated_text = response[0]["generated_text"]
    return generated_text[len(prompt):].strip().splitlines()[0].strip()

# Get unique labels for classification
unique_labels = df['party'].unique()
candidate_labels= unique_labels.tolist()


# Quen2.5 is a high performance model, with long context support and is highly performant 
#  It uses a decoder-only architecture which supports in-context learning and is also instruction-tuned, enabling both zero-shot and few-shot classification through prompting.
model = 'Qwen/Qwen2.5-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model)
generator = pipeline(
   'text-generation',
   model=model,
   tokenizer=tokenizer,
   dtype=torch.bfloat16,
   device_map='auto',
)

# zero shot
y_pred = []
prompt_printed = False
for text in X_test.tolist():
    prompt = build_zero_shot_prompt(text, candidate_labels)

    if not prompt_printed:
        print("zero-shot prompt:")
        print_with_lines(prompt)

    # max_new_tokens set to a small number, since the expected output is only one token for each prompt 
    # do_sample is False because we want the response with the highest probability.
    response = generator(prompt, max_new_tokens=4, do_sample=False)
    generated_answer = extract_generated_answer(response, prompt)

    if not prompt_printed:
        print_with_lines(generated_answer)
        prompt_printed = True

    predicted = "Unknown"
    
    for party in candidate_labels:
        if party.lower() in generated_answer.lower():
            predicted = party
            break
    
    y_pred.append(predicted)

print_f1_score_and_classification_model(y_test, y_pred, "zero shot")



# few shot 
example_indices = []
for party in y_train.unique():
    #  picking 2 rows for each party value would ensure fairness in th e examples provided and reduce bias.
    party_indices = y_train[y_train == party].sample(n=2, random_state=26).index.tolist() 
    example_indices.extend(party_indices)

few_shot_examples = []
for idx in example_indices:
    speech = X_train.loc[idx]
    label = y_train.loc[idx]
    few_shot_examples.append(f"Speech: {speech}\nParty: {label}")

few_shot_examples_text = "\n\n".join(few_shot_examples)

y_pred = []
prompt_printed = False
for speech in X_test.tolist():
    # same as in zero-shot, candidate labels are provided and examples are given from a balanced list where there are 2 examples for each output.
    prompt = build_few_shot_prompt(speech, candidate_labels, few_shot_examples_text)

    if not prompt_printed:
        print("few-shot prompt:")
        print_with_lines(prompt)

    response = generator(prompt, max_new_tokens=4, do_sample=False)
    generated_answer = extract_generated_answer(response, prompt)

    if not prompt_printed:
        print_with_lines(generated_answer)
        prompt_printed = True

    # just in case it hallucinates we set a default of 'unknown'
    predicted = "Unknown" 

    for party in candidate_labels:
        if party.lower() in generated_answer.lower():
            predicted = party
            break

    y_pred.append(predicted)

print_f1_score_and_classification_model(y_test, y_pred, "few shot")


# Based on the F1 score and classification report results, few-shot achieved a higher macro F1 score (52%)
# compared to zero-shot (27%). This suggests that few-shot improved overall performance.
# In both zero-shot and few-shot, the Conservative party had the highest F1 score.
# Both approaches had similar F1 results for Labour.
# Few-shot performed better for the Scottish National Party, with an F1 score of 50%, while zero-shot did not predict that party at all.
# In fact, zero-shot failed to predict Scottish National Party, but produced comparable results to few-shot for Conservative and Labour.
 