from datasets import get_dataset_config_names
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from datasets import load_dataset
import os

# Q1
# configs = get_dataset_config_names('ai4bharat/naamapadam')
# print(configs)
# print("Total config of the dataset is: ", len(configs))

# dataset = load_dataset('ai4bharat/naamapadam', 'hi') # it returns a dictionary
# print(dataset)
# print(dataset['train'].shape)

ds = load_dataset('ai4bharat/naamapadam', 'ta')
# print(ds)

# List of NER label
# print(dataset['train'].features['ner_tags'].feature.names)

# Q2
# # Print the cache files
# # Get cache file paths
# cache_files = ds.cache_files
# for key in cache_files:
#   print(cache_files[key][0])

# Q3
# Calculate total size in MB
# total_size = sum(os.path.getsize(cache_files[key][0]['filename']) for key in cache_files) / (1024 * 1024)
# print(f"Total dataset size: {total_size:.2f} MB")

# Q4
# print(ds['train'].shape)

# Define a function to compute token counts

# # Q5
# def compute_num_tokens(example):
#     return {"num_tokens": len(example["tokens"])}

# # Apply the function to all splits
# ds = ds.map(compute_num_tokens)
# total_tokens = sum(sum(example["num_tokens"] for example in ds[split]) for split in ds)
# # # Convert to millions
# print(f"Total tokens: {total_tokens / 1_000_000:.2f} million")

# # Q6
# # Calculate total size in MB
# cache_files = ds.cache_files
# total_size = sum(os.path.getsize(cache_files[key][0]['filename']) for key in cache_files) / (1024 * 1024)
# print(f"Total dataset size: {total_size:.2f} MB")


# # Load all splits
# Q7
# Flatten indices for each split
train_ds = ds["train"].flatten_indices()
test_ds = ds["test"].flatten_indices()
validation_ds = ds["validation"].flatten_indices()

# Concatenate splits in order: [train, test, validation]
ds = concatenate_datasets([train_ds, test_ds, validation_ds])

# Create "text" column by joining tokens with space
ds = ds.map(lambda example: {"text": " ".join(example["tokens"])})

# Remove "ner_tags" and "tokens" columns
ds = ds.remove_columns(["ner_tags", "tokens"])

# Total number of samples
print(f"Total samples after concatenation: {len(ds)}")


# # Q8
# cache_files = ds.cache_files
# print(cache_files)

# Q9
# # Filter samples with at least six tokens
ds = ds.filter(lambda x: len(x["text"].split()) >= 6)

# # Get the number of samples after filtering
# print(len(ds))

# Q10


# Load the datasets
indic_glue_dataset = load_dataset("ai4bharat/indic_glue", "inltkh.ta")

indic_glue_train = indic_glue_dataset['train']
indic_glue_test = indic_glue_dataset['test']
indic_glue_validation = indic_glue_dataset['validation']

indic_glue_dataset = concatenate_datasets([indic_glue_train, indic_glue_test, indic_glue_validation])
indic_glue_dataset = indic_glue_dataset.filter(lambda x: len(x['text'].split())>=6)
# Filter function to keep samples with at least 6 words
# def filter_by_word_count(example, field="text"):
#     return len(example[field].split()) >= 6

# # Apply filtering to all splits of indic_glue dataset
# filtered_indic_glue = []
# for split in indic_glue_dataset:
#     filtered_split = indic_glue_dataset[split].filter(lambda x: filter_by_word_count(x, "text"))
#     filtered_indic_glue.append(filtered_split)
# filtered_indic_glue = sum(filtered_indic_glue)  # combine all splits

# # Filter naamapadam dataset
# filtered_naamapadam = naamapadam_dataset["train"].filter(lambda x: filter_by_word_count(x, "text"))

print(ds)
print(indic_glue_dataset)

# Interleave datasets with the given proportions
interleaved_dataset = interleave_datasets(
    [ds, indic_glue_dataset],
    probabilities=[0.8, 0.2],
    seed=42
)

# # Output the number of samples in the final dataset
print(interleaved_dataset.shape)
