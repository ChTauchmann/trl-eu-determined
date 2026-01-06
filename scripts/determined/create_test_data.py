#!/usr/bin/env python3
"""Create a simple test dataset for TRL training."""

from datasets import load_dataset

# Load a small sample dataset
print("Loading sample dataset...")
ds = load_dataset('HuggingFaceTB/smoltalk', 'everyday-conversations', split='train[:100]')

# Convert to the format expected by our training script
def convert_format(example):
    messages = example.get('messages', [])
    if len(messages) >= 2:
        return {
            'source_text': messages[0]['content'] if messages else '',
            'generated_text': messages[1]['content'] if len(messages) > 1 else '',
            'source_problem': messages[0]['content'] if messages else ''
        }
    return {
        'source_text': '',
        'generated_text': '',
        'source_problem': ''
    }

print("Converting format...")
ds_converted = ds.map(convert_format, remove_columns=ds.column_names)

# Save to JSONL
output_path = '/tmp/test_sft_data.jsonl'
ds_converted.to_json(output_path)
print(f"Created test dataset at {output_path}")
print(f"Dataset size: {len(ds_converted)} examples")
