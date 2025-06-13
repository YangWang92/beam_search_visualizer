# Beam Search Visualizer

This is a visualization tool for beam search decoding results. Unlike traditional beam search visualizers that perform inference, this tool reads pre-generated beam search data from JSON files and creates interactive HTML visualizations.

Clone from `https://huggingface.co/spaces/m-ric/beam_search_visualizer/tree/main`, thanks for Aymeric Roucher's contribution.

## Features

- **File-based visualization**: Load beam search results from JSON files
- **Interactive tree display**: Visualize the beam search tree with expandable nodes
- **Score tables**: View token scores and probabilities at each step
- **Sequence highlighting**: Selected sequences are highlighted in blue, non-selected in yellow
- **No inference required**: Works with pre-generated data, no model loading needed

## Installation

1. Install required dependencies:
```bash
pip install gradio numpy pandas
```

2. Run the application:
```bash
python app.py
```

## Usage

1. **Prepare your data**: Place your beam search data files in the `data/` directory as JSON files
2. **Select a file**: Use the dropdown to select a data file
3. **Visualize**: Click "Visualize Beam Search" to generate the tree visualization

## Data Format

Your JSON files should contain the following fields:

### Required Fields

- **`input_text`** (string): The input prompt used for generation
- **`scores`** (array): List of score matrices for each generation step
  - Shape: `[num_steps][num_active_beams][vocab_size]`
  - Each entry contains the logits/scores for tokens at that step
- **`sequences`** (array): Final generated sequences as token IDs
  - Shape: `[num_return_sequences][sequence_length]`
- **`decoded_sequences`** (array): Final sequences as decoded text strings
- **`vocab`** (object): Mapping from token IDs (as strings) to token text
- **`num_beams`** (integer): Number of beams used during search
- **`eos_token_id`** (integer): Token ID for end-of-sequence token

### Optional Fields

- **`sequences_scores`** (array): Final scores for each returned sequence
- **`length_penalty`** (float): Length penalty applied (default: 1.0)

### Example Data Structure

```json
{
  "input_text": "The quick brown fox",
  "num_beams": 3,
  "length_penalty": 1.0,
  "eos_token_id": 50256,
  "scores": [
    [
      [0.1, 0.2, 0.3, ...],  // Scores for beam 0 at step 0
      [0.2, 0.3, 0.4, ...],  // Scores for beam 1 at step 0
      [0.3, 0.4, 0.5, ...]   // Scores for beam 2 at step 0
    ],
    // ... more steps
  ],
  "sequences": [
    [1, 9, 9, 50256],     // Token IDs for sequence 1
    [1, 9, 8, 50256],     // Token IDs for sequence 2
    [1, 8, 9, 50256]      // Token IDs for sequence 3
  ],
  "sequences_scores": [2.85, 2.75, 2.65],
  "decoded_sequences": [
    "The quick brown fox jumps over<|endoftext|>",
    "The quick brown fox jumps high<|endoftext|>",
    "The quick brown fox runs fast<|endoftext|>"
  ],
  "vocab": {
    "1": " quick",
    "8": " runs",
    "9": " jumps",
    "50256": "<|endoftext|>",
    // ... more token mappings  
  }
}
```

## Generating Data Files

To generate data files from your own models, you can use the following approach:

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model and tokenizer
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-tokenizer")

# Generate with beam search
input_text = "Your input prompt"
inputs = tokenizer([input_text], return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    num_beams=4,
    num_return_sequences=3,
    return_dict_in_generate=True,
    output_scores=True,
    do_sample=False,
)

# Extract and convert data
data = {
    "input_text": input_text,
    "num_beams": 4,
    "length_penalty": 1.0,
    "eos_token_id": tokenizer.eos_token_id,
    "scores": [score.cpu().numpy().tolist() for score in outputs.scores],
    "sequences": outputs.sequences.cpu().numpy().tolist(),
    "sequences_scores": outputs.sequences_scores.cpu().numpy().tolist() if hasattr(outputs, 'sequences_scores') else [],
    "decoded_sequences": tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False),
    "vocab": {str(i): tokenizer.decode([i]) for i in range(tokenizer.vocab_size)}
}

# Save to file
with open("data/my_beam_search.json", "w") as f:
    json.dump(data, f, indent=2)
```

## Visualization Features

### Tree Structure
- **Root node**: Shows the input text and initial score distribution
- **Internal nodes**: Show tokens with their score tables
- **Leaf nodes**: Show final tokens with total scores
- **Color coding**: 
  - Blue: Selected sequences (returned by beam search)
  - Yellow: Non-selected sequences (pruned during search)

### Score Tables
Each node displays a table showing:
- **Token**: The actual token text
- **Step score**: The score for this token at this step
- **Total score**: Cumulative score up to this point
- **Highlighting**: Chosen tokens are highlighted in the table

## Tips

1. **Large vocabularies**: The tool works best with smaller vocabularies. For large models, consider filtering the vocabulary to most common tokens.

2. **Performance**: Very deep trees or wide beams may slow down visualization. Consider limiting the number of steps or beams for better performance.

3. **File size**: Large score matrices can create large JSON files. Consider using compressed formats or pruning less relevant scores.

## Troubleshooting

### Common Issues

1. **"No files found"**: Make sure your JSON files are in the `data/` directory
2. **Parsing errors**: Validate your JSON format using a JSON validator
3. **Missing fields**: Ensure all required fields are present in your data file
4. **Token decoding errors**: Check that your vocab mapping covers all token IDs in your sequences

### File Validation

The tool will show error messages if your data file is missing required fields or has formatting issues. Check the console output for detailed error information. 