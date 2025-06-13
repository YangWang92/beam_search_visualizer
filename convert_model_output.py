#!/usr/bin/env python3
"""
Helper script to convert model beam search outputs to the JSON format required by the visualizer.

Usage:
    python convert_model_output.py --input_text "Your prompt" --model_name "gpt2" --output_file "data/my_output.json"

This script will:
1. Load the specified model
2. Generate beam search outputs
3. Convert to the required JSON format
4. Save to the specified file
"""

import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, Any


def generate_and_convert(
    input_text: str,
    model_name: str,
    output_file: str,
    max_new_tokens: int = 5,
    num_beams: int = 4,
    num_return_sequences: int = 3,
    length_penalty: float = 1.0,
    do_sample: bool = False,
    vocab_size_limit: int = 1000,
) -> None:
    """
    Generate beam search outputs and convert to JSON format.
    
    Args:
        input_text: Input prompt for generation
        model_name: HuggingFace model name or path
        output_file: Output JSON file path
        max_new_tokens: Maximum number of tokens to generate
        num_beams: Number of beams for beam search
        num_return_sequences: Number of sequences to return
        length_penalty: Length penalty for beam search
        do_sample: Whether to use sampling
        vocab_size_limit: Limit vocabulary size for smaller files
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Generating beam search outputs...")
    print(f"Input: {input_text}")
    print(f"Beams: {num_beams}, Max tokens: {max_new_tokens}")
    
    # Tokenize input
    inputs = tokenizer([input_text], return_tensors="pt", padding=True)
    
    # Generate with beam search
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=min(num_return_sequences, num_beams),
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=length_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    print(f"Generated {len(outputs.sequences)} sequences")
    
    # Create vocabulary mapping (limited to reduce file size)
    print("Creating vocabulary mapping...")
    vocab_size = min(tokenizer.vocab_size, vocab_size_limit)
    vocab = {}
    
    # Add tokens that appear in the sequences
    unique_tokens = set()
    for seq in outputs.sequences:
        unique_tokens.update(seq.cpu().numpy().tolist())
    
    # Add tokens that appear in the scores (top tokens)
    for step_scores in outputs.scores:
        for beam_scores in step_scores:
            top_tokens = torch.topk(beam_scores, k=min(50, len(beam_scores))).indices
            unique_tokens.update(top_tokens.cpu().numpy().tolist())
    
    # Create vocab mapping for relevant tokens
    for token_id in unique_tokens:
        if token_id < tokenizer.vocab_size:
            try:
                vocab[str(token_id)] = tokenizer.decode([token_id])
            except:
                vocab[str(token_id)] = f"<UNK_{token_id}>"
    
    # Add common special tokens
    special_tokens = [
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.bos_token_id if tokenizer.bos_token_id else None,
        tokenizer.unk_token_id if tokenizer.unk_token_id else None,
    ]
    
    for token_id in special_tokens:
        if token_id is not None and str(token_id) not in vocab:
            try:
                vocab[str(token_id)] = tokenizer.decode([token_id])
            except:
                vocab[str(token_id)] = f"<SPECIAL_{token_id}>"
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Convert outputs to required format
    print("Converting to JSON format...")
    
    # Remove input tokens from sequences to get only generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_sequences = [seq[input_length:].cpu().numpy().tolist() for seq in outputs.sequences]
    
    # Decode sequences
    decoded_sequences = []
    for seq in outputs.sequences:
        decoded = tokenizer.decode(seq, skip_special_tokens=False)
        decoded_sequences.append(decoded)
    
    # Convert scores to list format
    scores_list = []
    for step_scores in outputs.scores:
        step_list = []
        for beam_scores in step_scores:
            step_list.append(beam_scores.cpu().numpy().tolist())
        scores_list.append(step_list)
    
    # Get sequence scores if available
    sequences_scores = []
    if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
        sequences_scores = outputs.sequences_scores.cpu().numpy().tolist()
    
    # Create the data structure
    data = {
        "input_text": input_text,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "scores": scores_list,
        "sequences": generated_sequences,
        "sequences_scores": sequences_scores,
        "decoded_sequences": decoded_sequences,
        "vocab": vocab,
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,
            "model_name": model_name,
        }
    }
    
    # Save to file
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved beam search data to {output_file}")
    print(f"File size: {len(json.dumps(data))} characters")
    
    # Print summary
    print("\nGeneration Summary:")
    print(f"Input: {input_text}")
    print(f"Generated {len(decoded_sequences)} sequences:")
    for i, (seq, score) in enumerate(zip(decoded_sequences, sequences_scores or [None]*len(decoded_sequences))):
        score_str = f" (score: {score:.3f})" if score is not None else ""
        print(f"  {i+1}: {seq}{score_str}")


def main():
    parser = argparse.ArgumentParser(description="Convert model beam search outputs to visualizer format")
    parser.add_argument("--input_text", type=str, required=True, help="Input text prompt")
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Maximum new tokens to generate")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams")
    parser.add_argument("--num_return_sequences", type=int, default=3, help="Number of sequences to return")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling")
    parser.add_argument("--vocab_size_limit", type=int, default=1000, help="Limit vocabulary size")
    
    args = parser.parse_args()
    
    generate_and_convert(
        input_text=args.input_text,
        model_name=args.model_name,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        length_penalty=args.length_penalty,
        do_sample=args.do_sample,
        vocab_size_limit=args.vocab_size_limit,
    )


if __name__ == "__main__":
    main() 