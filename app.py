import json
import numpy as np
import gradio as gr
from typing import Dict, List, Any
from dataclasses import dataclass
import os

# CSS styling for the visualization
STYLE = """
.custom-container {
	display: grid;
	align-items: center;
    margin: 0!important;
    overflow-y: hidden;
}
.prose ul ul {
    font-size: 10px!important;
}
.prose li {
    margin-bottom: 0!important;
}
.prose table {
    margin-bottom: 0!important;
}
.prose td, th {
    padding-left: 2px;
    padding-right: 2px;
    padding-top: 0;
    padding-bottom: 0;
    text-wrap:nowrap;
}
.tree {
	padding: 0px;
	margin: 0!important;
	box-sizing: border-box;
    font-size: 10px;
	width: 100%;
	height: auto;
	text-align: center;
    display:inline-block;
    padding-bottom: 10px!important;
}
#root {
    display: inline-grid!important;
    width:auto!important;
    min-width: 220px;
}
.tree ul {
    padding-left: 20px;
    position: relative;
    transition: all 0.5s ease 0s;
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin: 0px !important;
}
.tree li {
    display: flex;
    text-align: center;
    list-style-type: none;
    position: relative;
    padding-left: 20px;
    transition: all 0.5s ease 0s;
    flex-direction: row;
    justify-content: start;
    align-items: center;
}
.tree li::before, .tree li::after {
    content: "";
    position: absolute;
    left: 0px;
    border-left: 1px solid var(--body-text-color);
    width: 20px;
}
.tree li::before {
    top: 0;
    height:50%;
}
.tree li::after {
    top: 50%;
    height: 55%;
    bottom: auto;
    border-top: 1px solid var(--body-text-color);
}
.tree li:only-child::after, li:only-child::before {
    display: none;
}
.tree li:first-child::before, .tree li:last-child::after {
    border: 0 none;
}
.tree li:last-child::before {
	border-bottom: 1px solid var(--body-text-color);
	border-radius: 0px 0px 0px 5px;
	-webkit-border-radius: 0px 0px 0px 5px;
	-moz-border-radius: 0px 0px 0px 5px;
}
.tree li:first-child::after {
	border-radius: 5px 0 0 0;
	-webkit-border-radius: 5px 0 0 0;
	-moz-border-radius: 5px 0 0 0;
}
.tree ul ul::before {
    content: "";
    position: absolute;
    left: 0;
    top: 50%;
    border-top: 1px solid var(--body-text-color);
    width: 20px;
    height: 0;
}
.tree ul:has(> li:only-child)::before {
    width:40px;
}
.child:before {
    border-right: 2px solid var(--body-text-color);
    border-bottom: 2px solid var(--body-text-color);
    content: "";
    position: absolute;
    width: 10px;
    left: 8px;
    height: 10px;
    top: 50%;
    margin-top: -5px;
    transform: rotate(315deg);
}
.tree li a {
	border: 1px solid var(--body-text-color);
	padding: 5px;
	border-radius: 5px;
	text-decoration-line: none;
	border-radius: 5px;
	transition: .5s;
    display: flex;
    align-items: center;
    justify-content: space-between;
    overflow: hidden;
}
.tree li a span {
	padding: 5px;
	font-size: 12px;
	letter-spacing: 1px;
	font-weight: 500;
}
/*Hover-Section*/
.tree li a:hover, .tree li a:hover+ul li a {
	background: var(--primary-500);
}
.tree li a:hover+ul li::after, .tree li a:hover+ul li::before, .tree li a:hover+ul::before, .tree li a:hover+ul ul::before, .tree li a:hover+ul a::before {
	border-color: var(--primary-500);
}
.chosen-token {
    background-color: var(--primary-400);
}
.chosen-token td, .chosen-token tr {
    color: black!important;
}
.end-of-text {
    width:auto!important;
}
.nonfinal {
    width:280px;
    min-width: 280px;
}
.selected-sequence {
    background-color: var(--secondary-500);
}
.nonselected-sequence {
    background-color: var(--primary-500);
}
.nopadding {
    padding-left: 0;
}
"""

@dataclass
class BeamNode:
    """Represents a node in the beam search tree."""
    current_token_ix: int
    cumulative_score: float
    children_score_divider: float
    table: str
    current_sequence: str
    children: Dict[int, "BeamNode"]
    total_score: float
    is_final: bool
    is_selected_sequence: bool

class BeamSearchData:
    """Container for beam search data loaded from file."""
    def __init__(self, data: Dict[str, Any]):
        self.input_text = data.get("input_text", "")
        # Convert scores to numpy arrays to avoid list/array issues
        raw_scores = data.get("scores", [])
        self.scores = []
        for step_scores in raw_scores:
            step_arrays = []
            for beam_scores in step_scores:
                step_arrays.append(np.array(beam_scores) if not isinstance(beam_scores, np.ndarray) else beam_scores)
            self.scores.append(step_arrays)
        
        self.sequences = data.get("sequences", [])  # Final sequences (token IDs)
        self.sequences_scores = data.get("sequences_scores", [])  # Scores for final sequences
        self.decoded_sequences = data.get("decoded_sequences", [])  # Decoded text sequences
        
        # Convert vocab keys to integers for proper token lookup
        raw_vocab = data.get("vocab", {})
        self.vocab = {}
        for k, v in raw_vocab.items():
            try:
                self.vocab[int(k)] = v
            except (ValueError, TypeError):
                self.vocab[k] = v
                
        self.eos_token_id = data.get("eos_token_id", 50256)  # End of sequence token ID
        self.num_beams = data.get("num_beams", 1)
        self.length_penalty = data.get("length_penalty", 1.0)

def clean(s: str) -> str:
    """Clean string for display by escaping newlines and tabs."""
    return s.replace("\n", r"\n").replace("\t", r"\t").strip()

def decode_token(token_id: int, vocab: Dict[int, str]) -> str:
    """Decode a token ID to its string representation."""
    return vocab.get(token_id, f"<UNK_{token_id}>")

def generate_markdown_table(
    scores: np.ndarray, 
    previous_cumul_score: float, 
    score_divider: float, 
    vocab: Dict[int, str],
    top_k: int = 4, 
    chosen_tokens: List[str] = None
) -> str:
    """Generate an HTML table showing the top-k tokens and their scores."""
    markdown_table = """
    <table>
        <tr>
            <th><b>Token</b></th>
            <th><b>Step score</b></th>
            <th><b>Total score</b></th>
        </tr>"""
    
    # Convert to numpy array if it's a list
    scores_array = np.array(scores) if not isinstance(scores, np.ndarray) else scores
    
    for token_idx in np.array(np.argsort(scores_array)[-top_k:])[::-1]:
        token = decode_token(token_idx, vocab)
        item_class = ""
        if chosen_tokens and token in chosen_tokens:
            item_class = "chosen-token"
        markdown_table += f"""
        <tr class={item_class}>
            <td>{clean(token)}</td>
            <td>{scores_array[token_idx]:.4f}</td>
            <td>{(scores_array[token_idx] + previous_cumul_score)/score_divider:.4f}</td>
        </tr>"""
    markdown_table += """
    </table>"""
    return markdown_table

def generate_nodes(node: BeamNode, step: int, vocab: Dict[int, str]) -> str:
    """Recursively generate HTML for the tree nodes."""
    token = decode_token(node.current_token_ix, vocab)

    if node.is_final:
        if node.is_selected_sequence:
            selected_class = "selected-sequence"
        else:
            selected_class = "nonselected-sequence"
        return f"<li> <a class='end-of-text child {selected_class}'> <span> <b>{clean(token)}</b> <br>Total score: {node.total_score:.2f}</span> </a> </li>"

    html_content = (
        f"<li> <a class='nonfinal child'> <span> <b>{clean(token)}</b> </span>"
    )
    if node.table is not None:
        html_content += node.table
    html_content += "</a>"

    if len(node.children.keys()) > 0:
        html_content += "<ul> "
        for token_ix, subnode in node.children.items():
            html_content += generate_nodes(subnode, step=step + 1, vocab=vocab)
        html_content += "</ul>"
    html_content += "</li>"

    return html_content

def generate_html(start_sentence: str, original_tree: BeamNode, vocab: Dict[int, str]) -> str:
    """Generate the complete HTML visualization."""
    html_output = f"""<div class="custom-container">
				<div class="tree">
                <ul> <li> <a id='root' class="nopadding"> <span> <b>{start_sentence}</b> </span> {original_tree.table} </a>"""
    html_output += "<ul> "
    for subnode in original_tree.children.values():
        html_output += generate_nodes(subnode, step=1, vocab=vocab)
    html_output += "</ul>"
    html_output += """
        </li> </ul>
        </div>
    </body>
    """
    return html_output

def generate_beams(beam_data: BeamSearchData) -> BeamNode:
    """Generate the beam search tree from loaded data."""
    original_tree = BeamNode(
        cumulative_score=0,
        current_token_ix=None,
        table=None,
        current_sequence=beam_data.input_text,
        children={},
        children_score_divider=(1 ** beam_data.length_penalty),
        total_score=None,
        is_final=False,
        is_selected_sequence=False,
    )
    
    n_beams = beam_data.num_beams
    scores = beam_data.scores
    length_penalty = beam_data.length_penalty
    decoded_sequences = beam_data.decoded_sequences
    vocab = beam_data.vocab
    eos_token_id = beam_data.eos_token_id
    
    beam_trees = [original_tree] * n_beams
    generation_length = len(scores)

    for step, step_scores in enumerate(scores):
        # Gather all possible descendants for each beam
        (
            top_token_indexes,
            top_cumulative_scores,
            beam_indexes,
            current_sequence,
            top_tokens,
            token_scores,
        ) = ([], [], [], [], [], [])

        score_idx = 0
        for beam_ix in range(len(beam_trees)):
            current_beam = beam_trees[beam_ix]

            # Skip if the beam is already final
            if current_beam.is_final:
                continue

            # Get top cumulative scores for the current beam
            current_top_token_indexes = list(
                np.array(step_scores[score_idx].argsort()[-n_beams:])[::-1]
            )
            top_token_indexes += current_top_token_indexes
            token_scores += list(np.array(step_scores[score_idx][current_top_token_indexes]))
            top_cumulative_scores += list(
                np.array(step_scores[score_idx][current_top_token_indexes])
                + current_beam.cumulative_score
            )
            beam_indexes += [beam_ix] * n_beams
            current_sequence += [beam_trees[beam_ix].current_sequence] * n_beams
            top_tokens += [decode_token(el, vocab) for el in current_top_token_indexes]
            score_idx += 1

        # Create dataframe for processing
        import pandas as pd
        top_df = pd.DataFrame.from_dict(
            {
                "token_index": top_token_indexes,
                "cumulative_score": top_cumulative_scores,
                "beam_index": beam_indexes,
                "current_sequence": current_sequence,
                "token": top_tokens,
                "token_score": token_scores,
            }
        )
        
        maxes = top_df.groupby(["token_index", "current_sequence"])[
            "cumulative_score"
        ].idxmax()
        top_df = top_df.loc[maxes]

        # Sort all top probabilities and keep top n_beams * 2
        top_df_selected = top_df.sort_values("cumulative_score", ascending=False).iloc[
            :n_beams * 2
        ]
        
        beams_to_keep = 0
        unfinished_beams = 0
        for _, row in top_df_selected.iterrows():
            beams_to_keep += 1
            current_token_choice_ix = row["token_index"]
            is_final = step == len(scores) - 1 or current_token_choice_ix == eos_token_id
            if not is_final:
                unfinished_beams += 1
            if unfinished_beams >= n_beams:
                break
            if step == generation_length - 1 and beams_to_keep == n_beams:
                break
        top_df_selected_filtered = top_df_selected.iloc[:beams_to_keep]

        # Write the scores table in each beam tree
        score_idx = 0
        for beam_ix in range(len(beam_trees)):
            current_beam = beam_trees[beam_ix]
            if current_beam.table is None:
                selected_tokens = top_df_selected_filtered.loc[
                    top_df_selected_filtered["current_sequence"] == current_beam.current_sequence
                ]
                markdown_table = generate_markdown_table(
                    step_scores[score_idx],
                    current_beam.cumulative_score,
                    current_beam.children_score_divider,
                    vocab,
                    chosen_tokens=list(selected_tokens["token"].values),
                )
                beam_trees[beam_ix].table = markdown_table
            if not current_beam.is_final:
                score_idx = min(score_idx + 1, n_beams - 1)

        # Add new children to each beam
        cumulative_scores = [beam.cumulative_score for beam in beam_trees]
        for _, row in top_df_selected_filtered.iterrows():
            # Update the source tree
            source_beam_ix = int(row["beam_index"])
            current_token_choice_ix = row["token_index"]
            current_token_choice = decode_token(current_token_choice_ix, vocab)
            token_scores = row["token_score"]

            cumulative_score = cumulative_scores[source_beam_ix] + np.asarray(token_scores)
            current_sequence = (
                beam_trees[source_beam_ix].current_sequence + current_token_choice
            )
            is_final = step == len(scores) - 1 or current_token_choice_ix == eos_token_id
            beam_trees[source_beam_ix].children[current_token_choice_ix] = BeamNode(
                current_token_ix=current_token_choice_ix,
                table=None,
                children={},
                current_sequence=current_sequence,
                cumulative_score=cumulative_score,
                total_score=cumulative_score / (step + 1 ** length_penalty),
                children_score_divider=((step + 2) ** length_penalty),
                is_final=is_final,
                is_selected_sequence=(
                    current_sequence.replace("<|endoftext|>", "")
                    in [el.replace("<|endoftext|>", "") for el in decoded_sequences]
                ),
            )

        # Swap all beams by descending cumul score
        beam_trees = [
            beam_trees[int(top_df_selected_filtered.iloc[beam_ix]["beam_index"])]
            for beam_ix in range(beams_to_keep)
        ]

        # Advance all beams by one token
        for beam_ix in range(beams_to_keep):
            current_token_choice_ix = top_df_selected_filtered.iloc[beam_ix]["token_index"]
            beam_trees[beam_ix] = beam_trees[beam_ix].children[current_token_choice_ix]

    return original_tree

def load_beam_search_data(file_path: str) -> BeamSearchData:
    """Load beam search data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return BeamSearchData(data)

def get_available_files() -> List[str]:
    """Get list of available beam search data files."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    return files

def visualize_beam_search(selected_file: str) -> tuple[str, str]:
    """Generate HTML visualization from selected beam search data file."""
    try:
        file_path = os.path.join("data", selected_file)
        beam_data = load_beam_search_data(file_path)
        
        # Generate description markdown
        markdown = "The conclusive sequences are the ones that end in an `<|endoftext|>` token or at the end of generation."
        markdown += "\n\nThey are ranked by their scores, as given by the formula `score = cumulative_score / (output_length ** length_penalty)`.\n\n"
        markdown += f"Number of beams: {beam_data.num_beams}, Length penalty: {beam_data.length_penalty}"
        markdown += "\n\nSelected sequences are highlighted in **<span style='color:var(--secondary-500)!important'>blue</span>**."
        markdown += " Non-selected sequences are highlighted in **<span style='color:var(--primary-500)!important'>yellow</span>**."
        markdown += "\n#### <span style='color:var(--secondary-500)!important'>Output sequences:</span>"
        
        # Add sequences to markdown
        if beam_data.sequences_scores:
            for i, (sequence, score) in enumerate(zip(beam_data.decoded_sequences, beam_data.sequences_scores)):
                markdown += f"\n- Score `{score:.2f}`: `{clean(sequence)}`"
        else:
            for i, sequence in enumerate(beam_data.decoded_sequences):
                markdown += f"\n- `{clean(sequence)}`"
        
        # Generate beam tree
        original_tree = generate_beams(beam_data)
        
        # Generate HTML
        html = generate_html(beam_data.input_text, original_tree, beam_data.vocab)
        
        return html, markdown
        
    except Exception as e:
        error_msg = f"Error loading file: {str(e)}"
        return f"<p style='color: red;'>{error_msg}</p>", error_msg

# Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.yellow,
        secondary_hue=gr.themes.colors.blue,
    ),
    css=STYLE,
) as demo:
    gr.Markdown(
        """# <span style='color:var(--primary-500)!important'>Beam Search Visualizer</span>

This tool visualizes beam search decoding results loaded from pre-generated data files.

Upload your beam search data files to the `data/` directory in JSON format and select one to visualize.

#### <span style='color:var(--primary-500)!important'>How to use:</span>
1. Place your beam search data files (`.json` format) in the `data/` directory
2. Select a file from the dropdown below
3. Click "Visualize" to generate the beam search tree

#### <span style='color:var(--primary-500)!important'>Data format:</span>
Your JSON files should contain the following fields:
- `input_text`: The input prompt
- `scores`: List of score matrices for each generation step
- `sequences`: Final sequences as token IDs
- `sequences_scores`: Scores for final sequences
- `decoded_sequences`: Final sequences as text
- `vocab`: Mapping from token IDs to token strings
- `eos_token_id`: End of sequence token ID
- `num_beams`: Number of beams used
- `length_penalty`: Length penalty applied
"""
    )
    
    with gr.Row():
        file_dropdown = gr.Dropdown(
            label="Select beam search data file",
            choices=get_available_files(),
            value=None,
            interactive=True,
        )
        refresh_btn = gr.Button("Refresh Files", scale=0)
    
    visualize_btn = gr.Button("Visualize Beam Search", variant="primary")
    
    with gr.Row():
        output_html = gr.HTML(label="Beam Search Tree")
    
    output_markdown = gr.Markdown(label="Sequences Information")
    
    # Event handlers
    refresh_btn.click(
        fn=lambda: gr.Dropdown(choices=get_available_files()),
        outputs=file_dropdown
    )
    
    visualize_btn.click(
        fn=visualize_beam_search,
        inputs=[file_dropdown],
        outputs=[output_html, output_markdown]
    )

if __name__ == "__main__":
    demo.launch() 