import gradio as gr
import random
import re
import threading
import time

import spaces
import torch
import numpy as np

# Assuming the transformers library is installed
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- Global Settings ---
# These variables are placed in the global scope and will be loaded once when the Gradio app starts
system_prompt = []
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "Embformer-MiniMind-Base (0.1B)": ["HighCWu/Embformer-MiniMind-Base-0.1B", "Embformer-MiniMind-Base-0.1B"],
    "Embformer-MiniMind-Seqlen512 (0.1B)": ["HighCWu/Embformer-MiniMind-Seqlen512-0.1B", "Embformer-MiniMind-Seqlen512-0.1B"],
    "Embformer-MiniMind (0.1B)": ["HighCWu/Embformer-MiniMind-0.1B", "Embformer-MiniMind-0.1B"],
    "Embformer-MiniMind-RLHF (0.1B)": ["HighCWu/Embformer-MiniMind-RLHF-0.1B", "Embformer-MiniMind-RLHF-0.1B"],
    "Embformer-MiniMind-R1 (0.1B)": ["HighCWu/Embformer-MiniMind-R1-0.1B", "Embformer-MiniMind-R1-0.1B"],
}

# --- Helper Functions (Mostly unchanged) ---

def process_assistant_content(content, model_source, selected_model_name):
    """
    Processes the model output, converting <think> tags to HTML details elements,
    and handling content after </think>, filtering out <answer> tags.
    """
    is_r1_model = False
    if model_source == "API":
        if 'R1' in selected_model_name:
            is_r1_model = True
    else:
        model_identifier = MODEL_PATHS.get(selected_model_name, ["", ""])[1]
        if 'R1' in model_identifier:
            is_r1_model = True
    
    if not is_r1_model:
        return content

    # Fully closed <think>...</think> block
    if '<think>' in content and '</think>' in content:
        # Using re.split is more robust than finding indices
        parts = re.split(r'(</think>)', content, 1)
        think_part = parts[0] + parts[1] # All content from <think> to </think>
        after_think_part = parts[2] if len(parts) > 2 else ""

        # 1. Process the think part
        processed_think = re.sub(
            r'(<think>)(.*?)(</think>)',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">Reasoning (Click to expand)</summary>\2</details>',
            think_part,
            flags=re.DOTALL
        )
        
        # 2. Process the part after </think>, filtering <answer> tags
        # Using re.sub to replace <answer> and </answer> with an empty string
        processed_after_think = re.sub(r'</?answer>', '', after_think_part)
        
        # 3. Concatenate the results
        return processed_think + processed_after_think

    # Only an opening <think>, indicating reasoning is in progress
    if '<think>' in content and '</think>' not in content:
        return re.sub(
            r'<think>(.*?)$',
            r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">Reasoning...</summary>\1</details>',
            content,
            flags=re.DOTALL
        )

    # This case should be rare in streaming output, but kept for completeness
    if '<think>' not in content and '</think>' in content:
        # Also need to process content after </think>
        parts = re.split(r'(</think>)', content, 1)
        think_part = parts[0] + parts[1]
        after_think_part = parts[2] if len(parts) > 2 else ""

        processed_think = re.sub(
            r'(.*?)</think>',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">Reasoning (Click to expand)</summary>\1</details>',
            think_part,
            flags=re.DOTALL
        )
        processed_after_think = re.sub(r'</?answer>', '', after_think_part)
        
        return processed_think + processed_after_think

    # If there are no <think> tags, return the content directly
    return content


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Gradio App Logic ---

# Gradio uses global variables or functions to load models, similar to st.cache_resource
# We cache models and tokenizers in a dictionary to avoid reloading
loaded_models = {}

def load_model_tokenizer_gradio(model_name):
    """
    Gradio version of the model loading function with caching.
    """
    if model_name in loaded_models:
        # print(f"Using cached model: {model_name}")
        return loaded_models[model_name]
    
    # print(f"Loading model: {model_name}...")
    model_path = MODEL_PATHS[model_name][0]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=".cache",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=".cache",
    )
    loaded_models[model_name] = (model, tokenizer)
    print("Model loaded.")
    return model, tokenizer

@spaces.GPU
def chat_fn(
    user_message, 
    history, 
    model_source,
    # Local model settings
    selected_model,
    # API settings
    api_url,
    api_model_id,
    api_model_name,
    api_key,
    # Generation parameters
    history_chat_num,
    max_new_tokens,
    temperature
):
    """
    Gradio's core chat processing function.
    It receives the current values of all UI components as input.
    """
    history = history or []
    
    # Build context for the model based on the passed, unmodified history
    chat_messages_for_model = []
    # Limit the number of history turns
    if history_chat_num > 0 and len(history) > history_chat_num:
        relevant_history_turns = history[-history_chat_num:]
    else:
        relevant_history_turns = history
        
    for user_msg, assistant_msg in relevant_history_turns:
        chat_messages_for_model.append({"role": "user", "content": user_msg})
        if assistant_msg:
            chat_messages_for_model.append({"role": "assistant", "content": assistant_msg})
    
    # Add the current user message to the model's context
    chat_messages_for_model.append({"role": "user", "content": user_message})
    
    final_chat_messages = system_prompt + chat_messages_for_model
    
    # Now, update the history for UI display
    history.extend([*chat_messages_for_model, {"role": "assistant", "content": user_message}])

    # --- Model Invocation ---
    if model_source == "API":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_url)
            
            response = client.chat.completions.create(
                model=api_model_id,
                messages=final_chat_messages,
                stream=True,
                temperature=temperature
            )
            
            answer = ""
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                answer += content
                processed_answer = process_assistant_content(answer, model_source, api_model_name)
                history[-1]["content"] = processed_answer
                yield history, history
        
        except Exception as e:
            history[-1]["content"] = f"API call error: {str(e)}"
            yield history, history

    else: # Local Model
        try:
            model, tokenizer = load_model_tokenizer_gradio(selected_model)
            
            random_seed = random.randint(0, 2**32 - 1)
            setup_seed(random_seed)

            new_prompt = tokenizer.apply_chat_template(
                final_chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": 1,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": temperature,
                "top_p": 0.85,
                "streamer": streamer,
            }

            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            answer = ""
            for new_text in streamer:
                answer += new_text
                processed_answer = process_assistant_content(answer, model_source, selected_model)
                history[-1]["content"] = processed_answer
                yield history, history
        except Exception as e:
            history[-1]["content"] = f"Local model call error: {str(e)}"
            yield history, history

# --- Gradio UI Layout ---
css = """
.gradio-container { font-family: 'sans-serif'; }
footer { display: none !important; }
"""
image_url = "https://chunte-hfba.static.hf.space/images/modern%20Huggies/Huggy%20Sunny%20hello.png"

# Define example data
prompt_datas = [
    '请介绍一下自己。',
    '你更擅长哪一个学科？',
    '鲁迅的《狂人日记》是如何批判封建礼教的？',
    '我咳嗽已经持续了两周，需要去医院检查吗？',
    '详细的介绍光速的物理概念。',
    '推荐一些杭州的特色美食吧。',
    '请为我讲解“大语言模型”这个概念。',
    '如何理解ChatGPT？',
    'Introduce the history of the United States, please.'
]

with gr.Blocks(theme='soft', css=css) as demo:
    # History state, this is the Gradio equivalent of st.session_state
    chat_history = gr.State([])
    chat_input_cache = gr.State("")

    # Top Title and Badge
    title_html = """
<div style="text-align: center;">
    <h1>Embformer: An Embedding-Weight-Only Transformer Architecture</h1>
    <div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 10px;">
        <a href="https://doi.org/10.5281/zenodo.15736957">
            <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15736957-blue.svg" alt="DOI">
        </a>
        <a href="https://github.com/HighCWu/embformer">
            <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="code">
        </a>
        <a href="https://huggingface.co/collections/HighCWu/embformer-minimind-685be74dc761610439241bd5">
            <img src="https://img.shields.io/badge/Model-🤗-yellow" alt="model">
        </a>
    </div>
</div>
"""
    gr.HTML(title_html)
    gr.Markdown("""
This is the official demo of [Embformer: An Embedding-Weight-Only Transformer Architecture](https://doi.org/10.5281/zenodo.15736957).

**Note**: Since the model dataset used in this demo is derived from the MiniMind dataset, which contains a large proportion of Chinese content, please try to use Chinese as much as possible in the conversation.
""")

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### Model Settings")
            
            # Model source switcher
            model_source_radio = gr.Radio(["Local Model", "API"], value="Local Model", label="Select Model Source", visible=False)
            
            # Local model settings
            with gr.Group(visible=True) as local_model_group:
                selected_model_dd = gr.Dropdown(
                    list(MODEL_PATHS.keys()), 
                    value="Embformer-MiniMind (0.1B)", 
                    label="Select Local Model"
                )

            # API settings
            with gr.Group(visible=False) as api_model_group:
                api_url_tb = gr.Textbox("http://127.0.0.1:8000/v1", label="API URL")
                api_model_id_tb = gr.Textbox("embformer-minimind", label="Model ID")
                api_model_name_tb = gr.Textbox("Embformer-MiniMind (0.1B)", label="Model Name (for feature detection)")
                api_key_tb = gr.Textbox("none", label="API Key", type="password")

            # Common generation parameters
            history_chat_num_slider = gr.Slider(0, 6, value=0, step=2, label="History Turns")
            max_new_tokens_slider = gr.Slider(256, 8192, value=1024, step=1, label="Max New Tokens")
            temperature_slider = gr.Slider(0.6, 1.2, value=0.85, step=0.01, label="Temperature")

            # Clear history button
            clear_btn = gr.Button("🗑️ Clear History")

        with gr.Column(scale=4):
            gr.Markdown("### Chat")
            
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                avatar_images=(None, image_url),
                type="messages",
                height=350
            )
            chat_input = gr.Textbox(
                show_label=False,
                placeholder="Send a message to MiniMind...  (Enter to send)",
                container=False,
                scale=7,
                elem_id="chat-textbox",
            )
            examples = gr.Examples(
                examples=prompt_datas,
                inputs=chat_input, # After clicking, the example content will fill chat_input
                label="Click an example to ask (will automatically clear chat and continue)"
            )

    # --- Event Listeners and Bindings ---
    
    # Show/hide corresponding setting groups when switching model source
    def toggle_model_source_ui(source):
        return {
            local_model_group: gr.update(visible=source == "Local Model"),
            api_model_group: gr.update(visible=source == "API")
        }
    model_source_radio.change(
        fn=toggle_model_source_ui,
        inputs=model_source_radio,
        outputs=[local_model_group, api_model_group]
    )

    # Define the list of input components for the submit event
    submit_inputs = [
        chat_input_cache, chat_history, model_source_radio, selected_model_dd,
        api_url_tb, api_model_id_tb, api_model_name_tb, api_key_tb,
        history_chat_num_slider, max_new_tokens_slider, temperature_slider
    ]

    # When chat_input is submitted (user presses enter or an example is clicked), run chat_fn
    submit_event = chat_input.submit(
        fn=lambda text: ("", text),
        inputs=chat_input,
        outputs=[chat_input, chat_input_cache],
    ).then(
        fn=chat_fn,
        inputs=submit_inputs,
        outputs=[chatbot, chat_history],
    )
    
    # Event chain for clicking an example
    examples.load_input_event.then(
        fn=lambda text: ("", text, [], []), # A function to clear the history
        inputs=chat_input,
        outputs=[chat_input, chat_input_cache, chatbot, chat_history], # This affects the chatbot and chat_history
    ).then(
        fn=chat_fn, # Use the dedicated run_example function
        inputs=submit_inputs, # Pass example text and other settings
        outputs=[chatbot, chat_history],
    )

    # Clear history button logic
    def clear_history():
        return [], []
    clear_btn.click(fn=clear_history, outputs=[chatbot, chat_history])
    chatbot.clear(fn=clear_history, outputs=[chatbot, chat_history])


if __name__ == "__main__":
    # Pre-load the default model on startup
    print("Pre-loading default model...")
    load_model_tokenizer_gradio("Embformer-MiniMind (0.1B)")
    
    # Launch the Gradio app
    demo.queue().launch(share=False)
