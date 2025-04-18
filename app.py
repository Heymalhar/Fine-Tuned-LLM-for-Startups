import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
import threading

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant that provides support to startup founders."}
    ]

# Title
st.title("🚀 Startup Support Chatbot")

# User input
user_input = st.chat_input("Ask a question about your startup...")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_dir = "./qwen-startup-finetuned-lora"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    return model, tokenizer

model, tokenizer = load_model()

# Display chat history
for msg in st.session_state.messages[1:]:  # exclude system prompt from display
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# On new user input
if user_input:
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant message container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        # Prepare messages for prompt
        prompt = tokenizer.apply_chat_template(
            st.session_state.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generation parameters
        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            streamer=streamer
        )

        # Run generation in thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream output
        generated_text = ""
        for token in streamer:
            generated_text += token
            response_placeholder.markdown(generated_text)

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": generated_text})
