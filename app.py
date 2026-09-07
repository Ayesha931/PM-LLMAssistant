import os
import faiss
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from PyPDF2 import PdfReader
import json
import requests
import csv
from datetime import datetime
import streamlit as st
import re


# --- Settings ---
DOC_FOLDER = "PM-Docs"
CHUNK_SIZE = 200  # chars
INDEX_PATH = "vectorstore/faiss_index"
META_PATH = "vectorstore/metadata.pkl"
JSON_TEMPLATE = {
    "project_name": "",
    "company_goals": [],
    "recent_issues": [],
    "team": {},
    "prior_successes": []
}
OUTPUT_FOLDER = "generated_contexts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# # -- Save the JSON context, one for viewing and one for LLM usage
# def save_context_json(context_data, project_name):
#     # File paths
#     #pretty_file_path = os.path.join(OUTPUT_FOLDER, f"{project_name}_context_pretty.json")
#     #compact_file_path = os.path.join(OUTPUT_FOLDER, f"{project_name}_context_compact.json")

#     # Pretty print for debugging
#     with open(pretty_file_path, 'w', encoding='utf-8') as f:
#         json.dump(context_data, f, indent=4, ensure_ascii=False)
#     print(f"✅ Pretty JSON saved at: {pretty_file_path}")

#     # Compact version for LLMs
#     with open(compact_file_path, 'w', encoding='utf-8') as f:
#         json.dump(context_data, f, separators=(',', ':'), ensure_ascii=False)
#     print(f"✅ Compact JSON saved at: {compact_file_path}")

#     # Returning compact JSON as a string for LLM usage
#     return json.dumps(context_data, separators=(',', ':'), ensure_ascii=False)


# --- Load Model ---
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(texts):
    # encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # with torch.no_grad():
    #     model_output = model(**encoded_input)
    # return mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    
    # Use the correct device
    device = torch.device("cpu")  # or "cuda" if you have a GPU

    # Move model to the correct device
    model.to(device)

    # Prepare the inputs
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)

    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Return embeddings as numpy array
    return mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()


# --- Load & Chunk Text ---
def read_all_docs():
    chunks = []
    metadata = []
    for filename in os.listdir(DOC_FOLDER):
        if filename.endswith((".txt", ".md", ".json")):
            with open(os.path.join(DOC_FOLDER, filename), 'r', encoding='utf-8') as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(DOC_FOLDER, filename))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            continue

        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i+CHUNK_SIZE].strip()
            if chunk and not all(word in ENGLISH_STOP_WORDS for word in chunk.split()):
                chunks.append(chunk)
                metadata.append({"filename": filename, "chunk_id": i})
    return chunks, metadata

# --- Build and Save FAISS Index ---
def build_index():
    chunks, metadata = read_all_docs()
    embeddings = embed_text(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump((chunks, metadata), f)
    print("✅ FAISS index built and saved.")

# --- RAG Context Builder ---
def generate_context(query, num_results=5):
    # Load the FAISS index and metadata
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        chunks, metadata = pickle.load(f)

    # Embed the query
    query_embedding = embed_text([query])
    
    # Search for similar chunks
    distances, indices = index.search(query_embedding, num_results)
    relevant_chunks = [chunks[idx] for idx in indices[0] if idx != -1]

    return relevant_chunks

# --- Generate Full JSON Context ---
def build_json_context(project_name):
    json_context = JSON_TEMPLATE.copy()

    # ✅ Set the project name correctly
    json_context["project_name"] = project_name.strip()

    # Define keyword groups for each JSON section
    search_prompts = {
        # "project_name": ["project title", "project name", "current project", "app name", "product name"],
        "company_goals": ["company goals", "business objectives", "strategic goals", "growth targets", "mission"],
        "recent_issues": ["recent issues", "current challenges", "problems faced", "roadblocks", "barriers"],
        "team": ["project team", "team members", "contributors", "project resources", "people involved"],
        "prior_successes": ["prior successes", "previous wins", "past achievements", "milestones", "notable outcomes"]
    }

    # Build each section
    for section, keywords in search_prompts.items():
        section_text = []
        for keyword in keywords:
            relevant_chunks = generate_context(keyword, num_results=5)
            section_text.extend(relevant_chunks)

        # Add the retrieved chunks to the JSON
        if section in ["company_goals", "recent_issues", "prior_successes"]:
            json_context[section] = section_text
        # elif section == "project_name":
        #     json_context[section] = section_text[0] if section_text else "Unknown Project"
        elif section == "team":
            # Basic team extraction (more advanced parsing can be added later)
            json_context[section] = {"members": len(section_text)}

    return json_context

# --- Load project context file ---
def load_context(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def input_prompt_validator(decision_query):
    #validation # 1
    #is the prompt really realated to Project Managers role
    url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-r1-distill-llama-70b",
        "messages": [{"role": "system", "content": "Your role is to assess whether the user question is allowed or not. The allowed topics are everyday Project Management Decisions. If the topic is allowed, say 'allowed' otherwise say 'not_allowed'. Only answer in one word, no fluff."},
                     {"role": "user", "content": decision_query}],
        "temperature": 0.5
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        # print(f"Validator: {res.json()["choices"][0]["message"]["content"]}")
        result = res.json()["choices"][0]["message"]["content"]
        cleaned_result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        print(cleaned_result)
        return cleaned_result
    else:
        print(f"Validator Error: {res.status_code} - {res.text}")
        return f"Validator Error: {res.status_code} - {res.text}"
    #validation 2:
    #is the context data relevant to the decision_query asked

# --- Build prompt for the model ---
def build_prompt(context, decision_query):
    goals = "\n- ".join(context["company_goals"])
    issues = "\n- ".join(context["recent_issues"])
    wins = "\n- ".join(context["prior_successes"])

    return f"""
You are an expert strategic decision advisor for project managers.
Here is the project context:

Project: {context["project_name"]}

Company Goals:
- {goals}

Recent Issues:
- {issues}

Prior Successes:
- {wins}

The PM is facing this decision:
"{decision_query}"

Use the following structure to explain your reasoning:

1. **Situation Overview**: Summarize the strategic decision and context.
2. **Strategic Options**: List the available options or paths.
3. **Impact Assessment**: Analyze short-term and long-term impact of each option.
4. **Lessons from Past**: Reflect on past experiences or outcomes that relate.
5. **Recommended Action**: Suggest the most strategic option.
6. **Why This is the Best Option**: Justify your recommendation with alignment to project/business goals.
7. **Confidence Score (1–10)**: Indicate how confident you are and explain any assumptions or uncertainties.
"""

# Please suggest a strategic recommendation, and explain your reasoning step-by-step to help the PM trust your guidance.
# """

# --- Call Grok Cloud DeepSeek model ---
def call_grok_model(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-r1-distill-llama-70b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {res.status_code} - {res.text}"

# --- Log results to a CSV file ---
def log_decision(decision_query, prompt, result, think_content, project_name, log_file="decision_logs_main2.csv"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [now, project_name, decision_query, prompt, result, think_content]

    file_exists = os.path.exists(log_file)
    with open(log_file, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "project", "decision_query", "prompt", "model_output", "raw_thought_process"])
        writer.writerow(row)

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="AI Strategic Decision Partner", layout="centered")
    st.title("📊 Strategic Decision-Making AI Partner")

    st.markdown("Use this tool to get AI-generated strategy suggestions with reasoning based on your project context.")

    # Select context file
    # context_files = {
    #     "HealthLingo (App Redesign)": "project_context.json",
    #     "CloudFlow (SaaS Scaling)": "project_context_saas.json",
    #     "SmartPrep (EdTech Features)": "project_context_edtech.json"
    # }

    # selected_project = st.selectbox("Choose Project Context", list(context_files.keys()))
    # context_path = context_files[selected_project]
    # context = load_context(context_path)

    project_name = st.text_area("Name of Project:")
    build_index()
    # context_json = build_json_context(project_name)
    context = build_json_context(project_name)

    #compact_json = save_context_json(context_json, "HealthLingo")
    # print("\n🔍 Compact JSON for LLM:\n")
    # print(compact_json)

    
    # context_json_str = save_context_json(context_json, project_name)
    # context = json.loads(context_json)

    

    #context = load_context("generated_contexts\HealthLingo_context_compact.json")

    # st.subheader("📁 Project Context")
    # st.json(context)

    st.subheader("🎯 Your Strategic Decision")
    decision = st.text_area("Describe the strategic decision you're facing:", height=100)

    if st.button("Get AI Recommendation") and decision:
        validation = input_prompt_validator(decision)
        if validation == 'not_allowed':
            st.markdown("### !! OFF-TOPIC QUERY")
            st.markdown("I can only assist with routine PM decisions, please ask relevant query.")
            print("Topical guardrail triggered")

        else:
            prompt = build_prompt(context, decision)
            with st.spinner("Thinking..."):
                result = call_grok_model(prompt)

            # Extract <think> content and clean output for display
            think_match = re.search(r"<think>(.*?)</think>", result, re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else ""
            cleaned_result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

            st.markdown("### 🤖 AI Recommendation and Reasoning")
            st.markdown(cleaned_result)

            log_decision(decision, prompt, cleaned_result, think_content, context["project_name"])


if __name__ == "__main__":
    main()

    # print("\n🔍 Generated Context JSON:\n")
    # print(context_json)
