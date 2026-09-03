# PM-LLMAssistant 🚀  
![Python](https://img.shields.io/badge/Python-3.10-blue)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.24-red)  
![FAISS](https://img.shields.io/badge/FAISS-1.7.3-green)  
![Deepseek R1](https://img.shields.io/badge/Deepseek%20R1-CoT%20Reasoning-orange)  
![Status](https://img.shields.io/badge/Status-Active-brightgreen)  
![License](https://img.shields.io/badge/License-None-lightgrey)  

---

An intelligent project management assistant designed to **save project managers' time**, **enhance decision-making quality**, and **build trust in AI** for strategic planning. PM-LLMAssistant leverages **Retrieval-Augmented Generation (RAG)** and **Chain-of-Thought (CoT)** reasoning to provide context-rich, reasoned responses based on company documents.  

🔗 **[Demo Video](https://drive.google.com/file/d/1Yk8uPtd0279bkissONHwN5cmqU4g12SH/view?usp=sharing)** | 📂 **[Repository](https://https://github.com/Ayesha931/PM-LLMAssistant)**  

---

## 🌟 Key Features  

- 🗂 **Efficient Context Retrieval**: Automates context retrieval using RAG to save PMs from manually searching for documents.  
- 🧠 **Strategic Decision Support**: Provides context-aware, reasoned responses using CoT via Deepseek R1 distilled LLaMA.  
- 📁 **Multi-Project Support**: Handles multiple projects, each with unique contexts.  
- 🛠️ **Customizable Prompt Templates**: Trigger reasoned responses tailored to specific project management scenarios.  
- 🛡️ **Off-Topic Guardrail**: Prevents responses to out-of-scope queries, keeping the assistant focused on project management tasks.

---

## 📦 Project Structure  

PM-LLMAssistant/

├── 📁 PM-Docs/                 # Project-specific context files (PDF, DOCX, TXT)

├── 📁 faiss_index/             # Serialized FAISS index for semantic retrieval

├── 📄 requirements.txt         # Python dependencies

├── 🖥️ app.py                   # Main Streamlit app

├── 📝 build_index_and_generate_context.py  # Embedding and context generation scripts

└── 📄 README.md                # Project documentation


---

## 🚀 How It Works  

### 1. 📥 Context Ingestion  
- Extracts project data from various file formats (PDF, DOCX, TXT) in the `PM-Docs` folder.  
- Embeds this data using a hosted `paraphrase-mpnet-base-v2` model for semantic understanding.  
- Builds a FAISS index for efficient retrieval.  

### 2. 🔎 Real-Time Context Retrieval  
- Uses RAG to automate the extraction of relevant context.  
- Handles multiple sections like company goals, recent issues, and prior successes.  

### 3. 🧠 Chain-of-Thought Reasoning  
- Triggers CoT reasoning through tailored prompts that encourage the model to consider intermediate reasoning steps.  
- Provides structured, explainable responses to complex project management queries.  

---

## 💡 Example Use Case  

Imagine you're managing a HealthLingo app redesign project. PM-LLMAssistant can:  
- Retrieve the project's strategic goals, recent challenges, and key metrics.  
- Provide insights into potential decisions, like:  

**Decision Scenario:**  
*Should we release our new accessibility feature quickly or delay it slightly to conduct thorough user testing?*  

**PM-LLMAssistant Response:**  
Based on the extracted context, the assistant might identify relevant company goals (e.g., increasing accessibility), recent user feedback, and previous successes, then provide a reasoned response that considers both immediate user impact and long-term brand trust.  

---

## 🔍 How RAG and CoT are Implemented  

### RAG (Retrieval-Augmented Generation)  
- Uses FAISS for fast similarity search against document embeddings.  
- Automates the retrieval of relevant context, ensuring that responses are well-informed.  

### CoT (Chain-of-Thought)  
- Deepseek's reasoning model (R1 distilled LLaMA) is prompted to generate intermediate reasoning steps.  
- Provides logically coherent outputs that mimic the decision-making patterns of human managers.  

---

## 🚧 Known Limitations  

1. **Scope Restriction**: Focused on routine project management tasks.  
2. **Limited Use Case**: Built around a single dummy use case due to time constraints.  
3. **Budget Constraints**: Uses only open-source models; lacks enterprise-grade LLMs.  

---

## 🎯 Future Improvements  

- **Broader Scope**: Expand beyond routine project management to include strategic, operational, and risk management decisions.  
- **Advanced Reasoning**: Fine-tune CoT prompts for more precise, domain-specific reasoning.  
- **Reduced AI Dependency**: Implement checks to prevent over-reliance on AI, maintaining human oversight in critical decisions.  
- **Enhanced Data Privacy**: Move towards local model deployment to address data security concerns.  

---

## 📚 Licensing and Attribution  

- **Open Source**: No specific license currently.  
- **Solo Project**: Currently a solo endeavor, but open to future collaboration.  

---

