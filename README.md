# Practice 03: Sign-In Travel Assistant with RAG Chat (Lumen Travel)

A grounded travel assistant web application built with **Streamlit**, **Firebase Authentication**, and **Retrieval-Augmented Generation (RAG)** via an **Ollama + Pinggy** server (`embeddinggemma` + `qwen3:4b`).

---

## 1. Student Information & Chosen City

- **Student Name:** *Hoàng Nguyễn Thế Hiển*
- **Student ID:** *25127326*
- **Course:** Computational Thinking
- **Chosen City:** **Ho Chi Minh City (TP. Hồ Chí Minh), Vietnam**
- **Travel Guide (`knowledge_base.json`):** Contains **16 short entries** (`G01`–`G16`) written in our own words about Ho Chi Minh City attractions and transport. Each entry includes `id`, `title`, `category`, `tags`, `text`, and `source` (prepared for **Assignment 4: RAG-Rec**).


## 2. Project Structure

```text
LAB_3_AUTH_RAG_CHAT_COMPUTATIONAL_THINKING/
├── app.py                 # Streamlit UI: Firebase Auth gate, session chat history, expandable sources
├── rag.py                 # RAG pipeline: embed once (embeddinggemma), cosine top-k, qwen3:4b generation
├── firebase_client.py     # Firebase Email/Password REST API client (adapted from Lab 03, no Firestore)
├── config.py              # Loads environment variables from .env safely
├── knowledge_base.json    # 16 Ho Chi Minh City travel guide entries (id, title, category, tags, text, source)
├── requirements.txt       # Python dependencies
├── .env.example           # Template for environment variables with empty values
├── .gitignore             # Excludes .env, .venv, and __pycache__ from Git
└── README.md              # Project documentation, test table, limitations, and video link
```


## 3. Installation & Running Instructions

### Step 1: Create and activate a virtual environment
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Start the Ollama + Pinggy Server (from Lab 4 / Lab 5)
1. Open the server notebook (`05_Ollama_Pinggy_Server.ipynb`) in Google Colab (select a **T4 GPU** runtime for fast inference).
2. Run the cells to start `ollama serve`, pull `embeddinggemma` and `qwen3:4b`, and open the Pinggy tunnel (`ssh -p 443 -R0:localhost:11434 -t qr@free.pinggy.io "u:Host:localhost:11434"`).
3. Copy the public `https://xxxx.run.pinggy-free.link` URL.

### Step 4: Configure `.env`
Copy `.env.example` to `.env` and fill in your Firebase Web App credentials and Pinggy URL:
```bash
cp .env.example .env
```
Edit `.env`:
```env
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_API_KEY=your-firebase-web-api-key
OLLAMA_BASE_URL=https://your-pinggy-tunnel.run.pinggy-free.link
```

### Step 5: Run the Streamlit app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, create an account or sign in, and ask questions about Ho Chi Minh City!


## 4. Test Table (6 Questions Evaluation)

Below is the evaluation of **6 test questions** (**3 answerable**, **1 paraphrase**, and **2 not in the guide**) using $k = 3$:

| # | Category | Question | Expected Entry ID | Retrieved IDs (Top-3 & Cosine Scores) | Answer Correct? | Actual Answer & Failure Analysis (*missing data, retrieval, or generation*) |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | **Answerable (1)** | *"How much is the student ticket for Independence Palace?"* | `G01` | `G01` (`0.6864`), `G11` (`0.4447`), `G02` (`0.3784`) | **Yes** | **Success:** `G01` ranked #1 (`0.6864` $\ge 0.50$). The assistant accurately answered `20,000 VND` for students with a valid student ID card and cited `[G01]`. |
| **Q2** | **Answerable (2)** | *"How much does it cost to take the elevator at 42 Nguyen Hue Cafe Apartment?"* | `G10` | `G10` (`0.7269`), `G16` (`0.4489`), `G01` (`0.4207`) | **Yes** | **Success:** *“The elevator costs 3,000 VND per ride at 42 Nguyen Hue Cafe Apartment [G10].”* (`G10` ranked #1 with `0.7269`). |
| **Q3** | **Answerable (3)** | *"Which markets in the guide sell street food in the evening, and what are their night hours?"* | `G03`, `G09` | `G09` (`0.5397`), `G03` (`0.5164`), `G12` (`0.4259`) | **Yes** | **Success:** *“Ho Thi Ky Market offers street food until 22:30 [G09], and Ben Thanh Market's night market runs until 23:00 [G03].”* With $k=3$, both relevant markets (`G09` and `G03`) were retrieved and cited. |
| **Q4** | **Paraphrase (1)** | *"Where can I find a quiet pedestrian lane without motorbikes to read books and drink iced coffee?"* | `G06` | `G06` (`0.5304`), `G10` (`0.2793`), `G16` (`0.2507`) | **Yes** | **Success:** *“Nguyen Van Binh Book Street in District 1 is a quiet pedestrian-only lane without motorbikes where you can read books and drink iced coffee, as per [G06].”* Semantic search matched `G06` (`0.5304` $\ge 0.50`) despite no keyword overlap on the street name. |
| **Q5** | **Not in Guide (1)** | *"How much does it cost to park a motorbike overnight at Independence Palace?"* | None (`G01` covers palace tickets, not parking) | `G01` (`0.5496`), `G11` (`0.3752`), `G02` (`0.3392`) | **Yes** (Says *"I don't know"*) | **Missing Data (Handled Correctly):** `G01` was retrieved with `0.5496` due to *"Independence Palace"*, but contains no parking fee data. The strict system prompt prevented guessing and replied *"I don't know based on the travel guide."* |
| **Q6** | **Not in Guide (2) / Failure Case** | *"What is the price of a VIP express pass to skip the queue at Banh Mi Huynh Hoa, and do they deliver to District 7?"* | None (`G16` covers sandwich price & queues, not VIP pass or District 7 delivery) | `G16` (`0.6861`), `G09` (`0.4661`), `G06` (`0.4413`) | **Yes** (Says *"I don't know"*) | **Missing Data:** *“I don't know based on the travel guide. The evidence [G16] mentions Banh Mi Huynh Hoa has separate queues for delivery drivers but does not specify VIP express pass prices or delivery coverage to District 7.”* *(Note: During initial testing with `num_predict=250`, **Generation Failure** occurred because `qwen3:4b` exhausted all 250 tokens on internal `<think>` reasoning before outputting the final answer; increasing `num_predict=1024` and stripping text before `</think>` resolved this generation failure).* |

> **Note on Evaluation Environment:**
> All retrieved IDs, Cosine similarity scores, and answers in the table above were verified on the **Streamlit web application (`streamlit run app.py`)** as well as the CLI test suite (`python rag.py`), which share the exact same `build_index()`, `retrieve()`, and `answer_question()` pipeline (`temperature=0`, `think=False`).


## 5. Computational Thinking Analysis (4 Steps)

1. **Abstraction:**
   - **Inputs:** User credentials (`email`, `password`), the static guide (`knowledge_base.json`), and a user `question`.
   - **Outputs:** Session access gate, a grounded `answer` citing `[Gxx]` (or *"I don't know"*), and an expandable panel listing the `top-k` retrieved entries and their Cosine similarity scores.
   - **Guide Entry Fields Kept:** `id` (unique citation key), `title` and `text` (core factual passage for embedding and generation), `source` (attribution), plus `category` and `tags` (kept to support content-based recommendation in **Assignment 4: RAG-Rec**). Raw GPS coordinates and user reviews were omitted as unnecessary for text QA.
2. **Decomposition:**
   - `Sign-in` (`email, password` $\rightarrow$ `st.session_state["user"]`) $\rightarrow$ `Embed` (`entries, question` $\rightarrow$ cached `doc_vectors (16, 768)`, `query_vector (768,)`) $\rightarrow$ `Retrieve` (`query_vector, doc_vectors, k=3` $\rightarrow$ `top-k hits with score`) $\rightarrow$ `Build Context` (`hits` $\rightarrow$ formatted `[ID] Title: Text` string) $\rightarrow$ `Generate` (`question, context, SYSTEM` $\rightarrow$ grounded `answer`) $\rightarrow$ `Display` (`answer, hits` $\rightarrow$ Streamlit chat UI & `st.expander`).
3. **Pattern Recognition:**
   - **Reused from Lab 3:** Firebase REST Authentication (`create_account`, `sign_in`, `FirebaseError`) and session cleanup (`forget_user()`).
   - **Reused from Lab 4 / Lab 5:** Single-batch indexing (`build_index` cached via `@st.cache_resource`), vectorized Cosine similarity search (`retrieve`), context formatting (`build_context`), and zero-temperature grounded chat (`generate`).
   - **Where `Filter → Score → Top-k` appears:** In this assignment, we **filter** valid guide entries (and apply a minimum similarity threshold), **score** each entry via Cosine similarity against the embedded question, and select the **top-$k$** entries. In **Assignment 4 (RAG-Rec / Lab 06)**, the exact same `Filter → Score → Top-k` pipeline reappears in `POIDataset.filter(category)` $\rightarrow$ scoring items via **Jaccard similarity** on `category + tags` (or **Popularity** $\text{rating} \times \log(1 + \text{num\_reviews})$) $\rightarrow$ sorting and selecting `.head(top_k)` recommendations.
4. **Algorithm Design:**
   - Selected **$k = 3$** so multi-entry questions (e.g. comparing night markets in `G03` and `G09`) retrieve all relevant passages without flooding the prompt with unrelated entries.
   - Designed a **two-layer "I don't know" rule**: (1) if `hits[0]["score"] < 0.5` (`MIN_SIMILARITY_SCORE = 0.5`), immediately return *"I don't know based on the travel guide."* without calling the LLM; (2) when an entity matches above the `0.5` threshold (`G01` score `0.5496` for Independence Palace motorbike parking) but the specific fact is absent, the strict `SYSTEM` prompt at `temperature=0` forces `qwen3:4b` to reply *"I don't know based on the travel guide"* rather than guessing.


## 6. Known Limitations & Note on AI Tools Used

### Known Limitations
- **Remote Tunnel Latency & Expiration:** Because `embeddinggemma` and `qwen3:4b` run on Google Colab over a free Pinggy SSH tunnel, the URL expires periodically and inference speed depends heavily on whether Colab is attached to a T4 GPU or CPU runtime.
- **Fixed Top-$k$ Context Window:** With $k = 3$, broad questions that require aggregating across more than 3 entries (e.g., *"List all 5 free attractions in the guide"*) will only retrieve the top 3 matching entries.
- **Ephemeral Chat History:** By design (`st.session_state` only, no Firestore), refreshing the browser tab or signing out clears all previous messages.

### Note on AI Tools Used
- **AI Tools:** Google Antigravity (AI coding assistant) was used to assist with writing codes (with many syntaxes inherited from previous lab files), refining Streamlit UI components, and debugging `qwen3:4b`'s `<think>` token truncation behavior. All code logic (`app.py`, `rag.py`, `firebase_client.py`), test analyses, and guide entries (`knowledge_base.json`) were verified by myself.


## 7. Demonstration Video

- **Demonstration Video Link:** https://youtu.be/QAgOAe8yVN0

