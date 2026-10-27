# Movie Expert Chatbot using RAG

A conversational movie-expert chatbot named **Bob**, built with a small instruction-tuned language model and **Retrieval-Augmented Generation (RAG)**. Instead of relying only on the model's internal knowledge, Bob retrieves a relevant movie plot and includes it in the prompt before generating an answer.

## Project Overview

This project combines:

- **Qwen2.5-0.5B-Instruct** — a compact instruction-tuned language model for generating answers.
- **Sentence Transformers (`all-MiniLM-L6-v2`)** — converts movie descriptions and questions into embeddings for semantic similarity search.
- **Pandas** — loads the movie dataset.
- **PyTorch** — supports tensor operations, similarity scoring, and model inference.
- **Movie plot dataset** — provides the movie titles, release years, and plots used as the chatbot's reference material.

Bob is instructed to answer in 3–4 sentences and to use only the supplied movie information. If the retrieved information does not contain an answer, the prompt tells Bob to say that he does not know.

## How It Works

1. **Load the language model:** The notebook downloads the Qwen2.5-0.5B-Instruct model and its tokenizer from Hugging Face.
2. **Load movie data:** The notebook reads the `Release Year`, `Title`, and `Plot` columns from `wiki_movie_plots_deduped.csv`.
3. **Create embeddings:** Sentence Transformers encodes the movie records into vector representations.
4. **Retrieve relevant information:**
   - First, the code looks for an exact movie-title match in the user's question. If multiple titles match, it chooses the longest match.
   - If no title matches, it embeds the question and retrieves the movie record with the highest cosine similarity.
5. **Build the prompt:** The retrieved movie information and user's question are passed to Qwen along with the system instructions and recent conversation history.
6. **Generate an answer:** The model produces a response, which is added to the conversation history for subsequent turns.

## Requirements

- Python 3.10 or later is recommended.
- Jupyter Notebook or JupyterLab.
- Internet access for the initial download of the model and embedding model.
- Sufficient RAM and disk space for the dataset and downloaded models.
- A compatible GPU is optional; inference can run on a supported GPU or CPU, depending on the installed PyTorch environment.

## Setup

### 1. Clone the repository

### 2. Install dependencies

```bash
pip install torch transformers pandas sentence-transformers jupyter
```

Install the PyTorch build appropriate for your system from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) if necessary.

### 3. Add the movie dataset

Place `wiki_movie_plots_deduped.csv` from https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots in the notebook's working directory. The notebook expects this exact filename and these columns:

- `Release Year`
- `Title`
- `Plot`

### 4. Run the notebook

Start Jupyter:

```bash
jupyter notebook
```

Open `movie_expert.ipynb` and run the cells from top to bottom. On the first run, the required Hugging Face models will be downloaded. The notebook also encodes the movie records, which may take some time depending on your hardware and dataset size.

## Example Usage

After running the setup cells and creating the chatbot, ask Bob questions:

```python
bob = BobTheChatbot(model, tokenizer)

bob.chatWithRAG(
    "Can you tell me about the movie The Dark Knight?"
)

bob.chatWithRAG(
    "Can you tell me something about the movie Swades?"
)

bob.chatWithRAG(
    "Who played the lead role in the movie Swades?"
)
```

The notebook also includes an example question about *Handphone*.

**Note:** Answers depend on the retrieved record. The dataset contains movie plots, which may not include cast details or other facts. For questions like “Who played the lead role?”, Bob may not be able to answer reliably unless that information is present in the retrieved text.

## Project Structure

```text
.
├── movie_expert.ipynb
├── wiki_movie_plots_deduped.csv   # Add this dataset locally
└── README.md
```

The CSV file is a runtime dependency and is not included automatically by the notebook.

## Current Implementation Notes

- **Title matching:** Normalizes text by converting it to lowercase and removing punctuation before looking for movie titles in the question.
- **Semantic fallback:** Uses cosine similarity between the question embedding and stored movie embeddings when no title match is found.
- **Context length:** Truncates the retrieved movie record to 2,000 characters before adding it to the prompt.
- **Conversation memory:** Retains up to three recent question-and-answer turns. The retrieved plot is not stored in the conversation history.
- **Generation:** Uses deterministic decoding (`do_sample=False`) with a repetition penalty.

## Limitations

- The semantic fallback selects the single most similar movie record; it does not retrieve multiple documents or use a dedicated vector database.
- Semantic similarity does not guarantee that the retrieved movie is the correct one, particularly for vague questions or similarly named movies.
- The model can still generate incorrect information. Instructions to use only the supplied context reduce this risk but do not eliminate it.
- The dataset's movie plots may not contain cast lists, ratings, genres, or other metadata.
- Movie embeddings are computed when the chatbot object is initialized, so startup time and memory use depend on dataset size.
- The code uses a local CSV file and does not provide a web interface or API.

## Possible Improvements

- Retrieve and rank multiple candidate movie records instead of only one.
- Add movie metadata such as cast, director, genre, and ratings from a reliable source.
- Use a vector index such as FAISS for more scalable retrieval.
- Add a confidence or relevance threshold so the chatbot can decline to answer when retrieval is weak.
- Build a web interface with Streamlit or Gradio.
- Add tests for title matching, retrieval quality, and context-grounded answers.

## Technologies

`Python` · `PyTorch` · `Hugging Face Transformers` · `Sentence Transformers` · `Pandas` · `Jupyter Notebook` · `Retrieval-Augmented Generation (RAG)`
