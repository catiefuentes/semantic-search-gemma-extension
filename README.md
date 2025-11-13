# Semantic Search Engine Demo

An educational Jupyter notebook demonstrating semantic search using sentence embeddings. This project shows how to use pre-trained transformer models to find semantically similar documents based on meaning rather than keyword matching.

## 🎯 What is Semantic Search?

Semantic search understands the *meaning* of text, not just keywords. Unlike traditional keyword-based search, semantic search can find relevant documents even when they don't share exact words with the query.

**Example:**
- Query: "Tell me about recent phone technology releases"
- Matches: Documents about "iPhone releases" or "mobile device announcements" even without the word "phone"

## 📚 Learning Objectives

By working through this notebook, you will learn:

- How to use pre-trained embedding models from Hugging Face
- How to convert text into numerical vectors (embeddings)
- How to calculate semantic similarity using cosine similarity
- How to implement a basic semantic search system

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd SemanticSearchEngine
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Open and run the notebook:**
   - Open `semantic_search_demo.ipynb`
   - Run all cells sequentially (or use "Run All" from the Cell menu)

### First Run

The first time you run the notebook, it will download the `all-MiniLM-L6-v2` model (~80MB) from Hugging Face. This is a one-time download and will be cached for future use.

## 📦 Dependencies

- **sentence-transformers**: For loading pre-trained embedding models
- **numpy**: For numerical operations
- **scikit-learn**: For cosine similarity calculations

See `requirements.txt` for specific versions.

## 📖 How It Works

1. **Load Model**: Loads a pre-trained sentence transformer model
2. **Create Corpus**: Defines a collection of documents to search
3. **Generate Embeddings**: Converts each document into a numerical vector
4. **Query Processing**: Converts the search query into an embedding
5. **Similarity Calculation**: Computes cosine similarity between query and all documents
6. **Rank Results**: Returns documents ranked by semantic similarity

## 🔧 Usage

The notebook is designed to be run cell-by-cell. Each cell includes:

- **Markdown explanations**: Educational content explaining concepts
- **Code cells**: Executable Python code
- **Output examples**: Expected results

### Customizing the Demo

You can easily customize the notebook:

- **Change the corpus**: Modify the `documents` list with your own text
- **Try different queries**: Experiment with various search queries
- **Explore scores**: Use the optional cell to see similarity scores for all documents

## 🧪 Example Output

```
==================================================
Query: **Tell me about the recent phone technology releases.**
==================================================
Best Match (Score: 0.7234):
'The newest iPhone model was released with a powerful new chip.'
```

## 📝 Project Structure

```
SemanticSearchEngine/
├── semantic_search_demo.ipynb  # Main educational notebook
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## 🎓 Educational Use

This project is designed for educational purposes and can be used to teach:

- Natural Language Processing (NLP) concepts
- Vector embeddings and semantic similarity
- Information retrieval systems
- Machine learning applications

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Add more examples
- Improve documentation
- Suggest enhancements
- Report issues

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) by UKP Lab
- [Hugging Face](https://huggingface.co/) for hosting pre-trained models
- The `all-MiniLM-L6-v2` model creators

## 📚 Additional Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)

## ❓ FAQ

**Q: Do I need a GPU?**  
A: No, this demo works on CPU. GPU will speed up processing for larger datasets.

**Q: Can I use a different model?**  
A: Yes! Replace `'all-MiniLM-L6-v2'` with any sentence transformer model from Hugging Face.

**Q: How do I scale this to larger datasets?**  
A: For production use, consider using vector databases like Pinecone, Weaviate, or FAISS for efficient similarity search.

---

**Happy Learning! 🚀**

