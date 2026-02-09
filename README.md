# Semantic Search with Transformers

A machine learning project that demonstrates how to build an efficient semantic search engine using sentence-transformers and Facebook's Faiss library to search through a corpus of 41,000 machine learning research papers.

## Overview

This project implements semantic search capabilities that can retrieve research papers based on meaning and context, rather than just exact keyword matches. Using pre-trained Transformer models (MPNet), the search engine understands synonyms and similar contexts, making it far more powerful than traditional lexical search. The system leverages GPU acceleration for both embedding generation and similarity search using Faiss.

## Features

- **Semantic Embeddings**: Generate high-quality 768-dimensional embeddings using MPNet sentence-transformers model
- **GPU-Accelerated Indexing**: Create and manage search indices using Faiss GPU implementation for fast k-nearest-neighbors search
- **Context-Aware Search**: Retrieve research papers based on semantic similarity with L2 distance metrics
- **Data Preprocessing**: Encode and prepare datasets using scikit-learn LabelEncoder
- **Persistent Embeddings**: Save and load pre-computed embeddings using pickle for efficient reuse
- **Interactive Jupyter Notebook**: Explore and experiment with the search engine in an interactive environment

## Technologies

- **Python 3.9+** - Programming language
- **PyTorch** - Deep learning framework with CUDA 12.4 support
- **sentence-transformers** - For generating semantic embeddings (MPNet model)
- **Faiss GPU** - For efficient GPU-accelerated similarity search
- **Pandas** - For data manipulation and analysis
- **scikit-learn** - For data preprocessing and encoding
- **NumPy** - For numerical operations
- **Jupyter Notebook** - For interactive development and experimentation

## Prerequisites

- CUDA-compatible GPU (required for GPU-accelerated search)
- CUDA 12.4+
- Python 3.9
- Git LFS (for cloning data files)
- Access to SLURM-based GPU cluster (optional)

## Installation

### Local Setup

1. Install Git LFS if not already installed:

```bash
# On macOS
brew install git-lfs

# On Ubuntu/Debian
sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install
```

2. Clone the repository (Git LFS will automatically download large data files):

```bash
git clone https://github.com/sheygs/semantic-search.git
cd semantic-search
```

3. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install PyTorch with CUDA 12.4 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

5. Install other dependencies:

```bash
pip install -r requirements.txt
```

### GPU Cluster Setup (SLURM)

For running on a SLURM-based GPU cluster, use the automated setup script:

```bash
bash scripts/setup.sh
```

You can optionally pass a custom Jupyter port (default is 8888):

```bash
bash scripts/setup.sh 9999
```

The script will:

- Detect if you are on a login node and automatically request a GPU node via SLURM (1 GPU, 16 CPUs, 32GB RAM, 4-hour time limit)
- Load CUDA 12.4 module and export the necessary environment variables
- Create a Python 3.9 virtual environment (using `/usr/bin/python3`) and install all dependencies
- Verify the installation (PyTorch CUDA, Faiss GPU, SentenceTransformers)
- Start Jupyter Notebook and display the SSH tunnel command for your specific compute node

**Connecting with VSCode:** Once Jupyter is running on the GPU node, open your notebook in VSCode, click the kernel selector, choose **"Select Another Kernel"** > **"Existing Jupyter Server"**, and paste the URL printed by the script (e.g., `http://gpu8:8888/?token=...`).

## Project Structure

```text
semantic-search/
├── data/                           # Data directory (tracked with Git LFS)
│   ├── research_papers.json        # Input dataset (41,000 papers) - LFS
│   └── new_embeddings.pickle       # Pre-computed embeddings - LFS
├── scripts/
│   └── setup.sh                    # Automated GPU cluster setup script
├── src/
│   └── semantic_search.ipynb       # Main Jupyter notebook
├── .gitattributes                  # Git LFS configuration
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Usage

### Running the Notebook

#### On a SLURM GPU cluster

1. Run the setup script (starts Jupyter on a GPU node automatically):

```bash
bash scripts/setup.sh
```

2. Connect to the Jupyter server using the URL printed by the script.

3. Open `src/semantic_search.ipynb` and run cells sequentially.

#### Locally

1. Activate the virtual environment and start Jupyter:

```bash
source venv/bin/activate
jupyter notebook
```

2. Open `src/semantic_search.ipynb`

3. Run the cells sequentially to:
   - Load the research papers dataset
   - Generate embeddings (or load pre-computed ones)
   - Create Faiss GPU index
   - Perform semantic searches

## How It Works

1. **Data Loading**: The system loads 41,000 machine learning research papers from a JSON file containing titles, summaries, and metadata.

2. **Encoding**: The MPNet sentence-transformer model converts each paper's summary into a 768-dimensional dense vector embedding that captures semantic meaning.

3. **Indexing**: Faiss creates a GPU-accelerated index structure using L2 distance metrics for fast similarity search across all embeddings.

4. **Searching**: When you query the system, it:
   - Converts your query to an embedding using the same model
   - Finds the k-nearest neighbors in the GPU index using L2 distance
   - Returns the most semantically similar papers ranked by distance

5. **Retrieval**: Results include L2 distances, paper IDs, titles, and summaries of the top-k most similar papers.

## Model Information

- **Model**: `all-mpnet-base-v2`
- **Embedding Dimension**: 768
- **Max Sequence Length**: 128 tokens
- **Pooling Strategy**: Mean tokens
- **Note**: This model is deprecated. For production use, check [Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers/models) for updated models.

## Performance

- **Dataset Size**: 41,000 research papers
- **Embedding Generation**: ~43 seconds on GPU (29.72 it/s)
- **Index Size**: 41,000 embeddings
- **Search Speed**: Near real-time with GPU acceleration
- **GPU Memory**: Optimized for single GPU with CUDA 12.4

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size when encoding or use CPU version (`faiss-cpu`)
2. **Module not found**: Ensure all dependencies are installed in the active virtual environment
3. **Deprecated model warning**: Consider using newer models from [sentence-transformers](https://huggingface.co/sentence-transformers/models)
4. **NumPy version conflicts**: Install `numpy<2.0` to avoid compatibility issues

### GPU Cluster Connection

If using the setup script, follow the SSH tunnel command printed at the end. Generally:

1. On your local machine, SSH to the cluster with port forwarding to the compute node:

```bash
ssh -L 8888:<compute-node>:8888 username@cluster.domain
```

2. Open `http://localhost:8888` in your browser and paste the token provided by Jupyter.

**VSCode users:** Instead of a browser, use **"Select Another Kernel"** > **"Existing Jupyter Server"** in the notebook kernel selector and paste the URL.

## Future Improvements

- Implement additional search strategies (cosine similarity, inner product)
- Add support for multi-modal search (text + metadata filtering)
- Add batch query processing
- Implement approximate nearest neighbor search for larger datasets
- Create a web interface for easier interaction

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- [sentence-transformers](https://www.sbert.net/) for the excellent embedding library
- [Facebook AI Research](https://github.com/facebookresearch/faiss) for the Faiss library
- [Hugging Face](https://huggingface.co/) for model hosting and transformers ecosystem
- The machine learning research community for the dataset

## References

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- [MPNet](https://arxiv.org/abs/2004.09297)
