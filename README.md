# Multi-Domain RAG

A Retrieval-Augmented Generation (RAG) system that enables natural language querying of document collections. The system automatically detects document domain (academic or financial) and provides contextually appropriate responses using local language models.

## Prerequisites

### Dependencies

1. **Install Ollama**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows: Download from https://ollama.ai/download
   ```

2. **Download Language Model**
   ```bash
   # Recommended
   ollama pull llama3.1:8b
   
   # Alternatives
   ollama pull mistral:7b
   ollama pull llama3.1:8b-instruct-q4_0  # Smaller memory footprint
   ```

3. **Install Python Packages**
   ```bash
   pip install langchain ollama pypdf chromadb sentence-transformers
   ```

## Installation

1. Clone the repository

2. Create document directories:
   ```bash
   mkdir -p school finance general
   ```

3. Add PDF documents to appropriate directories:
   - `school/`: Academic materials (textbooks, notes, assignments)
   - `finance/`: Financial documents (statements, receipts, bills)
   - `general/`: Other documents

## Usage

### Running the Application
```bash
python rag.py
```

### Available Commands
- `stats`: Display document statistics
- `quit`, `exit`, `bye`, `/bye`: Exit the application

### Query Examples

**Academic Queries**:
- "What is the main concept in chapter 5?"
- "Explain photosynthesis from my biology notes"
- "Summarize the assignment requirements"

**Financial Queries**:
- "How much did I spend last month?"
- "What are my largest spending categories?"
- "Show restaurant expenses from March"

### Sample Output
```
Question: How much did I spend on groceries last month?
Detected domain: finance
Searching through your documents...

Answer: Based on your bank statements, you spent $347.82 on groceries 
last month, including transactions at Walmart ($156.23), Safeway 
($98.45), and other grocery stores ($93.14).

Sources used:
  Finance Documents:
    1. march_statement.pdf (Page 2)
    2. march_statement.pdf (Page 4)
```

## Configuration

### Model Selection
Modify the model parameter in the main function:
```python
chatbot = SmartDocumentChatBot(
    base_directory="./",
    model_name="llama3.1:8b"  # Change to preferred model
)
```

### Document Processing Parameters
Adjust text chunking in the `setup_components` method:
```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Text chunk size
    chunk_overlap=200,  # Overlap between chunks
)
```

## Troubleshooting

### Common Issues

**No documents found**
- Verify PDF files exist in `school/`, `finance/`, or `general/` directories
- Ensure files have `.pdf` extension

**Ollama connection errors**
- Start Ollama service: `ollama serve`
- Verify model installation: `ollama list`
- Test model directly: `ollama run llama3.1:8b`

**Memory issues**
- Use smaller model: `mistral:7b` or quantized versions
- Reduce chunk size and retrieval count
- Monitor system resources

**Slow performance**
- Switch to faster model: `mistral:7b`
- Reduce retrieved document chunks (modify `k` parameter)
- Ensure sufficient system resources

## Technical Architecture

The system implements a multi-stage RAG pipeline:

1. **Document Ingestion**: PDF parsing and text extraction
2. **Text Chunking**: Recursive text splitting with overlap
3. **Embedding Generation**: Sentence transformer encoding
4. **Vector Storage**: ChromaDB persistent storage
5. **Query Processing**: Domain detection and retrieval
6. **Response Generation**: LLM-based answer synthesis

## Supported File Types

Currently supported:
- PDF documents (.pdf)
