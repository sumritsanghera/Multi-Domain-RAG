import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import re
from datetime import datetime

class SmartDocumentChatBot:
    def __init__(self, base_directory="./", model_name="llama3.1:8b"):
        """
        Multi-domain RAG system for school and financial documents
        
        Directory structure:
        ./
        ├── school/          # Academic PDFs, notes, textbooks
        ├── finance/         # Bank statements, receipts, financial docs
        └── general/         # Any other documents
        """
        self.base_directory = base_directory
        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None
        self.current_mode = "general"  # general, school, finance
        
        # Create directory structure
        self.setup_directories()
        
        # Initialize components
        self.setup_components()
        self.load_and_process_documents()
        self.setup_qa_chain()
    
    def setup_directories(self):
        """Create organized directory structure"""
        directories = [
            os.path.join(self.base_directory, "school"),
            os.path.join(self.base_directory, "finance"), 
            os.path.join(self.base_directory, "general")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_components(self):
        """Set up the core components"""
        print("Setting up components...")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Ollama LLM
        self.llm = Ollama(model=self.model_name)
        
        # Text splitter with different strategies for different document types
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_financial_data(self, text, filename):
        """Extract structured financial data from text"""
        financial_patterns = {
            'amounts': r'\$[\d,]+\.?\d*',
            'dates': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'categories': r'(?i)(restaurant|gas|groceries|entertainment|other)'
        }
        
        extracted_data = []
        for category, pattern in financial_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                extracted_data.append(f"{category.title()}: {', '.join(matches[:10])}")  # Limit to 10 matches
        
        return extracted_data
    
    def load_and_process_documents(self):
        """Load and process documents from all directories"""
        print(f"Loading documents from {self.base_directory}...")
        
        all_documents = []
        
        # Process each subdirectory
        for subdir in ['school', 'finance', 'general']:
            subdir_path = os.path.join(self.base_directory, subdir)
            
            if not os.path.exists(subdir_path):
                continue
            
            pdf_files = [f for f in os.listdir(subdir_path) if f.endswith('.pdf')]
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(subdir_path, pdf_file)
                print(f"Processing: {subdir}/{pdf_file}")
                
                try:
                    # Load PDF
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    
                    # Add metadata to each document
                    for doc in documents:
                        doc.metadata.update({
                            'source_file': pdf_file,
                            'document_type': subdir,
                            'file_path': pdf_path
                        })
                        
                        # Special processing for financial documents
                        if subdir == 'finance':
                            financial_data = self.extract_financial_data(doc.page_content, pdf_file)
                            if financial_data:
                                # Create a summary document with extracted financial data
                                summary_content = f"Financial Summary for {pdf_file}:\n" + "\n".join(financial_data)
                                summary_doc = Document(
                                    page_content=summary_content,
                                    metadata={
                                        'source_file': pdf_file,
                                        'document_type': 'finance_summary',
                                        'file_path': pdf_path
                                    }
                                )
                                all_documents.append(summary_doc)
                    
                    all_documents.extend(documents)
                    
                except Exception as e:
                    print(f"Error processing {pdf_file}: {str(e)}")
        
        if not all_documents:
            print("No documents found. Please add PDFs to the subdirectories.")
            return
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        texts = self.text_splitter.split_documents(all_documents)
        print(f"Created {len(texts)} text chunks from {len(all_documents)} documents")
        
        # Create vector database with separate collections
        print("Creating vector database...")
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./multi_domain_chroma_db"
        )
        
        print("Vector database created successfully!")
    
    def setup_qa_chain(self):
        """Set up domain-aware question-answering chain"""
        if not self.vectorstore:
            print("No vector database found. Please load documents first.")
            return
        
        # Domain-specific prompt templates
        prompts = {
            'school': """You are a study assistant. Use the academic materials below to answer the student's question. Focus on educational concepts, explanations, and learning objectives.

Academic Context:
{context}

Student's Question: {question}

Educational Answer:""",
            
            'finance': """You are a personal finance assistant. Use the financial documents below to answer questions about spending, budgets, transactions, and financial patterns. Be specific with amounts and dates when available.

Financial Context:
{context}

Question: {question}

Financial Analysis:""",
            
            'general': """You are a helpful document assistant. Use the following context to answer the question accurately and helpfully.

Context:
{context}

Question: {question}

Answer:"""
        }
        
        # Use general prompt by default
        prompt_template = prompts['general']
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def detect_question_domain(self, question):
        """Automatically detect which domain the question belongs to"""
        question_lower = question.lower()
        
        # Financial keywords
        financial_keywords = ['spend', 'spent', 'money', 'cost', 'budget', 'transaction', 'bank', 
                             'payment', 'purchase', 'bought', 'expense', 'income', 'balance',
                             'statement', 'card', 'dollar', '$', 'financial', 'category']
        
        # Academic keywords  
        academic_keywords = ['study', 'learn', 'chapter', 'textbook', 'lecture', 'assignment',
                            'exam', 'test', 'course', 'class', 'professor', 'homework',
                            'concept', 'theory', 'definition', 'explain', 'academic']
        
        financial_score = sum(1 for keyword in financial_keywords if keyword in question_lower)
        academic_score = sum(1 for keyword in academic_keywords if keyword in question_lower)
        
        if financial_score > academic_score and financial_score > 0:
            return 'finance'
        elif academic_score > 0:
            return 'school'
        else:
            return 'general'
    
    def ask_question(self, question):
        """Ask a question with domain-aware processing"""
        if not self.qa_chain:
            return "Please load documents first."
        
        # Detect domain
        detected_domain = self.detect_question_domain(question)
        
        # Update prompt based on detected domain
        self.update_prompt_for_domain(detected_domain)
        
        print(f"\n🤔 Question: {question}")
        print(f"🎯 Detected domain: {detected_domain}")
        print("🔍 Searching through your documents...")
        
        # Get answer
        result = self.qa_chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result["source_documents"]
        
        print(f"\n💡 Answer: {answer}")
        
        # Show sources organized by type
        self.display_sources(source_docs)
        
        return answer
    
    def update_prompt_for_domain(self, domain):
        """Update the QA chain prompt based on domain"""
        prompts = {
            'school': """You are a study assistant. Use the academic materials below to answer the student's question. Focus on educational concepts, explanations, and learning objectives.

Academic Context:
{context}

Student's Question: {question}

Educational Answer:""",
            
            'finance': """You are a personal finance assistant. Use the financial documents below to answer questions about spending, budgets, transactions, and financial patterns. Be specific with amounts and dates when available.

Financial Context:
{context}

Question: {question}

Financial Analysis:""",
            
            'general': """You are a helpful document assistant. Use the following context to answer the question accurately and helpfully.

Context:
{context}

Question: {question}

Answer:"""
        }
        
        PROMPT = PromptTemplate(
            template=prompts.get(domain, prompts['general']),
            input_variables=["context", "question"]
        )
        
        # Update the QA chain prompt
        self.qa_chain.combine_documents_chain.llm_chain.prompt = PROMPT
    
    def display_sources(self, source_docs):
        """Display sources organized by document type"""
        print(f"\n📚 Sources used:")
        
        # Group sources by document type
        sources_by_type = {}
        for doc in source_docs:
            doc_type = doc.metadata.get('document_type', 'general')
            source_file = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            
            if doc_type not in sources_by_type:
                sources_by_type[doc_type] = []
            sources_by_type[doc_type].append(f"{source_file} (Page {page})")
        
        # Display organized sources
        for doc_type, sources in sources_by_type.items():
            print(f"  📁 {doc_type.title()} Documents:")
            for i, source in enumerate(sources, 1):
                print(f"    {i}. {source}")
    
    def get_document_stats(self):
        """Display statistics about loaded documents"""
        if not self.vectorstore:
            print("No documents loaded.")
            return
        
        # This is a simplified version - in a real implementation you'd query the vectorstore
        print("\n📊 Document Statistics:")
        
        for subdir in ['school', 'finance', 'general']:
            subdir_path = os.path.join(self.base_directory, subdir)
            if os.path.exists(subdir_path):
                pdf_count = len([f for f in os.listdir(subdir_path) if f.endswith('.pdf')])
                print(f"  📁 {subdir.title()}: {pdf_count} PDF files")
    
    def chat_loop(self):
        """Interactive chat with domain detection"""
        print("\n" + "="*60)
        print("🎓💰 Multi-Domain Document Assistant Ready!")
        print("Ask me about:")
        print("  📚 School materials (textbooks, notes, assignments)")  
        print("  💳 Financial documents (bank statements, spending)")
        print("  📄 Any other documents you've uploaded")
        print("\nType 'stats' to see document statistics")
        print("Type 'quit' to exit")
        print("="*60)
        
        # Show document stats
        self.get_document_stats()
        
        while True:
            question = input("\n📝 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye', '/bye']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'stats':
                self.get_document_stats()
                continue
            
            if not question:
                continue
            
            self.ask_question(question)

# Example usage
if __name__ == "__main__":
    print("Setting up Multi-Domain Document Assistant...")
    print("\nDirectory Structure:")
    print("./school/     - Put textbooks, notes, assignments here")
    print("./finance/    - Put bank statements, receipts here") 
    print("./general/    - Put any other documents here")
    
    # Create the multi-domain chatbot
    chatbot = SmartDocumentChatBot(
        base_directory="./",  # Uses current directory
        model_name="llama3.1:8b"
    )
    
    # Start interactive chat
    chatbot.chat_loop()
    
    # Example questions you could ask:
    # School: "What is the main concept in chapter 5?"
    # Finance: "How much did I spend on restaurants last month?"
    # Finance: "What are my biggest spending categories?"
    # General: "Summarize the key points from the latest document"