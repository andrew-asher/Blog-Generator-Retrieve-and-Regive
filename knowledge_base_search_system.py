import faiss
import os
import json
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Set up API keys 
google_api = os.environ["GOOGLE_API_KEY"]
# Faiss directory
faiss_dir = "/Users/andrewasher/XYGen_ai/ML/Knowledge-Based-Search-Retrieval-System-main/faiss-index"

class KnowledgeBaseSearchSystem:
    def __init__(self, document_texts_file="document_texts.json"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
        self.vector_store = None
        self.document_texts_file = document_texts_file
        self.document_texts = self.load_document_texts()

    def load_document_texts(self):
        if os.path.exists(self.document_texts_file):
            with open(self.document_texts_file, 'r') as f:
                return json.load(f)
        return {}

    def save_document_texts(self):
        with open(self.document_texts_file, 'w') as f:
            json.dump(self.document_texts, f)

    def load_documents(self, file_paths: List[str]):
        if os.path.exists("faiss_index"):
            self.vector_store = FAISS.load_local(faiss_dir, self.embeddings)
        
        documents = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            if file_name not in self.document_texts:
                loader = PyPDFLoader(file_path)
                pdf_documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, 
                                                               chunk_overlap=500,
                                                               separators = ["\n\n", "\n# ", "\n- ", "\n\t"],
                                                               length_function=len)
                split_docs = text_splitter.split_documents(pdf_documents)

                self.document_texts[file_name] = {}
                for doc in split_docs:
                    page_label = doc.metadata.get('page_label', str(doc.metadata.get('page', 1)))
                    if page_label not in self.document_texts[file_name]:
                        self.document_texts[file_name][page_label] = ""
                    self.document_texts[file_name][page_label] += doc.page_content

                documents.extend(split_docs)
            else:
                # Reconstruct documents from stored text
                for page_label, content in self.document_texts[file_name].items():
                    documents.append(Document(page_content=content, metadata={'source': file_path, 
                                                                              'page_label': page_label}))

        self.save_document_texts()

        if not self.vector_store:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        self.vector_store.save_local(faiss_dir)

    def llm_process_query(self, query: str) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
        
        new_db = FAISS.load_local(faiss_dir, self.embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(query)

        prompt_template = """
            You are an expert AI assistant for an advanced document retrieval system. 
            Carefully analyze the context provided to deliver a concise, accurate, and professional summary in response to the user's query. 
            If the context does not contain the necessary information, clearly state that the information is not available, without speculating.

            Context:
            {context}

            User Query:
            {query}

            Expert Response:
        """

        # promt
        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "query"])
        
        # load the chain
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)

        response = chain({"input_documents":docs, "query": query}, return_only_outputs=True)

        return response["output_text"]

    def search_documents(self, query: str) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
        results = []
        for file_name, pages in self.document_texts.items():
            total_occurrences = 0
            page_occurrences = []
            for page_label, content in pages.items():
                count = content.lower().count(query.lower())
                if count > 0:
                    total_occurrences += count
                    page_occurrences.append((page_label, count))
            if total_occurrences > 0:
                results.append((file_name, total_occurrences, page_occurrences))
        
        llm_response = self.llm_process_query(query)
        
        return results, llm_response
    
    def qa_conversation(self, query: str) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
        
        new_db = FAISS.load_local(faiss_dir, self.embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(query)

        prompt_template = """
            Please provide a comprehensive and accurate response based on the context provided. Include all relevant details in your answer. 
            If the context does not contain the necessary information, clearly state, "The answer is not available in the provided context."
            
            Context:
            {context}

            Question:
            {question}

            Answer:
        """

        # promt
        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
        
        # load the chain
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)

        response = chain({"input_documents":docs, "question": query}, return_only_outputs=True)

        return response["output_text"]