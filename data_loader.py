# data_loader.py
import pandas as pd
import os
import json
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from dotenv import load_dotenv

class LinkedInDataLoader:
    def __init__(self):
        self.setup_environment()
        self.setup_vector_store()
    
    def setup_environment(self):
        load_dotenv()
        
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("[ERROR] QDRANT_URL or QDRANT_API_KEY is not set in the .env file")
    
    def setup_vector_store(self):
        self.embeddings = FastEmbedEmbeddings()
        self.bm25_model = FastEmbedSparse(model_name="Qdrant/BM25")
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=120,  # Increased timeout for large datasets
            check_compatibility=False  # Skip version check to avoid warning
        )
        
        collection_name = "people_collection"
        
        # Create collection if it doesn't exist
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE, on_disk=True)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                }
            )
            print(f"[INFO] Created new collection: {collection_name}")
        else:
            print(f"[INFO] Using existing collection: {collection_name}")
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.bm25_model,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
    
    def add_documents_with_progress(self, documents, batch_size=100):
        """Add documents with progress tracking"""
        total_docs = len(documents)
        print(f"[INFO] Uploading {total_docs} documents in batches of {batch_size}...")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(batch)
            
            progress = min(i + batch_size, total_docs)
            percentage = (progress / total_docs) * 100
            print(f"[PROGRESS] Uploaded {progress}/{total_docs} documents ({percentage:.1f}%)")
        
        print(f"[SUCCESS] Completed uploading {total_docs} documents!")
    
    def load_csv_data(self, csv_file_path: str):
        """Load LinkedIn data from CSV file into Qdrant"""
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            print(f"[INFO] Loaded CSV with {len(df)} rows")
            print(f"[INFO] Columns: {df.columns.tolist()}")
            
            documents = []
            successful_records = 0
            skipped_records = 0
            
            for index, row in df.iterrows():
                try:
                    # Skip rows with missing essential data
                    if pd.isna(row.get('Linkedin')) or pd.isna(row.get('firstName')):
                        skipped_records += 1
                        continue
                    
                    # Create comprehensive document content
                    content_parts = []
                    
                    # Name
                    first_name = str(row['firstName']).strip() if pd.notna(row.get('firstName')) else ""
                    last_name = str(row['lastName']).strip() if pd.notna(row.get('lastName')) else ""
                    full_name = f"{first_name} {last_name}".strip()
                    
                    if full_name:
                        content_parts.append(f"Name: {full_name}")
                    
                    # About
                    if pd.notna(row.get('About')):
                        content_parts.append(f"About: {row['About']}")
                    
                    # Current position and company
                    if pd.notna(row.get('title')):
                        content_parts.append(f"Current Position: {row['title']}")
                    
                    if pd.notna(row.get('LastCompany')):
                        content_parts.append(f"Current Company: {row['LastCompany']}")
                    
                    # Location
                    if pd.notna(row.get('city')):
                        content_parts.append(f"Location: {row['city']}")
                    
                    # Previous positions and companies
                    if pd.notna(row.get('Position_1')):
                        content_parts.append(f"Previous Position: {row['Position_1']}")
                    
                    if pd.notna(row.get('SecondLastCompany')):
                        content_parts.append(f"Previous Company: {row['SecondLastCompany']}")
                    
                    if pd.notna(row.get('Position_2')):
                        content_parts.append(f"Earlier Position: {row['Position_2']}")
                    
                    # Combine all parts
                    content = "\n".join(content_parts)
                    
                    if not content.strip():
                        skipped_records += 1
                        continue
                    
                    # Create metadata
                    metadata = {
                        'name': full_name,
                        'linkedin': str(row['Linkedin']).strip(),
                        'location': str(row['city']).strip() if pd.notna(row.get('city')) else "",
                        'current_position': str(row['title']).strip() if pd.notna(row.get('title')) else "",
                        'current_company': str(row['LastCompany']).strip() if pd.notna(row.get('LastCompany')) else "",
                        'about': str(row['About']).strip() if pd.notna(row.get('About')) else "",
                        'previous_position': str(row['Position_1']).strip() if pd.notna(row.get('Position_1')) else "",
                        'previous_company': str(row['SecondLastCompany']).strip() if pd.notna(row.get('SecondLastCompany')) else "",
                        'source': 'linkedin_csv'
                    }
                    
                    # Create document
                    document = Document(page_content=content, metadata=metadata)
                    documents.append(document)
                    successful_records += 1
                    
                    # Print progress every 500 records
                    if (successful_records % 500 == 0):
                        print(f"[INFO] Processed {successful_records} records...")
                        
                except Exception as e:
                    print(f"[WARNING] Error processing row {index}: {e}")
                    skipped_records += 1
                    continue
            
            print(f"[INFO] Successfully processed {successful_records} out of {len(df)} records")
            print(f"[INFO] Skipped {skipped_records} records due to missing data or errors")
            
            # Add documents to vector store with progress tracking
            if documents:
                print("[INFO] Adding documents to Qdrant...")
                self.add_documents_with_progress(documents)
                
                # Verify the count
                final_count = self.check_collection_status()
                print(f"[SUCCESS] Data loading completed! Total vectors: {final_count}")
                
            else:
                print("[WARNING] No valid documents to add to vector store")
                
            return len(documents)
            
        except Exception as e:
            print(f"[ERROR] Failed to load CSV data: {e}")
            return 0
    
    def check_collection_status(self):
        """Check the current status of the collection"""
        try:
            collection_info = self.client.get_collection(collection_name="people_collection")
            return collection_info.points_count
        except Exception as e:
            print(f"[ERROR] Failed to check collection status: {e}")
            return 0
    
    def test_search(self, query: str = "software engineer", k: int = 3):
        """Test if search is working with loaded data"""
        try:
            print(f"\n[TEST] Testing search with query: '{query}'")
            results = self.vector_store.similarity_search(query, k=k)
            print(f"[TEST] Found {len(results)} results")
            
            for i, doc in enumerate(results):
                print(f"\n--- Test Result {i+1} ---")
                print(f"Content preview: {doc.page_content[:200]}...")
                print(f"Name: {doc.metadata.get('name', 'N/A')}")
                print(f"Position: {doc.metadata.get('current_position', 'N/A')}")
                print(f"Company: {doc.metadata.get('current_company', 'N/A')}")
            
            return len(results)
        except Exception as e:
            print(f"[TEST ERROR] Search test failed: {e}")
            return 0

def main():
    """Main function to load data"""
    print("[INFO] Starting LinkedIn Data Loader...")
    
    # Initialize the data loader
    loader = LinkedInDataLoader()
    
    # Check current collection status
    current_count = loader.check_collection_status()
    print(f"[INFO] Current vectors in collection: {current_count}")
    
    # Path to your CSV file
    csv_file_path = r"C:\Users\acer\Desktop\Targetini\ai_agents\sampleconnects.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"[ERROR] CSV file not found at: {csv_file_path}")
        return
    
    # Load the data
    print(f"[INFO] Loading data from: {csv_file_path}")
    loaded_count = loader.load_csv_data(csv_file_path)
    
    # Test search functionality
    if loaded_count > 0:
        print("\n[INFO] Testing search functionality...")
        loader.test_search("software engineer")
        loader.test_search("data scientist")
        loader.test_search("manager")

if __name__ == "__main__":
    main()


    