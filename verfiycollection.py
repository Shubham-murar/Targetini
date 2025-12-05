# verify_collection.py
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def verify_collection():
    load_dotenv()
    
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        check_compatibility=False  # Skip version check
    )
    
    try:
        # Get collection info
        collection_info = client.get_collection(collection_name="people_collection")
        
        print("=== COLLECTION VERIFICATION ===")
        print(f"Collection Name: people_collection")
        print(f"Total Points: {collection_info.points_count}")
        print(f"Status: {collection_info.status}")
        
        # Get actual count
        print(f"\n=== DETAILED COUNT ===")
        total_count = client.count(
            collection_name="people_collection",
            exact=True
        )
        print(f"Exact Count: {total_count.count}")
        
        # Sample some records to verify data
        print(f"\n=== SAMPLE RECORDS (First 3) ===")
        records, next_page = client.scroll(
            collection_name="people_collection",
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
        
        print(f"Found {len(records)} sample records:")
        for i, record in enumerate(records):
            print(f"\n--- Record {i+1} ---")
            print(f"ID: {record.id}")
            print(f"Payload keys: {list(record.payload.keys())}")
            
            # Check if metadata exists
            if 'metadata' in record.payload:
                metadata = record.payload['metadata']
                print(f"Name: {metadata.get('name', 'N/A')}")
                print(f"Position: {metadata.get('current_position', 'N/A')}")
                print(f"Company: {metadata.get('current_company', 'N/A')}")
                print(f"LinkedIn: {metadata.get('linkedin', 'N/A')}")
            else:
                print("No metadata found - checking direct fields:")
                for key, value in record.payload.items():
                    if key != 'page_content':
                        print(f"  {key}: {value}")
        
        return total_count.count
        
    except Exception as e:
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    count = verify_collection()
    print(f"\n🎯 FINAL VERIFICATION: {count} profiles in collection")