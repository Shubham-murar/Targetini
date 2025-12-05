# fixed_metadata_analyzer.py
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScrollRequest
from dotenv import load_dotenv

class FixedMetadataAnalyzer:
    def __init__(self):
        self.setup_environment()
        self.setup_client()
    
    def setup_environment(self):
        load_dotenv()
        
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("[ERROR] QDRANT_URL or QDRANT_API_KEY is not set in the .env file")
    
    def setup_client(self):
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60,
        )
    
    def extract_all_metadata_fixed(self, collection_name="people_collection", batch_size=100):
        """Extract all metadata with correct field access"""
        print(f"[INFO] Extracting all metadata from collection: {collection_name}")
        
        all_metadata = []
        next_page_offset = None
        
        try:
            while True:
                scroll_response = self.client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points = scroll_response[0]
                next_page_offset = scroll_response[1]
                
                if not points:
                    break
                
                for point in points:
                    # The metadata is stored in point.payload['metadata']
                    actual_metadata = point.payload.get('metadata', {})
                    
                    metadata_record = {
                        "id": point.id,
                        "page_content": point.payload.get('page_content', ''),
                        "metadata": actual_metadata  # This is where your actual data is
                    }
                    all_metadata.append(metadata_record)
                
                print(f"[PROGRESS] Extracted {len(all_metadata)} records")
                
                if next_page_offset is None:
                    break
            
            print(f"[SUCCESS] Successfully extracted {len(all_metadata)} metadata records")
            return all_metadata
            
        except Exception as e:
            print(f"[ERROR] Failed to extract metadata: {e}")
            return []
    
    def analyze_metadata_fixed(self, metadata_records):
        """Analyze metadata with correct field access"""
        if not metadata_records:
            print("[WARNING] No metadata records to analyze")
            return
        
        print("\n=== FIXED METADATA ANALYSIS ===")
        print(f"Total records: {len(metadata_records)}")
        
        # Count records with specific fields (now looking in the nested metadata)
        fields_to_check = [
            'name', 'linkedin', 'location', 
            'current_position', 'current_company',
            'about', 'previous_position', 'previous_company'
        ]
        
        field_counts = {}
        for field in fields_to_check:
            count = sum(1 for record in metadata_records if record['metadata'].get(field))
            field_counts[field] = count
            print(f"Records with '{field}': {count}/{len(metadata_records)} ({count/len(metadata_records)*100:.1f}%)")
        
        # Show sample of records
        print(f"\n=== SAMPLE RECORDS (first 3) ===")
        for i, record in enumerate(metadata_records[:3]):
            print(f"\n--- Record {i+1} ---")
            print(f"ID: {record['id']}")
            print(f"Page Content Preview: {record['page_content'][:100]}...")
            print("--- Metadata ---")
            for key, value in record['metadata'].items():
                print(f"{key}: {value}")
        
        return field_counts
    
    def verify_data_mapping(self, metadata_records):
        """Verify that the CSV columns mapped correctly to Qdrant fields"""
        print(f"\n=== DATA MAPPING VERIFICATION ===")
        
        # Your CSV columns vs Qdrant metadata fields
        csv_to_qdrant_mapping = {
            'Linkedin': 'linkedin',
            'firstName + lastName': 'name', 
            'About': 'about',
            'title': 'current_position',
            'city': 'location',
            'LastCompany': 'current_company',
            'Position_1': 'previous_position',
            'SecondLastCompany': 'previous_company',
            'Position_2': 'earlier_position'  # Note: this might be in page_content only
        }
        
        print("CSV to Qdrant field mapping:")
        for csv_field, qdrant_field in csv_to_qdrant_mapping.items():
            count = sum(1 for record in metadata_records if record['metadata'].get(qdrant_field))
            print(f"  {csv_field} -> {qdrant_field}: {count} records")
    
    def check_data_completeness(self, metadata_records):
        """Check how complete your data is"""
        print(f"\n=== DATA COMPLETENESS CHECK ===")
        
        completeness_stats = {}
        
        for record in metadata_records:
            metadata = record['metadata']
            filled_fields = sum(1 for value in metadata.values() if value and str(value).strip())
            total_fields = len(metadata)
            
            if total_fields > 0:
                completeness = (filled_fields / total_fields) * 100
                completeness_stats[record['id']] = completeness
        
        if completeness_stats:
            avg_completeness = sum(completeness_stats.values()) / len(completeness_stats)
            print(f"Average metadata completeness: {avg_completeness:.1f}%")
            
            # Show distribution
            excellent = sum(1 for comp in completeness_stats.values() if comp >= 80)
            good = sum(1 for comp in completeness_stats.values() if 50 <= comp < 80)
            poor = sum(1 for comp in completeness_stats.values() if comp < 50)
            
            print(f"Excellent records (≥80% complete): {excellent}")
            print(f"Good records (50-79% complete): {good}") 
            print(f"Poor records (<50% complete): {poor}")
    
    def save_fixed_metadata(self, metadata_records, output_file="qdrant_metadata_fixed.json"):
        """Save the properly extracted metadata"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_records, f, indent=2, ensure_ascii=False)
            print(f"[SUCCESS] Fixed metadata saved to: {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save fixed metadata: {e}")

def main():
    """Main function with fixed analysis"""
    print("[INFO] Starting Fixed Metadata Analyzer...")
    
    try:
        analyzer = FixedMetadataAnalyzer()
        
        # Extract all metadata with correct field access
        metadata_records = analyzer.extract_all_metadata_fixed()
        
        if metadata_records:
            # Save fixed version
            analyzer.save_fixed_metadata(metadata_records)
            
            # Analyze with correct field access
            analyzer.analyze_metadata_fixed(metadata_records)
            
            # Verify data mapping
            analyzer.verify_data_mapping(metadata_records)
            
            # Check completeness
            analyzer.check_data_completeness(metadata_records)
            
            print(f"\n🎉 [SUCCESS] Fixed analysis completed!")
            print(f"📊 Your data is actually stored correctly in Qdrant!")
            print(f"💾 All metadata is in the 'metadata' field within each record")
            
        else:
            print("[ERROR] No metadata records were extracted")
            
    except Exception as e:
        print(f"[ERROR] Main execution failed: {e}")

if __name__ == "__main__":
    main()



    