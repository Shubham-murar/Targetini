

from advanced_agent import TargetiniAgent, AdvancedRecruitingAgent
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import uuid
import threading
from datetime import datetime
from typing import List, Dict
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from dotenv import load_dotenv
import re

app = Flask(__name__)
CORS(app, resources={r"/api/*": {
    "origins": [
        "http://localhost:3000",
        "http://localhost:5000"
    ]}
})
CORS(app)

# For ASGI compatibility
from asgiref.wsgi import WsgiToAsgi
asgi_app = WsgiToAsgi(app)

# Global storage for background searches
active_searches = {}

class LinkedinRAG:
    def __init__(self):
        self.setup_environment()
        self.setup_vector_store()
        self.setup_llm()
        self.build_graph()
    
    def setup_environment(self):
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("[ERROR] GOOGLE_API_KEY is not set in the .env file")
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
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
            timeout=60,
        )
        
        collection_name = "people_collection"
        
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE, on_disk=True)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                }
            )
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.bm25_model,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
    
    def setup_llm(self):
        # Analyzer: Creative to find synonyms
        self.llm_analyzer = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        # Extractor: Strict to ensure data accuracy
        self.llm_extractor = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
        # --- PROMPT 1: THE RECRUITER BRAIN ---
        self.ANALYSIS_TEMPLATE = """
        You are an Expert Technical Recruiter.
        User Query: "{query}"
        
        Task: Break this query down to find EVERY relevant person.
        1. Identify the Core Role.
        2. Identify Domain/Skills.
        3. Identify Seniority.
        
        Output valid JSON only:
        {{
            "expanded_queries": [
                "Query 1: Wide search with synonyms",
                "Query 2: Domain specific search",
                "Query 3: Exact intent search"
            ]
        }}
        """

        # --- PROMPT 2: THE PROFILE EXTRACTOR ---
        self.SYSTEM_TEMPLATE = """
        You are an expert Talent Sourcer.
        User Query: "{query}"
        
        GOAL: Extract and Return ALL profiles relevant to the query. 
        High Recall is priority. If unsure, INCLUDE them.
        
        INSTRUCTIONS:
        1. Match titles, skills, and past experience.
        2. A former "Founder" matches "Founder" queries.
        3. "Building AI" matches "AI Engineer".
        
        Return a JSON array of objects:
        [
          {{
            "name": "Full Name",
            "linkedin": "https://linkedin.com/in/...",
            "location": "City",
            "current_position": "Role",
            "current_company": "Company",
            "match_reason": "Why they match"
          }}
        ]
        """

    def build_graph(self):
        class RAGState(TypedDict):
            question: str
            context: List[Document]
            answer: str
            analysis: Dict
        
        def analyze_query(state: RAGState):
            print(f"[INFO] Analyzing query intent: {state['question']}")
            messages = [{"role": "user", "content": self.ANALYSIS_TEMPLATE.format(query=state["question"])}]
            response = self.llm_analyzer.invoke(messages)
            try:
                content = response.content.replace('```json', '').replace('```', '')
                analysis = json.loads(content)
                print(f"[INFO] Query Expanded: {analysis.get('expanded_queries')}")
                return {"analysis": analysis}
            except:
                return {"analysis": {"expanded_queries": [state["question"]]}}

        def multi_query_search(state: RAGState):
            analysis = state.get("analysis", {})
            queries = analysis.get("expanded_queries", [state["question"]])
            all_docs = []
            
            for q in queries:
                print(f"[INFO] Searching Qdrant for: '{q}'")
                # Increased K to 30 to get maximum recall before filtering
                docs = self.vector_store.similarity_search(q, k=30)
                all_docs.extend(docs)
                
            unique_docs = self.deduplicate_documents(all_docs)
            print(f"[INFO] Unique candidates before filtering: {len(unique_docs)}")
            return {"context": unique_docs}
        
        def generate(state: RAGState):
            if not state["context"]: return {"answer": []}

            enhanced_context = []
            for doc in state["context"]:
                meta = doc.metadata
                if 'metadata' in meta and isinstance(meta['metadata'], dict):
                    meta = meta['metadata']
                
                entry = (
                    f"Name: {meta.get('name')}\n"
                    f"Role: {meta.get('current_position')}\n"
                    f"Company: {meta.get('current_company')}\n"
                    f"Loc: {meta.get('location')}\n"
                    f"About: {meta.get('about') or ''}\n"
                    f"URL: {meta.get('linkedin')}\n"
                    f"History: {meta.get('previous_position')} at {meta.get('previous_company')}\n"
                    "---"
                )
                enhanced_context.append(entry)
            
            context_str = "\n".join(enhanced_context)
            prompt = self.SYSTEM_TEMPLATE.format(query=state["question"])
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Candidates:\n{context_str}"},
            ]
            
            print("[INFO] Filtering candidates with LLM...")
            try:
                response = self.llm_extractor.invoke(messages)
                match = re.search(r"\[\s*{.*?}\s*\]", response.content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    valid_results = []
                    for p in parsed:
                        if p.get('name') and p.get('linkedin'):
                            valid_results.append(p)
                    print(f"[INFO] Final results: {len(valid_results)}")
                    return {"answer": valid_results}
            except Exception as e:
                print(f"[ERROR] LLM Filter failed: {e}")
            
            return {"answer": []}
        
        workflow = StateGraph(RAGState)
        workflow.add_node("analyze", analyze_query)
        workflow.add_node("search", multi_query_search)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", "search")
        workflow.add_edge("search", "generate")
        workflow.add_edge("generate", END)
        self.graph = workflow.compile()
    
    def deduplicate_documents(self, documents):
        seen = set()
        unique_docs = []
        for doc in documents:
            meta = doc.metadata
            if 'metadata' in meta and isinstance(meta['metadata'], dict):
                meta = meta['metadata']
            uid = meta.get('linkedin') or meta.get('name')
            if uid and uid not in seen:
                seen.add(uid)
                unique_docs.append(doc)
        return unique_docs
    
    def query(self, question: str):
        try:
            result = self.graph.invoke({"question": question})
            return result["answer"]
        except Exception as e:
            print(f"[ERROR] Query workflow failed: {e}")
            return []

# Initialize RAG system
print("[INFO] Initializing Linkedin Intelligent RAG...")
rag_system = LinkedinRAG()
print("[INFO] System Ready!")

# Initialize AI Agents
print("[INFO] Initializing AI Recruiting Agents...")
try:
    # Initialize conversational agent
    conversational_agent = TargetiniAgent(rag_system)
    # Initialize learning agent
    learning_agent = AdvancedRecruitingAgent(rag_system)
    print("[INFO] AI Agents Ready!")
    AGENTS_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Could not initialize AI Agents: {e}")
    conversational_agent = None
    learning_agent = None
    AGENTS_AVAILABLE = False

def process_search_background(search_id, query):
    """Process search in background thread"""
    try:
        print(f"[BACKGROUND] Processing search: {query}")
        
        # Update status to processing
        active_searches[search_id]['status'] = 'processing'
        
        # Your existing RAG pipeline
        all_results = rag_system.query(query)
        
        # Update with results
        active_searches[search_id].update({
            'status': 'completed',
            'results': all_results,
            'completed_at': datetime.utcnow().isoformat(),
            'total_count': len(all_results)
        })
        
        print(f"[BACKGROUND] Search completed: {len(all_results)} results")
        
    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        active_searches[search_id].update({
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.utcnow().isoformat()
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_people():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400

        print(f"\n[REQUEST] New search query: {query}")
        
        # Generate unique search ID
        search_id = str(uuid.uuid4())
        
        # Store search in active searches
        active_searches[search_id] = {
            'status': 'queued',
            'query': query,
            'started_at': datetime.utcnow().isoformat(),
            'results': None,
            'error': None,
            'total_count': 0
        }
        
        # Start background processing
        thread = threading.Thread(
            target=process_search_background,
            args=(search_id, query)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'search_id': search_id,
            'status': 'queued',
            'message': 'Search started in background'
        })
        
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search/<search_id>', methods=['GET'])
def get_search_status(search_id):
    """Check search status and get results"""
    if search_id not in active_searches:
        return jsonify({'success': False, 'error': 'Search not found'}), 404
    
    search_data = active_searches[search_id]
    
    # Clean up old completed searches (older than 1 hour)
    if search_data['status'] in ['completed', 'error']:
        completed_time = datetime.fromisoformat(search_data['completed_at'])
        if (datetime.utcnow() - completed_time).total_seconds() > 3600:  # 1 hour
            del active_searches[search_id]
            return jsonify({'success': False, 'error': 'Search expired'}), 404
    
    response_data = {
        'success': True,
        'search_id': search_id,
        'status': search_data['status'],
        'query': search_data['query']
    }
    
    # Add results if completed
    if search_data['status'] == 'completed':
        response_data['results'] = search_data['results']
        response_data['total_count'] = search_data['total_count']
    elif search_data['status'] == 'error':
        response_data['error'] = search_data['error']
    
    return jsonify(response_data)

# NEW: AI Agent Endpoints
@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    """Chat with the AI recruiting agent"""
    if not AGENTS_AVAILABLE or not conversational_agent:
        return jsonify({
            'success': False, 
            'error': 'AI Agent not available'
        }), 503
    
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        user_id = data.get("user_id", "default")
        
        if not user_input:
            return jsonify({'success': False, 'error': 'Message is required'}), 400
        
        response = conversational_agent.chat(user_input)
        return jsonify({
            "success": True,
            "response": response["response"],
            "agent_thinking": response.get("agent_thinking", ""),
            "reasoning_steps": response.get("reasoning_steps", []),
            "is_agentic": True
        })
    except Exception as e:
        print(f"[AGENT ERROR] Chat failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/agent/learn', methods=['POST'])
def learn_from_feedback():
    """Let the agent learn from user feedback"""
    if not AGENTS_AVAILABLE or not learning_agent:
        return jsonify({
            'success': False, 
            'error': 'Learning Agent not available'
        }), 503
    
    try:
        data = request.get_json()
        user_id = data.get("user_id", "default")
        feedback = data.get("feedback", {})
        
        if not feedback:
            return jsonify({'success': False, 'error': 'Feedback is required'}), 400
        
        learning_result = learning_agent.learn_preferences(feedback, user_id)
        
        return jsonify({
            "success": True,
            "message": "Agent learned from your feedback",
            "learning_result": learning_result,
            "current_preferences": learning_agent.learned_preferences.get(user_id, {})
        })
    except Exception as e:
        print(f"[AGENT ERROR] Learning failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/agent/adaptive-search', methods=['POST'])
def adaptive_search():
    """Use the learning agent for adaptive search"""
    if not AGENTS_AVAILABLE or not learning_agent:
        return jsonify({
            'success': False, 
            'error': 'Learning Agent not available'
        }), 503
    
    try:
        data = request.get_json()
        query = data.get("query", "")
        user_id = data.get("user_id", "default")
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400
        
        results = learning_agent.process_request(query, user_id)
        
        return jsonify({
            "success": True,
            "agent_type": "adaptive_learning_agent",
            "results": results["results"],
            "patterns": results["patterns_recognized"],
            "personalized": results["personalized"],
            "explanation": results["explanation"]
        })
    except Exception as e:
        print(f"[AGENT ERROR] Adaptive search failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/agent/status', methods=['GET'])
def agent_status():
    """Get agent status and capabilities"""
    agent_info = []
    
    if AGENTS_AVAILABLE:
        if conversational_agent:
            agent_info.append({
                "name": "Conversational Recruiting Agent",
                "type": "langchain_agent",
                "capabilities": ["conversation", "tool_usage", "reasoning"],
                "has_memory": True,
                "status": "active"
            })
        
        if learning_agent:
            agent_info.append({
                "name": "Learning Recruiting Agent",
                "type": "adaptive_agent",
                "capabilities": ["preference_learning", "pattern_recognition", "adaptive_search"],
                "learned_users": len(learning_agent.learned_preferences),
                "status": "active"
            })
    
    return jsonify({
        "agents_available": agent_info,
        "total_agents": len(agent_info),
        "langgraph_workflows": ["reasoning_graph", "search_orchestration", "agent_orchestration"],
        "hackathon_alignment": "AI Agents with LangChain/LangGraph - Targetini: Autonomous Recruiting Agent System"
    })

@app.route('/api/agent/quick-search', methods=['POST'])
def agent_quick_search():
    """Quick search using the conversational agent"""
    if not AGENTS_AVAILABLE or not conversational_agent:
        # Fall back to regular search
        return search_people()
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400
        
        # Use agent for a smarter search
        agent_response = conversational_agent.chat(f"Find professionals for: {query}")
        
        # Also get regular results for comparison
        regular_results = rag_system.query(query)
        
        return jsonify({
            "success": True,
            "agent_response": agent_response["response"],
            "agent_thinking": agent_response.get("agent_thinking", ""),
            "regular_results_count": len(regular_results),
            "agent_enhanced": True,
            "suggestions": agent_response.get("suggestions", [])
        })
    except Exception as e:
        print(f"[AGENT ERROR] Quick search failed: {e}")
        # Fall back to regular search
        return search_people()

@app.route('/api/agent/patterns', methods=['GET'])
def get_agent_patterns():
    """Get search patterns recognized by the agent"""
    if not AGENTS_AVAILABLE or not learning_agent:
        return jsonify({
            'success': False, 
            'error': 'Learning Agent not available'
        }), 503
    
    try:
        patterns = learning_agent.recognize_patterns(learning_agent.search_patterns)
        return jsonify({
            "success": True,
            "patterns": patterns,
            "total_searches": len(learning_agent.search_patterns)
        })
    except Exception as e:
        print(f"[AGENT ERROR] Getting patterns failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'active_searches': len(active_searches),
        'rag_system': 'ready',
        'ai_agents': AGENTS_AVAILABLE,
        'agent_count': 2 if AGENTS_AVAILABLE else 0,
        'features': {
            'rag_pipeline': True,
            'langgraph_workflow': True,
            'ai_agents': AGENTS_AVAILABLE,
            'adaptive_learning': AGENTS_AVAILABLE,
            'conversational_agent': AGENTS_AVAILABLE
        }
    })

if __name__ == '__main__':
    # Display startup information
    print("\n" + "="*60)
    print("Targetini - AI Recruiting Agent System")
    print("="*60)
    print(f"RAG System: {'READY' if rag_system else 'NOT READY'}")
    print(f"AI Agents: {'READY' if AGENTS_AVAILABLE else 'NOT READY'}")
    print(f"Total Endpoints: 9 (including agent endpoints)")
    print(f"Hackathon Alignment: AI Agents with LangChain/LangGraph")
    print("="*60)
    print("\nAvailable Endpoints:")
    print("  - /                    : Main UI")
    print("  - /api/search          : Regular search")
    print("  - /api/agent/chat      : Chat with AI agent")
    print("  - /api/agent/learn     : Teach agent preferences")
    print("  - /api/agent/adaptive-search : Personalized search")
    print("  - /api/agent/status    : Agent capabilities")
    print("  - /api/agent/quick-search : Agent-enhanced search")
    print("  - /api/agent/patterns  : Search pattern analysis")
    print("  - /health              : System health")
    print("\nStarting server...")
    
    app.run(debug=True, host='0.0.0.0', port=5004)