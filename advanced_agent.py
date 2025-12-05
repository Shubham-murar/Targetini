"""Advanced AI Agent Module for Targetini
Adds conversational memory, tool usage, and autonomous decision-making
"""

from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, get_buffer_string
import json
import re

class TargetiniAgent:
    """
    Enhanced AI Agent with conversation memory and tool usage
    """
    
    def __init__(self, rag_system):
        self.rag = rag_system
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agent()
        self.build_agent_graph()
    
    def setup_agent(self):
        """Setup the LangChain agent with tools - FIXED VERSION"""
        
        # Define tools for the agent
        tools = [
            Tool(
                name="LinkedIn_Search",
                func=self.search_linkedin,
                description="Search LinkedIn profiles using natural language queries"
            ),
            Tool(
                name="Refine_Search",
                func=self.refine_search,
                description="Refine search results based on user feedback"
            ),
            Tool(
                name="Extract_Skills",
                func=self.extract_skills,
                description="Extract key skills from search results"
            ),
            Tool(
                name="Rank_Candidates",
                func=self.rank_candidates,
                description="Rank candidates by relevance score"
            ),
            Tool(
                name="Suggest_Queries",
                func=self.suggest_better_queries,
                description="Suggest improved search queries based on conversation"
            )
        ]
        
        # FIXED: Proper ReAct prompt template
        agent_prompt = PromptTemplate.from_template(
            """You are Targetini - an advanced AI Recruiting Agent. 
            You help users find the perfect professionals from their LinkedIn network.
            
            You have access to the following tools:
            
            {tools}
            
            Use the following format:
            
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Begin!
            
            Previous conversation history:
            {chat_history}
            
            Question: {input}
            Thought:{agent_scratchpad}"""
        )
        
        # Create agent
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_output_tokens=1000
        )
        
        self.agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=agent_prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True
        )
    
    def build_agent_graph(self):
        """Build a LangGraph for multi-step agent reasoning"""
        
        from typing import TypedDict
        
        class AgentState(TypedDict):
            """State for the agent graph"""
            messages: List[Dict]
            user_query: str
            context: List[Dict]
            results: List[Dict]
            reasoning_steps: List[str]
            current_step: str
            needs_clarification: bool
        
        # Define graph nodes
        def understand_intent(state: AgentState):
            """Analyze user intent"""
            return {
                "reasoning_steps": state.get("reasoning_steps", []) + 
                [f"Step 1: Understanding user intent - {state['user_query']}"],
                "current_step": "analyzing_intent"
            }
        
        def plan_search_strategy(state: AgentState):
            """Plan the search strategy"""
            return {
                "reasoning_steps": state.get("reasoning_steps", []) + 
                ["Step 2: Planning optimal search strategy"],
                "current_step": "planning_strategy"
            }
        
        def execute_search(state: AgentState):
            """Execute the search"""
            results = self.rag.query(state["user_query"])
            return {
                "results": results,
                "reasoning_steps": state.get("reasoning_steps", []) + 
                [f"Step 3: Executed search - found {len(results)} results"],
                "current_step": "executing_search"
            }
        
        def evaluate_results(state: AgentState):
            """Evaluate and filter results"""
            return {
                "reasoning_steps": state.get("reasoning_steps", []) + 
                ["Step 4: Evaluating result quality and relevance"],
                "current_step": "evaluating_results"
            }
        
        def prepare_response(state: AgentState):
            """Prepare final response"""
            return {
                "reasoning_steps": state.get("reasoning_steps", []) + 
                ["Step 5: Preparing intelligent response with insights"],
                "current_step": "preparing_response"
            }
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        workflow.add_node("understand_intent", understand_intent)
        workflow.add_node("plan_strategy", plan_search_strategy)
        workflow.add_node("execute_search", execute_search)
        workflow.add_node("evaluate_results", evaluate_results)
        workflow.add_node("prepare_response", prepare_response)
        
        # Define the flow
        workflow.add_edge("understand_intent", "plan_strategy")
        workflow.add_edge("plan_strategy", "execute_search")
        workflow.add_edge("execute_search", "evaluate_results")
        workflow.add_edge("evaluate_results", "prepare_response")
        workflow.add_edge("prepare_response", END)
        
        workflow.set_entry_point("understand_intent")
        self.agent_graph = workflow.compile()
    
    # Tool implementations remain the same...
    def search_linkedin(self, query: str) -> str:
        """Search tool for the agent"""
        results = self.rag.query(query)
        
        # Format results nicely
        if not results:
            return "No results found. Try broadening your search terms."
        
        summary = f"Found {len(results)} professionals matching: {query}\n\n"
        summary += "Top 3 matches:\n"
        
        for i, person in enumerate(results[:3]):
            summary += f"\n{i+1}. {person.get('name', 'N/A')}"
            summary += f"\n   Position: {person.get('current_position', 'N/A')}"
            summary += f"\n   Company: {person.get('current_company', 'N/A')}"
            summary += f"\n   Location: {person.get('location', 'N/A')}"
            summary += f"\n   Match: {person.get('match_reason', 'N/A')}\n"
        
        return summary
    
    def refine_search(self, feedback: str) -> str:
        """Refine search based on feedback"""
        # Extract key terms from feedback
        refinements = []
        
        if "senior" in feedback.lower():
            refinements.append("Add 'senior' or 'lead' to titles")
        if "junior" in feedback.lower() or "entry" in feedback.lower():
            refinements.append("Focus on entry-level or junior roles")
        if "specific" in feedback.lower():
            refinements.append("Use more specific domain keywords")
        if "broad" in feedback.lower():
            refinements.append("Use broader search terms")
        
        if refinements:
            return f"Suggested refinements: {', '.join(refinements)}"
        return "Try being more specific about what you didn't like in previous results."
    
    def extract_skills(self, results: str) -> str:
        """Extract skills from results"""
        try:
            # Parse results if they're in JSON format
            if results.startswith("Found"):
                # Extract skills from the text
                skills = set()
                common_skills = [
                    "Python", "JavaScript", "Java", "React", "Node.js",
                    "Machine Learning", "AI", "Data Science", "AWS",
                    "Azure", "Docker", "Kubernetes", "SQL", "NoSQL",
                    "React", "Angular", "Vue", "TypeScript", "Go",
                    "Rust", "C++", "C#", "Swift", "Kotlin"
                ]
                
                for skill in common_skills:
                    if skill.lower() in results.lower():
                        skills.add(skill)
                
                if skills:
                    return f"Common skills found: {', '.join(sorted(skills))}"
                return "No specific technical skills detected in results."
            return "Could not parse results for skill extraction."
        except:
            return "Error extracting skills."
    
    def rank_candidates(self, candidates: str) -> str:
        """Rank candidates"""
        return "Candidates would be ranked by:\n1. Relevance to query\n2. Experience level\n3. Location match\n4. Company reputation\n5. Skill match percentage"
    
    def suggest_better_queries(self, history: str) -> str:
        """Suggest better queries"""
        suggestions = [
            "Try adding specific technologies: 'Python developer with Django experience'",
            "Include location if important: 'Product manager in San Francisco'",
            "Specify seniority level: 'Senior data scientist with 5+ years experience'",
            "Mention specific industries: 'Fintech software engineer'",
            "Combine role with skills: 'Frontend developer with React and TypeScript'"
        ]
        return "Query suggestions:\n- " + "\n- ".join(suggestions)
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Main chat interface for the agent - FIXED VERSION
        """
        try:
            # Use the agent executor for conversational interaction
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": get_buffer_string(self.memory.chat_memory.messages)
            })
            
            # Extract intermediate steps for reasoning display
            reasoning_steps = []
            if "intermediate_steps" in response:
                for action, observation in response["intermediate_steps"]:
                    if hasattr(action, 'tool'):
                        reasoning_steps.append(f"Used {action.tool} tool: {str(observation)[:100]}...")
            
            # Also run through the reasoning graph for advanced tasks
            if "search" in user_input.lower() or "find" in user_input.lower():
                # Extract just the search query part
                search_query = user_input
                if "for" in user_input.lower():
                    # Try to extract what they're looking for
                    parts = user_input.lower().split("for")
                    if len(parts) > 1:
                        search_query = parts[1].strip()
                
                graph_state = self.agent_graph.invoke({
                    "user_query": search_query,
                    "messages": [{"role": "user", "content": user_input}],
                    "reasoning_steps": [],
                    "context": [],
                    "results": [],
                    "current_step": "",
                    "needs_clarification": False
                })
                
                return {
                    "response": response["output"],
                    "reasoning_steps": reasoning_steps + graph_state["reasoning_steps"],
                    "agent_thinking": "I analyzed your request through 5 reasoning steps to find the best matches",
                    "results_count": len(graph_state.get("results", [])),
                    "suggestions": [
                        "Try being more specific with technologies",
                        "Consider adding location filters",
                        "Specify experience level for better results"
                    ],
                    "is_agentic": True
                }
            
            return {
                "response": response["output"],
                "reasoning_steps": reasoning_steps,
                "agent_thinking": "I used my specialized tools and memory to provide the best answer for you",
                "is_agentic": True
            }
            
        except Exception as e:
            print(f"[AGENT ERROR] Chat execution failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to simple response without agent
            return {
                "response": f"I can help you find professionals! Try asking me something like 'Find Python developers in San Francisco' or 'Search for marketing managers with startup experience'.",
                "reasoning_steps": ["Agent temporarily unavailable, using fallback"],
                "agent_thinking": "Having technical difficulties with my advanced tools",
                "is_agentic": False
            }

# The rest of the AdvancedRecruitingAgent class remains the same...

class AdvancedRecruitingAgent:
    """
    Even more advanced agent with learning capabilities
    """
    
    def __init__(self, rag_system):
        self.rag = rag_system
        self.learned_preferences = {}
        self.search_patterns = []
    
    def adaptive_search(self, query: str, user_id: str = "default") -> List[Dict]:
        """
        Search that adapts to learned user preferences
        """
        base_results = self.rag.query(query)
        
        # Apply learned preferences
        if user_id in self.learned_preferences:
            preferences = self.learned_preferences[user_id]
            filtered = self._apply_preferences(base_results, preferences)
            return filtered
        
        return base_results
    
    def learn_preferences(self, feedback: Dict, user_id: str = "default"):
        """
        Learn from user feedback to improve future searches
        """
        if user_id not in self.learned_preferences:
            self.learned_preferences[user_id] = {
                "preferred_locations": set(),
                "avoided_companies": set(),
                "preferred_titles": set(),
                "preferred_skills": set(),
                "seniority_preference": None,
                "company_size_preference": None
            }
        
        # Update preferences based on feedback
        if "liked" in feedback:
            for profile in feedback["liked"]:
                self._update_preferences_from_profile(
                    profile, self.learned_preferences[user_id], positive=True
                )
        
        if "disliked" in feedback:
            for profile in feedback["disliked"]:
                self._update_preferences_from_profile(
                    profile, self.learned_preferences[user_id], positive=False
                )
        
        # Extract preferences from text feedback
        if "text_feedback" in feedback:
            self._extract_preferences_from_text(
                feedback["text_feedback"], self.learned_preferences[user_id]
            )
        
        return f"Learned from {len(feedback.get('liked', []))} positive and {len(feedback.get('disliked', []))} negative examples"
    
    def _update_preferences_from_profile(self, profile: Dict, preferences: Dict, positive: bool):
        """Update preferences from a single profile"""
        if positive:
            if profile.get("location"):
                preferences["preferred_locations"].add(profile["location"])
            if profile.get("current_position"):
                preferences["preferred_titles"].add(profile["current_position"])
            if profile.get("current_company"):
                # Extract skills from about or position
                about_text = profile.get("about", "") + " " + profile.get("current_position", "")
                self._extract_skills_from_text(about_text, preferences["preferred_skills"])
        else:
            if profile.get("current_company"):
                preferences["avoided_companies"].add(profile["current_company"])
    
    def _extract_skills_from_text(self, text: str, skills_set: set):
        """Extract skills from text"""
        common_skills = [
            "python", "javascript", "java", "react", "node", "machine learning",
            "ai", "data science", "aws", "azure", "docker", "kubernetes",
            "sql", "nosql", "typescript", "go", "rust", "c++", "c#", "swift"
        ]
        
        text_lower = text.lower()
        for skill in common_skills:
            if skill in text_lower:
                skills_set.add(skill.title())
    
    def _extract_preferences_from_text(self, text: str, preferences: Dict):
        """Extract preferences from text feedback"""
        text_lower = text.lower()
        
        if "senior" in text_lower:
            preferences["seniority_preference"] = "senior"
        elif "junior" in text_lower or "entry" in text_lower:
            preferences["seniority_preference"] = "junior"
        
        if "startup" in text_lower:
            preferences["company_size_preference"] = "startup"
        elif "big" in text_lower or "large" in text_lower or "enterprise" in text_lower:
            preferences["company_size_preference"] = "large"
    
    def _apply_preferences(self, results: List[Dict], preferences: Dict) -> List[Dict]:
        """Apply learned preferences to filter results"""
        filtered = []
        for result in results:
            score = 0
            reasons = []
            
            # Location preference
            if result.get("location") in preferences["preferred_locations"]:
                score += 2
                reasons.append(f"Matches preferred location: {result['location']}")
            
            # Title preference
            if result.get("current_position") in preferences["preferred_titles"]:
                score += 1
                reasons.append(f"Matches preferred title: {result['current_position']}")
            
            # Skill preference
            about_text = (result.get("about", "") + " " + result.get("current_position", "")).lower()
            for skill in preferences["preferred_skills"]:
                if skill.lower() in about_text:
                    score += 1
                    reasons.append(f"Has preferred skill: {skill}")
            
            # Company avoidance
            if result.get("current_company") in preferences["avoided_companies"]:
                score -= 3
                reasons.append(f"Avoided company: {result['current_company']}")
            
            # Seniority preference
            if preferences["seniority_preference"]:
                position = result.get("current_position", "").lower()
                if preferences["seniority_preference"] == "senior" and any(word in position for word in ["senior", "lead", "principal", "head", "director"]):
                    score += 1
                    reasons.append("Matches seniority preference")
                elif preferences["seniority_preference"] == "junior" and any(word in position for word in ["junior", "associate", "entry", "graduate"]):
                    score += 1
                    reasons.append("Matches junior preference")
            
            if score >= 0:  # Only include non-negative matches
                result["preference_score"] = score
                result["preference_reasons"] = reasons
                filtered.append(result)
        
        # Sort by preference score
        filtered.sort(key=lambda x: x.get("preference_score", 0), reverse=True)
        return filtered
    
    def recognize_patterns(self, search_history: List[str]) -> Dict:
        """Recognize patterns in user search behavior"""
        from collections import Counter
        
        if not search_history:
            return {"common_terms": {}, "suggested_queries": []}
        
        all_terms = []
        for search in search_history:
            # Clean and tokenize
            terms = re.findall(r'\b\w+\b', search.lower())
            all_terms.extend(terms)
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        filtered_terms = [term for term in all_terms if term not in stop_words and len(term) > 2]
        
        term_freq = Counter(filtered_terms)
        common_patterns = term_freq.most_common(5)
        
        return {
            "common_terms": dict(common_patterns),
            "suggested_queries": self._generate_suggestions(common_patterns)
        }
    
    def _generate_suggestions(self, patterns: List) -> List[str]:
        """Generate query suggestions based on patterns"""
        if not patterns:
            return []
        
        suggestions = []
        top_terms = [term for term, freq in patterns[:3]]
        
        if "engineer" in top_terms:
            suggestions.append(f"Senior {top_terms[0]} with 5+ years experience")
            suggestions.append(f"{top_terms[0]} with remote work experience")
        if "manager" in top_terms:
            suggestions.append(f"{top_terms[0]} with team leadership experience")
        if "data" in top_terms:
            suggestions.append(f"{top_terms[0]} scientist with machine learning expertise")
        if "software" in top_terms:
            suggestions.append(f"{top_terms[0]} developer with full-stack experience")
        
        # Add general suggestions
        suggestions.append(f"{top_terms[0]} in tech industry")
        suggestions.append(f"{top_terms[0]} with startup experience")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def optimize_strategy(self, performance_data: Dict) -> str:
        """Optimize search strategy based on performance"""
        success_rate = performance_data.get("success_rate", 0)
        avg_results = performance_data.get("avg_results", 0)
        
        if success_rate < 0.3:
            return "Strategy: Broaden search terms significantly, use more synonyms"
        elif success_rate < 0.6:
            return "Strategy: Adjust query expansion, focus on domain-specific terms"
        elif avg_results < 3:
            return "Strategy: Use more specific industry keywords, include location if relevant"
        else:
            return "Strategy: Current approach working well, maintain with minor optimizations"
    
    def process_request(self, user_input: str, user_id: str = "default") -> Dict:
        """
        Process user request with learning agent
        """
        # Track search pattern
        self.search_patterns.append(user_input)
        
        # Use adaptive search
        results = self.adaptive_search(user_input, user_id)
        
        # Analyze patterns
        patterns = self.recognize_patterns(self.search_patterns[-10:])  # Last 10 searches
        
        # Optimize strategy
        performance_data = {
            "success_rate": len(results) / 10 if results else 0,
            "avg_results": len(results)
        }
        strategy = self.optimize_strategy(performance_data)
        
        return {
            "results": results,
            "patterns_recognized": patterns,
            "personalized": user_id in self.learned_preferences,
            "learned_preferences": self.learned_preferences.get(user_id, {}),
            "optimization_strategy": strategy,
            "agent_type": "learning_recruiting_agent",
            "explanation": f"I used your learned preferences and recognized search patterns to find these {len(results)} professionals. {strategy}"
        }
    


