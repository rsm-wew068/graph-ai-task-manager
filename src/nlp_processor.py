"""
NLP Processing Module
Handles entity extraction, topic modeling, task detection, and summarization
Uses BERTopic for advanced topic modeling
"""

import re
import spacy
import nltk
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# BERTopic imports
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False


@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class ExtractedTask:
    """Represents an extracted task"""
    task_description: str
    assignee: Optional[str]
    due_date: Optional[datetime]
    priority: str = "medium"
    confidence: float = 1.0
    context: str = ""


@dataclass
class EmailInsights:
    """Complete insights extracted from an email"""
    entities: List[ExtractedEntity]
    topics: List[Tuple[str, float]]
    tasks: List[ExtractedTask]
    summary: str
    sentiment: str
    urgency_score: float


class NLPProcessor:
    """Main NLP processing class with BERTopic integration"""
    
    def __init__(self):
        self.nlp = None
        self.summarizer = None
        self.sentiment_analyzer = None
        self.bertopic_model = None
        self.sentence_model = None
        self.task_patterns = self._compile_task_patterns()
        self.date_patterns = self._compile_date_patterns()
        self.urgency_keywords = self._load_urgency_keywords()
        
    def initialize_models(self):
        """Initialize NLP models (call this after installing dependencies)"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize transformers pipelines
            self.summarizer = pipeline("summarization", 
                                     model="facebook/bart-large-cnn",
                                     max_length=150, min_length=30)
            
            self.sentiment_analyzer = pipeline("sentiment-analysis",
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            # Initialize BERTopic if available
            if BERTOPIC_AVAILABLE:
                print("Initializing BERTopic model...")
                self._initialize_bertopic()
            else:
                print("BERTopic not available, falling back to LDA")
            
            print("NLP models initialized successfully")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            print("Please install required models with:")
            print("python -m spacy download en_core_web_sm")
    
    def _initialize_bertopic(self):
        """Initialize BERTopic model with optimized settings"""
        try:
            # Use a lightweight sentence transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Configure UMAP for dimensionality reduction
            umap_model = UMAP(
                n_neighbors=15, 
                n_components=5, 
                min_dist=0.0, 
                metric='cosine',
                random_state=42
            )
            
            # Configure HDBSCAN for clustering
            hdbscan_model = HDBSCAN(
                min_cluster_size=10,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            # Initialize BERTopic
            self.bertopic_model = BERTopic(
                embedding_model=self.sentence_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=TfidfVectorizer(
                    stop_words="english",
                    max_features=1000,
                    ngram_range=(1, 2)
                ),
                top_k_words=10,
                language="english",
                calculate_probabilities=True,
                verbose=True
            )
            
            print("✓ BERTopic model initialized successfully")
            
        except Exception as e:
            print(f"BERTopic initialization failed: {e}")
            self.bertopic_model = None
    
    def _compile_task_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for task detection"""
        patterns = [
            # Action verbs with objects
            r'(?:please|could you|can you|need to|should|must|have to)\s+([^.!?]+)',
            r'(?:action item|todo|to do|task):\s*([^.!?]+)',
            r'(?:by|due|deadline|complete by|finish by)\s+([^.!?]+)',
            r'(?:schedule|arrange|organize|coordinate)\s+([^.!?]+)',
            r'(?:review|check|verify|confirm|validate)\s+([^.!?]+)',
            r'(?:send|provide|deliver|submit|prepare)\s+([^.!?]+)',
            r'(?:call|contact|reach out to|follow up with)\s+([^.!?]+)',
            r'(?:update|revise|modify|change)\s+([^.!?]+)',
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _compile_date_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for date extraction"""
        patterns = [
            r'(?:by|due|deadline|before)\s+(\w+day,?\s+\w+\s+\d{1,2},?\s+\d{4})',
            r'(?:by|due|deadline|before)\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?:by|due|deadline|before)\s+(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(?:by|due|deadline|before)\s+(\d{1,2}-\d{1,2}-\d{2,4})',
            r'(?:next|this)\s+(\w+day)',
            r'(?:tomorrow|today)',
            r'(?:end of|eod|close of business)',
            r'(?:asap|urgent|immediately)',
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _load_urgency_keywords(self) -> Dict[str, float]:
        """Load keywords that indicate urgency"""
        return {
            'urgent': 0.9,
            'asap': 0.9,
            'immediately': 0.9,
            'critical': 0.8,
            'important': 0.7,
            'priority': 0.7,
            'deadline': 0.6,
            'due': 0.5,
            'soon': 0.4,
            'quick': 0.4,
            'fast': 0.4,
        }
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract named entities from text"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Focus on relevant entity types
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'MONEY', 'PRODUCT']:
                entities.append(ExtractedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0  # spaCy doesn't provide confidence scores
                ))
        
        return entities
    
    def extract_topics(self, texts: List[str], n_topics: int = 5) -> List[Tuple[str, float]]:
        """Extract topics using BERTopic (preferred) or LDA (fallback)"""
        if not texts:
            return []
        
        # Filter out very short texts
        filtered_texts = [text for text in texts if len(text.split()) > 10]
        if len(filtered_texts) < 5:  # Need minimum texts for topic modeling
            return []
        
        try:
            if self.bertopic_model and BERTOPIC_AVAILABLE:
                return self._extract_topics_bertopic(filtered_texts, n_topics)
            else:
                return self._extract_topics_lda(filtered_texts, n_topics)
                
        except Exception as e:
            print(f"Error in topic extraction: {e}")
            return []
    
    def _extract_topics_bertopic(self, texts: List[str], n_topics: int) -> List[Tuple[str, float]]:
        """Extract topics using BERTopic"""
        try:
            print(f"Extracting topics using BERTopic from {len(texts)} texts...")

            # Adjust UMAP parameters to avoid "k must be less than or equal to the number of training points" errors
            try:
                n = len(texts)
                if hasattr(self.bertopic_model, 'umap_model') and self.bertopic_model.umap_model is not None:
                    try:
                        from umap import UMAP
                        orig_umap = self.bertopic_model.umap_model
                        n_neighbors = min(getattr(orig_umap, 'n_neighbors', 15), max(2, n - 1))
                        self.bertopic_model.umap_model = UMAP(
                            n_neighbors=n_neighbors,
                            n_components=getattr(orig_umap, 'n_components', 5),
                            min_dist=getattr(orig_umap, 'min_dist', 0.0),
                            metric=getattr(orig_umap, 'metric', 'cosine'),
                            random_state=getattr(orig_umap, 'random_state', 42)
                        )
                    except Exception:
                        # If anything goes wrong adjusting UMAP, continue and let BERTopic handle or fallback
                        pass
            except Exception:
                pass

            # Fit BERTopic model
            topics, probabilities = self.bertopic_model.fit_transform(texts)
            
            # Get topic information
            topic_info = self.bertopic_model.get_topic_info()
            
            # Extract top topics (excluding outlier topic -1)
            valid_topics = topic_info[topic_info.Topic != -1].head(n_topics)
            
            result_topics = []
            for _, row in valid_topics.iterrows():
                topic_id = row['Topic']
                topic_words = self.bertopic_model.get_topic(topic_id)
                
                if topic_words:
                    # Create topic name from top words
                    top_words = [word for word, _ in topic_words[:3]]
                    topic_name = " ".join(top_words)
                    
                    # Use document count as weight
                    weight = float(row['Count']) / len(texts)
                    
                    result_topics.append((topic_name, weight))
            
            print(f"✓ BERTopic extracted {len(result_topics)} topics")
            return result_topics
            
        except Exception as e:
            print(f"BERTopic extraction failed: {e}, falling back to LDA")
            return self._extract_topics_lda(texts, n_topics)
    
    def _extract_topics_lda(self, texts: List[str], n_topics: int) -> List[Tuple[str, float]]:
        """Extract topics using LDA (fallback method)"""
        try:
            print(f"Extracting topics using LDA from {len(texts)} texts...")
            
            # Vectorize texts
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-5:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_name = " ".join(top_words[:3])
                weight = float(np.sum(topic))
                topics.append((topic_name, weight))
            
            result = sorted(topics, key=lambda x: x[1], reverse=True)
            print(f"✓ LDA extracted {len(result)} topics")
            return result
            
        except Exception as e:
            print(f"Error in LDA topic extraction: {e}")
            return []
    
    def get_topic_hierarchy(self, texts: List[str]) -> Dict:
        """Get hierarchical topic structure (BERTopic only)"""
        if not self.bertopic_model or not BERTOPIC_AVAILABLE:
            return {"message": "BERTopic not available for hierarchical topics"}
        
        try:
            if not hasattr(self.bertopic_model, 'topics_'):
                # Need to fit first
                self.bertopic_model.fit_transform(texts)
            
            # Get hierarchical topics
            hierarchical_topics = self.bertopic_model.hierarchical_topics(texts)
            
            return {
                "hierarchical_topics": hierarchical_topics.to_dict('records'),
                "topic_tree": "Use bertopic_model.visualize_hierarchy() for visualization"
            }
            
        except Exception as e:
            print(f"Error getting topic hierarchy: {e}")
            return {"error": str(e)}
    
    def get_topic_evolution(self, texts: List[str], timestamps: List[str]) -> Dict:
        """Analyze how topics evolve over time (BERTopic only)"""
        if not self.bertopic_model or not BERTOPIC_AVAILABLE:
            return {"message": "BERTopic not available for topic evolution"}
        
        try:
            if not hasattr(self.bertopic_model, 'topics_'):
                self.bertopic_model.fit_transform(texts)
            
            # Convert timestamps to datetime
            from datetime import datetime
            dt_timestamps = []
            for ts in timestamps:
                try:
                    if isinstance(ts, str):
                        dt_timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                    else:
                        dt_timestamps.append(ts)
                except:
                    dt_timestamps.append(datetime.now())
            
            # Get topics over time
            topics_over_time = self.bertopic_model.topics_over_time(
                texts, dt_timestamps, nr_bins=10
            )
            
            return {
                "topics_over_time": topics_over_time.to_dict('records'),
                "visualization": "Use bertopic_model.visualize_topics_over_time() for plots"
            }
            
        except Exception as e:
            print(f"Error in topic evolution analysis: {e}")
            return {"error": str(e)}
    
    def extract_tasks(self, text: str, from_email: str = "") -> List[ExtractedTask]:
        """Extract tasks from email text"""
        tasks = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check for task patterns
            for pattern in self.task_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    task_desc = match.strip()
                    if len(task_desc) > 5:  # Minimum task description length
                        
                        # Extract assignee (look for person names or email addresses)
                        assignee = self._extract_assignee(sentence, from_email)
                        
                        # Extract due date
                        due_date = self._extract_due_date(sentence)
                        
                        # Determine priority based on keywords
                        priority = self._determine_priority(sentence)
                        
                        tasks.append(ExtractedTask(
                            task_description=task_desc,
                            assignee=assignee,
                            due_date=due_date,
                            priority=priority,
                            confidence=0.7,
                            context=sentence
                        ))
        
        return tasks
    
    def _extract_assignee(self, text: str, from_email: str) -> Optional[str]:
        """Extract task assignee from text"""
        # Look for "you" (referring to recipient)
        if re.search(r'\byou\b', text, re.IGNORECASE):
            return "recipient"
        
        # Look for specific names or email addresses
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    return ent.text
        
        # Default to sender if no specific assignee found
        return from_email if from_email else None
    
    def _extract_due_date(self, text: str) -> Optional[datetime]:
        """Extract due date from text"""
        for pattern in self.date_patterns:
            match = pattern.search(text)
            if match:
                date_str = match.group(1) if match.groups() else match.group(0)
                return self._parse_relative_date(date_str)
        
        return None
    
    def _parse_relative_date(self, date_str: str) -> Optional[datetime]:
        """Parse relative date expressions"""
        date_str = date_str.lower().strip()
        now = datetime.now()
        
        if 'tomorrow' in date_str:
            return now + timedelta(days=1)
        elif 'today' in date_str:
            return now
        elif 'next week' in date_str:
            return now + timedelta(weeks=1)
        elif 'monday' in date_str:
            days_ahead = 0 - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return now + timedelta(days=days_ahead)
        # Add more relative date parsing as needed
        
        return None
    
    def _determine_priority(self, text: str) -> str:
        """Determine task priority based on keywords"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['urgent', 'asap', 'critical', 'immediately']):
            return "high"
        elif any(word in text_lower for word in ['important', 'priority', 'soon']):
            return "medium"
        else:
            return "low"
    
    def calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on keywords and patterns"""
        text_lower = text.lower()
        score = 0.0
        
        for keyword, weight in self.urgency_keywords.items():
            if keyword in text_lower:
                score += weight
        
        # Normalize score
        return min(score, 1.0)
    
    def summarize_text(self, text: str) -> str:
        """Generate summary of text"""
        if not self.summarizer or len(text) < 100:
            # Fallback: return first sentence or truncated text
            sentences = re.split(r'[.!?]+', text)
            return sentences[0][:200] + "..." if sentences else text[:200] + "..."
        
        try:
            # Clean text for summarization
            clean_text = re.sub(r'\s+', ' ', text).strip()
            if len(clean_text) > 1024:  # BART has token limits
                clean_text = clean_text[:1024]
            
            summary = self.summarizer(clean_text)
            return summary[0]['summary_text']
            
        except Exception as e:
            print(f"Error in summarization: {e}")
            # Fallback
            sentences = re.split(r'[.!?]+', text)
            return sentences[0][:200] + "..." if sentences else text[:200] + "..."
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text"""
        if not self.sentiment_analyzer:
            return "neutral"
        
        try:
            result = self.sentiment_analyzer(text[:512])  # Limit text length
            return result[0]['label'].lower()
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return "neutral"
    
    def process_email(self, email_text: str, subject: str = "", from_email: str = "") -> EmailInsights:
        """Process a complete email and extract all insights"""
        full_text = f"{subject} {email_text}".strip()
        
        # Extract components
        entities = self.extract_entities(full_text)
        tasks = self.extract_tasks(full_text, from_email)
        summary = self.summarize_text(email_text)
        sentiment = self.analyze_sentiment(full_text)
        urgency_score = self.calculate_urgency_score(full_text)
        
        # Topics will be extracted in batch processing
        topics = []
        
        return EmailInsights(
            entities=entities,
            topics=topics,
            tasks=tasks,
            summary=summary,
            sentiment=sentiment,
            urgency_score=urgency_score
        )