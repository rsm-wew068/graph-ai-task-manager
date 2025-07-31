#!/usr/bin/env python3
"""
RAGAS Evaluator - Replace confidence scores with comprehensive RAG evaluation
Provides multiple metrics for evaluating RAG system performance
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Union
import numpy as np

# Handle dotenv import gracefully
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# RAGAS imports with error handling
try:
    from ragas.metrics import Faithfulness, ContextRecall
    from ragas.dataset_schema import SingleTurnSample
    from ragas.run_config import RunConfig
    from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
    
    # LangChain wrappers for RAGAS
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    RAGAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAGAS not available: {e}")
    RAGAS_AVAILABLE = False

# LangChain imports for LLM and embeddings
try:
    from langchain_openai.chat_models import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain OpenAI not available: {e}")
    LANGCHAIN_AVAILABLE = False


class RAGASEvaluator:
    """
    Comprehensive RAG evaluation using RAGAS metrics.
    Now only uses Faithfulness and ContextRecall metrics.
    """
    
    def __init__(self, 
                 openai_model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize RAGAS evaluator with only Faithfulness and ContextRecall.
        """
        self.openai_model = openai_model
        self.embedding_model = embedding_model
        
        if not RAGAS_AVAILABLE or not LANGCHAIN_AVAILABLE:
            raise ImportError("RAGAS and LangChain OpenAI are required. Install with: pip install ragas langchain-openai")
        
        # Verify OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.llm = None
        self.embeddings = None
        self.metrics = []
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM, embeddings, and metrics."""
        try:
            # Initialize LangChain components
            self.llm = LangchainLLMWrapper(ChatOpenAI(
                model=self.openai_model,
                temperature=0.1  # Low temperature for consistent evaluation
            ))
            
            self.embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
                model=self.embedding_model
            ))
            
            # Only Faithfulness and ContextRecall
            self.metrics = [
                Faithfulness(),
                ContextRecall(),
            ]
            
            # Initialize each metric with LLM and embeddings
            for metric in self.metrics:
                if isinstance(metric, MetricWithLLM):
                    metric.llm = self.llm
                if isinstance(metric, MetricWithEmbeddings):
                    metric.embeddings = self.embeddings
                
                run_config = RunConfig()
                metric.init(run_config)
            
            logger.info(f"\u2705 RAGAS evaluator initialized with Faithfulness and ContextRecall metrics")
            
        except Exception as e:
            logger.error(f"\u274c Failed to initialize RAGAS evaluator: {e}")
            raise
    
    async def evaluate_single(self, 
                            query: str, 
                            response: str, 
                            contexts: List[str],
                            ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate a single query-response pair using Faithfulness and ContextRecall.
        ContextRecall requires ground_truth.
        """
        if not query or not response or not contexts:
            logger.warning("Empty query, response, or contexts provided")
            return {}
        if ground_truth is None:
            logger.warning("ContextRecall requires ground_truth. None provided.")
            return {}
        
        try:
            # Create RAGAS sample
            sample_data = {
                "user_input": query,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": ground_truth
            }
            sample = SingleTurnSample(**sample_data)
            
            # Evaluate with each metric
            scores = {}
            for metric in self.metrics:
                try:
                    logger.debug(f"Calculating {metric.name}")
                    score = await metric.single_turn_ascore(sample)
                    
                    # Handle numpy types and convert to float
                    if hasattr(score, 'item'):
                        score = float(score.item())
                    elif isinstance(score, (np.floating, np.integer)):
                        score = float(score)
                    
                    scores[metric.name] = score
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric.name}: {e}")
                    scores[metric.name] = 0.0
            
            return scores
            
        except Exception as e:
            logger.error(f"\u274c Error in RAGAS evaluation: {e}")
            return {}
    
    def evaluate_batch(self, 
                      queries: List[str],
                      responses: List[str], 
                      contexts_list: List[List[str]],
                      ground_truths: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """
        Evaluate multiple query-response pairs (synchronous batch processing).
        Faithfulness and ContextRecall only. ContextRecall requires ground_truths.
        """
        if len(queries) != len(responses) or len(queries) != len(contexts_list):
            raise ValueError("Queries, responses, and contexts must have the same length")
        if ground_truths is None or len(ground_truths) != len(queries):
            raise ValueError("ContextRecall requires ground_truths for each query.")
        
        # Run async evaluation in batch
        async def run_batch():
            tasks = []
            for i in range(len(queries)):
                gt = ground_truths[i]
                task = self.evaluate_single(
                    queries[i], responses[i], contexts_list[i], gt
                )
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
        
        try:
            # Run the async batch
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_batch())
                    results = future.result()
            else:
                results = asyncio.run(run_batch())
            
            # Aggregate results
            aggregated = {}
            for result in results:
                for metric_name, score in result.items():
                    if metric_name not in aggregated:
                        aggregated[metric_name] = []
                    aggregated[metric_name].append(score)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"\u274c Error in batch evaluation: {e}")
            return {}
    
    def get_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate overall RAGAS score from Faithfulness and ContextRecall.
        """
        if not scores:
            return 0.0
        
        # Define weights for the two metrics
        weights = {
            'faithfulness': 0.7,      # Faithfulness is most important
            'context_recall': 0.3     # ContextRecall is secondary
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, score in scores.items():
            weight = weights.get(metric_name, 0.1)  # Default weight for unknown metrics
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def format_evaluation_summary(self, scores: Dict[str, float]) -> str:
        """
        Format evaluation results into a human-readable summary for Faithfulness and ContextRecall.
        """
        if not scores:
            return "\u274c No evaluation scores available"
        
        overall = self.get_overall_score(scores)
        
        # Determine overall quality level
        if overall >= 0.8:
            quality_emoji = "\U0001F7E2"
            quality_text = "Excellent"
        elif overall >= 0.6:
            quality_emoji = "\U0001F7E1"
            quality_text = "Good"
        elif overall >= 0.4:
            quality_emoji = "\U0001F7E0"
            quality_text = "Fair"
        else:
            quality_emoji = "\U0001F534"
            quality_text = "Poor"
        
        summary_parts = [
            f"**\U0001F4CA RAGAS Evaluation Summary**",
            f"{quality_emoji} **Overall Quality:** {quality_text} ({overall:.3f})",
            "",
            "**Individual Metrics:**"
        ]
        
        # Format only Faithfulness and ContextRecall
        metric_display_names = {
            'faithfulness': '\U0001F3AF Faithfulness (Factual Accuracy)',
            'context_recall': '\U0001F504 Context Recall',
        }
        
        for metric_name, score in scores.items():
            display_name = metric_display_names.get(metric_name, metric_name)
            score_emoji = "\u2705" if score >= 0.7 else "\u26A0\uFE0F" if score >= 0.5 else "\u274C"
            summary_parts.append(f"   {score_emoji} {display_name}: {score:.3f}")
        
        return "\n".join(summary_parts)


# Convenience functions for backward compatibility
async def evaluate_rag_response(query: str, 
                              response: str, 
                              contexts: List[str],
                              ground_truth: Optional[str] = None) -> Dict[str, float]:
    """
    Quick evaluation function for single response.
    
    Args:
        query: User question
        response: Generated response
        contexts: Retrieved context chunks
        ground_truth: Optional ground truth answer
        
    Returns:
        Dictionary of RAGAS scores
    """
    evaluator = RAGASEvaluator()
    return await evaluator.evaluate_single(query, response, contexts, ground_truth)


def get_ragas_summary(scores: Dict[str, float]) -> str:
    """
    Get formatted RAGAS evaluation summary.
    
    Args:
        scores: Dictionary of RAGAS scores
        
    Returns:
        Formatted summary string
    """
    evaluator = RAGASEvaluator()
    return evaluator.format_evaluation_summary(scores)


# Test function
async def test_ragas_evaluator():
    """Test the RAGAS evaluator with sample data."""
    evaluator = RAGASEvaluator()
    
    # Sample data
    query = "What is the capital of France?"
    response = "The capital of France is Paris. It is located in the north-central part of the country."
    contexts = [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe.", 
        "Paris is located in northern France on both banks of the Seine River."
    ]
    
    print("🧪 Testing RAGAS Evaluator...")
    scores = await evaluator.evaluate_single(query, response, contexts)
    
    print("📊 Raw Scores:", scores)
    print("\n" + evaluator.format_evaluation_summary(scores))
    
    return scores


if __name__ == "__main__":
    # Run test
    asyncio.run(test_ragas_evaluator()) 