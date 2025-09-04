#!/usr/bin/env python3
"""
Comprehensive LLM Evaluation Script using DeepEval
Supports all available metrics and generates detailed CSV analysis
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import asyncio
import logging
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

# DeepEval imports
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ConversationalTestCase, MLLMTestCase, Turn
from deepeval.metrics import (
    # Custom Metrics
    GEval,
    # RAG Metrics
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    # Agentic Metrics
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    # Multi-Turn Metrics
    RoleAdherenceMetric,
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric,
    # Safety Metrics
    BiasMetric,
    PIILeakageMetric,
    PromptAlignmentMetric,
    # Others
    BaseMetric
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveLLMEvaluator:
    """
    Comprehensive LLM evaluator using all available DeepEval metrics
    """
    
    def __init__(self, model_name: str = "gpt-4", threshold: float = 0.7):
        self.model_name = model_name
        self.threshold = threshold
        self.results = []
        
        # Initialize all available metrics
        self.metrics = self._initialize_all_metrics()
        
    def _initialize_all_metrics(self) -> Dict[str, BaseMetric]:
        """Initialize all available DeepEval metrics"""
        metrics = {}
        
        try:
            # Custom Metrics
            metrics['g_eval_correctness'] = GEval(
                name="Correctness",
                criteria="Is the response factually correct and accurate?",
                evaluation_params=["input", "actual_output"],
                model=self.model_name,
                threshold=self.threshold
            )
            
            metrics['g_eval_relevance'] = GEval(
                name="Relevance", 
                criteria="Is the response relevant to the input question?",
                evaluation_params=["input", "actual_output"],
                model=self.model_name,
                threshold=self.threshold
            )
            
            metrics['g_eval_coherence'] = GEval(
                name="Coherence",
                criteria="Is the response coherent and well-structured?",
                evaluation_params=["actual_output"],
                model=self.model_name,
                threshold=self.threshold
            )
            
            metrics['g_eval_helpfulness'] = GEval(
                name="Helpfulness",
                criteria="Is the response helpful and actionable?",
                evaluation_params=["input", "actual_output"],
                model=self.model_name,
                threshold=self.threshold
            )
            
        except Exception as e:
            logger.warning(f"Could not initialize G-Eval metrics: {e}")
        
        try:
            # RAG Metrics
            metrics['answer_relevancy'] = AnswerRelevancyMetric(
                threshold=self.threshold,
                model=self.model_name,
                include_reason=True
            )
            
            metrics['faithfulness'] = FaithfulnessMetric(
                threshold=self.threshold,
                model=self.model_name,
                include_reason=True
            )
            
            metrics['contextual_relevancy'] = ContextualRelevancyMetric(
                threshold=self.threshold,
                model=self.model_name,
                include_reason=True
            )
            
            metrics['contextual_precision'] = ContextualPrecisionMetric(
                threshold=self.threshold,
                model=self.model_name,
                include_reason=True
            )
            
        except Exception as e:
            logger.warning(f"Could not initialize RAG metrics: {e}")
        
        try:
            # Safety Metrics
            metrics['bias'] = BiasMetric(
                threshold=self.threshold,
                model=self.model_name,
                include_reason=True
            )
            
            metrics['pii_leakage'] = PIILeakageMetric(
                threshold=self.threshold,
                model=self.model_name,
                include_reason=True
            )
            
        except Exception as e:
            logger.warning(f"Could not initialize Safety metrics: {e}")
        
        try:
            # Agentic Metrics (if applicable)
            metrics['task_completion'] = TaskCompletionMetric(
                threshold=self.threshold,
                model=self.model_name,
                include_reason=True
            )
            
        except Exception as e:
            logger.warning(f"Could not initialize Agentic metrics: {e}")
        
        # Custom metrics for specific analysis
        metrics.update(self._create_custom_metrics())
        
        logger.info(f"Initialized {len(metrics)} metrics: {list(metrics.keys())}")
        return metrics
    
    def _create_custom_metrics(self) -> Dict[str, BaseMetric]:
        """Create custom metrics for comprehensive analysis"""
        
        class LengthAnalysisMetric(BaseMetric):
            """Analyzes response length characteristics"""
            def __init__(self):
                super().__init__()
                self.name = "Length Analysis"
            
            def measure(self, test_case: LLMTestCase) -> float:
                response_length = len(test_case.actual_output.split())
                input_length = len(test_case.input.split())
                
                # Score based on reasonable response length (20-200 words typically good)
                if 20 <= response_length <= 200:
                    self.score = 1.0
                elif response_length < 20:
                    self.score = response_length / 20.0
                else:
                    self.score = max(0.3, 200 / response_length)
                
                self.reason = f"Response: {response_length} words, Input: {input_length} words"
                self.success = self.score >= 0.5
                return self.score
                
            def is_successful(self) -> bool:
                return self.success
        
        class SentimentAnalysisMetric(BaseMetric):
            """Analyzes sentiment appropriateness"""
            def __init__(self):
                super().__init__()
                self.name = "Sentiment Analysis"
            
            def measure(self, test_case: LLMTestCase) -> float:
                # Simple sentiment analysis based on keywords
                positive_words = ['good', 'great', 'excellent', 'helpful', 'useful', 'amazing']
                negative_words = ['bad', 'terrible', 'awful', 'useless', 'horrible']
                
                response_lower = test_case.actual_output.lower()
                pos_count = sum(1 for word in positive_words if word in response_lower)
                neg_count = sum(1 for word in negative_words if word in response_lower)
                
                if pos_count + neg_count == 0:
                    self.score = 0.5  # Neutral
                else:
                    self.score = pos_count / (pos_count + neg_count)
                
                self.reason = f"Positive indicators: {pos_count}, Negative indicators: {neg_count}"
                self.success = self.score >= 0.3
                return self.score
                
            def is_successful(self) -> bool:
                return self.success
        
        class ComplexityMetric(BaseMetric):
            """Analyzes response complexity and readability"""
            def __init__(self):
                super().__init__()
                self.name = "Complexity Analysis"
            
            def measure(self, test_case: LLMTestCase) -> float:
                response = test_case.actual_output
                
                # Calculate various complexity metrics
                sentences = response.split('.')
                words = response.split()
                avg_sentence_length = len(words) / max(len(sentences), 1)
                avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
                
                # Score based on readability (optimal: 15-20 words/sentence, 4-6 chars/word)
                sentence_score = 1.0 if 10 <= avg_sentence_length <= 25 else 0.5
                word_score = 1.0 if 3 <= avg_word_length <= 7 else 0.5
                
                self.score = (sentence_score + word_score) / 2
                self.reason = f"Avg sentence length: {avg_sentence_length:.1f}, Avg word length: {avg_word_length:.1f}"
                self.success = self.score >= 0.5
                return self.score
                
            def is_successful(self) -> bool:
                return self.success
        
        return {
            'length_analysis': LengthAnalysisMetric(),
            'sentiment_analysis': SentimentAnalysisMetric(),
            'complexity_analysis': ComplexityMetric()
        }
    
    def load_conversation_data(self, json_file_path: str) -> List[Dict]:
        """Load conversation data from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                conversations = data
            elif isinstance(data, dict):
                if 'conversations' in data:
                    conversations = data['conversations']
                elif 'data' in data:
                    conversations = data['data']
                else:
                    conversations = [data]
            else:
                raise ValueError("Unsupported JSON structure")
            
            logger.info(f"Loaded {len(conversations)} conversations from {json_file_path}")
            return conversations
            
        except Exception as e:
            logger.error(f"Error loading conversation data: {e}")
            raise
    
    def create_test_cases(self, conversations: List[Dict]) -> List[LLMTestCase]:
        """Convert conversation data to DeepEval test cases"""
        test_cases = []
        
        for i, conv in enumerate(conversations):
            try:
                # Extract conversation components based on common patterns
                input_text = self._extract_field(conv, ['input', 'question', 'prompt', 'user_message', 'human'])
                output_text = self._extract_field(conv, ['output', 'response', 'answer', 'assistant_message', 'ai', 'assistant'])
                expected_output = self._extract_field(conv, ['expected_output', 'expected', 'ground_truth', 'reference'])
                context = self._extract_field(conv, ['context', 'retrieval_context', 'background', 'documents'])
                
                if not input_text or not output_text:
                    logger.warning(f"Skipping conversation {i}: missing input or output")
                    continue
                
                # Create test case
                test_case = LLMTestCase(
                    input=input_text,
                    actual_output=output_text,
                    expected_output=expected_output,
                    retrieval_context=context if context else None,
                    additional_metadata={
                        'conversation_id': i,
                        'original_data': conv
                    }
                )
                
                test_cases.append(test_case)
                
            except Exception as e:
                logger.warning(f"Error processing conversation {i}: {e}")
                continue
        
        logger.info(f"Created {len(test_cases)} test cases")
        return test_cases
    
    def _extract_field(self, data: Dict, field_names: List[str]) -> Optional[str]:
        """Extract field from data using multiple possible field names"""
        for field_name in field_names:
            if field_name in data:
                value = data[field_name]
                if isinstance(value, str):
                    return value
                elif isinstance(value, list) and value:
                    return ' '.join(str(item) for item in value)
                elif value is not None:
                    return str(value)
        return None
    
    def evaluate_single_test_case(self, test_case: LLMTestCase, metrics_to_use: List[str] = None) -> Dict[str, Any]:
        """Evaluate a single test case with all applicable metrics"""
        results = {
            'conversation_id': test_case.additional_metadata.get('conversation_id', 'unknown'),
            'input': test_case.input,
            'actual_output': test_case.actual_output,
            'expected_output': test_case.expected_output,
            'has_context': test_case.retrieval_context is not None,
            'input_length': len(test_case.input.split()),
            'output_length': len(test_case.actual_output.split()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Use specified metrics or all available metrics
        metrics_to_evaluate = metrics_to_use or list(self.metrics.keys())
        
        for metric_name in metrics_to_evaluate:
            if metric_name not in self.metrics:
                logger.warning(f"Metric '{metric_name}' not available")
                continue
                
            try:
                metric = self.metrics[metric_name]
                
                # Check if metric requires context and test case has context
                if metric_name in ['faithfulness', 'contextual_relevancy', 'contextual_precision'] and not test_case.retrieval_context:
                    results[f'{metric_name}_score'] = np.nan
                    results[f'{metric_name}_reason'] = "No context provided"
                    results[f'{metric_name}_success'] = False
                    continue
                
                # Evaluate metric
                metric.measure(test_case)
                
                results[f'{metric_name}_score'] = getattr(metric, 'score', np.nan)
                results[f'{metric_name}_reason'] = getattr(metric, 'reason', '')
                results[f'{metric_name}_success'] = getattr(metric, 'success', False)
                
            except Exception as e:
                logger.warning(f"Error evaluating {metric_name}: {e}")
                results[f'{metric_name}_score'] = np.nan
                results[f'{metric_name}_reason'] = f"Error: {str(e)}"
                results[f'{metric_name}_success'] = False
        
        return results
    
    def evaluate_all_conversations(self, conversations: List[Dict], 
                                 metrics_to_use: List[str] = None,
                                 batch_size: int = 10) -> List[Dict[str, Any]]:
        """Evaluate all conversations with comprehensive metrics"""
        test_cases = self.create_test_cases(conversations)
        
        if not test_cases:
            logger.error("No valid test cases created")
            return []
        
        all_results = []
        
        # Process in batches to manage memory and API rate limits
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(test_cases)-1)//batch_size + 1}")
            
            for test_case in batch:
                try:
                    result = self.evaluate_single_test_case(test_case, metrics_to_use)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating test case: {e}")
                    continue
        
        return all_results
    
    def generate_comprehensive_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis of results"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        # Get all metric columns
        metric_columns = [col for col in df.columns if col.endswith('_score')]
        metric_names = [col.replace('_score', '') for col in metric_columns]
        
        analysis = {
            'summary': {
                'total_conversations': len(df),
                'metrics_evaluated': len(metric_names),
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'overall_scores': {},
            'metric_correlations': {},
            'success_rates': {},
            'score_distributions': {},
            'outlier_analysis': {},
            'quality_categories': {}
        }
        
        # Overall scores analysis
        for metric in metric_names:
            score_col = f'{metric}_score'
            if score_col in df.columns:
                scores = df[score_col].dropna()
                if len(scores) > 0:
                    analysis['overall_scores'][metric] = {
                        'mean': float(scores.mean()),
                        'median': float(scores.median()),
                        'std': float(scores.std()),
                        'min': float(scores.min()),
                        'max': float(scores.max()),
                        'q25': float(scores.quantile(0.25)),
                        'q75': float(scores.quantile(0.75)),
                        'count': int(len(scores))
                    }
        
        # Success rates
        for metric in metric_names:
            success_col = f'{metric}_success'
            if success_col in df.columns:
                success_rate = df[success_col].mean()
                analysis['success_rates'][metric] = float(success_rate) if not pd.isna(success_rate) else 0.0
        
        # Score distributions (histogram data)
        for metric in metric_names:
            score_col = f'{metric}_score'
            if score_col in df.columns:
                scores = df[score_col].dropna()
                if len(scores) > 0:
                    hist, bins = np.histogram(scores, bins=10, range=(0, 1))
                    analysis['score_distributions'][metric] = {
                        'histogram': hist.tolist(),
                        'bins': bins.tolist()
                    }
        
        # Quality categories
        score_columns = [col for col in df.columns if col.endswith('_score')]
        if score_columns:
            df['average_score'] = df[score_columns].mean(axis=1, skipna=True)
            
            # Categorize quality
            conditions = [
                df['average_score'] >= 0.8,
                df['average_score'] >= 0.6,
                df['average_score'] >= 0.4,
                df['average_score'] >= 0.2
            ]
            choices = ['Excellent', 'Good', 'Fair', 'Poor']
            df['quality_category'] = np.select(conditions, choices, default='Very Poor')
            
            analysis['quality_categories'] = df['quality_category'].value_counts().to_dict()
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            analysis['metric_correlations'] = correlation_matrix.to_dict()
        
        # Outlier analysis
        for metric in metric_names:
            score_col = f'{metric}_score'
            if score_col in df.columns:
                scores = df[score_col].dropna()
                if len(scores) > 0:
                    Q1 = scores.quantile(0.25)
                    Q3 = scores.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = scores[(scores < lower_bound) | (scores > upper_bound)]
                    analysis['outlier_analysis'][metric] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': (len(outliers) / len(scores)) * 100,
                        'outlier_values': outliers.tolist()
                    }
        
        return analysis
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], 
                           analysis: Dict[str, Any], 
                           output_file: str = None) -> str:
        """Save evaluation results and analysis to CSV files"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"llm_evaluation_results_{timestamp}.csv"
        
        # Save detailed results
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        # Save summary analysis
        summary_file = output_file.replace('.csv', '_summary.csv')
        
        # Create summary DataFrame
        summary_data = []
        
        # Overall scores
        if 'overall_scores' in analysis:
            for metric, stats in analysis['overall_scores'].items():
                summary_data.append({
                    'Category': 'Overall Score',
                    'Metric': metric,
                    'Statistic': 'Mean',
                    'Value': stats['mean']
                })
                summary_data.append({
                    'Category': 'Overall Score',
                    'Metric': metric,
                    'Statistic': 'Median',
                    'Value': stats['median']
                })
                summary_data.append({
                    'Category': 'Overall Score', 
                    'Metric': metric,
                    'Statistic': 'Std Dev',
                    'Value': stats['std']
                })
        
        # Success rates
        if 'success_rates' in analysis:
            for metric, rate in analysis['success_rates'].items():
                summary_data.append({
                    'Category': 'Success Rate',
                    'Metric': metric,
                    'Statistic': 'Success Rate',
                    'Value': rate
                })
        
        # Quality categories
        if 'quality_categories' in analysis:
            for category, count in analysis['quality_categories'].items():
                summary_data.append({
                    'Category': 'Quality Distribution',
                    'Metric': 'Count',
                    'Statistic': category,
                    'Value': count
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed analysis as JSON
        analysis_file = output_file.replace('.csv', '_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to:")
        logger.info(f"  - Detailed results: {output_file}")
        logger.info(f"  - Summary: {summary_file}")
        logger.info(f"  - Analysis: {analysis_file}")
        
        return output_file

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive LLM Evaluation using DeepEval")
    parser.add_argument("json_file", help="Path to JSON file containing conversation data")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--model", "-m", default="gpt-4", help="Model to use for evaluation")
    parser.add_argument("--threshold", "-t", type=float, default=0.7, help="Threshold for metrics")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--metrics", nargs="+", help="Specific metrics to use (default: all)")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ComprehensiveLLMEvaluator(
        model_name=args.model,
        threshold=args.threshold
    )
    
    try:
        # Load conversation data
        conversations = evaluator.load_conversation_data(args.json_file)
        
        # Evaluate all conversations
        logger.info("Starting comprehensive evaluation...")
        results = evaluator.evaluate_all_conversations(
            conversations,
            metrics_to_use=args.metrics,
            batch_size=args.batch_size
        )
        
        if not results:
            logger.error("No results generated")
            return
        
        # Generate analysis
        logger.info("Generating comprehensive analysis...")
        analysis = evaluator.generate_comprehensive_analysis(results)
        
        # Save results
        output_file = evaluator.save_results_to_csv(results, analysis, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total conversations evaluated: {len(results)}")
        print(f"Metrics used: {len(evaluator.metrics)}")
        print(f"Results saved to: {output_file}")
        
        if 'overall_scores' in analysis:
            print("\nTop performing metrics:")
            scores = [(metric, stats['mean']) for metric, stats in analysis['overall_scores'].items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            for metric, score in scores[:5]:
                print(f"  {metric}: {score:.3f}")
        
        if 'quality_categories' in analysis:
            print("\nQuality distribution:")
            for category, count in analysis['quality_categories'].items():
                print(f"  {category}: {count}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()