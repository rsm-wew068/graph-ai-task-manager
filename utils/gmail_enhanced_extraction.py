"""
Enhanced Task Extraction for Gmail API
Optimized extraction pipeline for Gmail emails with better context handling
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from utils.email_parser import parse_gmail_emails
from utils.langgraph_dag import run_extraction_pipeline
from utils.embedding import embed_dataframe
from utils.gmail_smart_filters import GmailSmartFilter, create_task_focused_filter

class GmailEnhancedExtractor:
    """Enhanced task extraction for Gmail API emails"""
    
    def __init__(self, gmail_service, smart_filter: Optional[GmailSmartFilter] = None):
        self.gmail_service = gmail_service
        self.smart_filter = smart_filter or create_task_focused_filter()
        self.extraction_history = []
    
    def extract_tasks_from_emails(self, emails: List[Dict[str, Any]], 
                                max_emails: int = 50) -> List[Dict[str, Any]]:
        """
        Extract tasks from Gmail emails with enhanced processing
        
        Args:
            emails: List of Gmail email dictionaries
            max_emails: Maximum number of emails to process
            
        Returns:
            List of extracted task results
        """
        # Apply smart filtering
        filtered_emails = self.smart_filter.filter_emails(emails)
        
        # Limit to max_emails
        if len(filtered_emails) > max_emails:
            filtered_emails = filtered_emails[:max_emails]
        
        if not filtered_emails:
            return []
        
        # Convert to DataFrame format using the new email parser
        df = parse_gmail_emails(filtered_emails)
        
        # Enhance email data with additional context
        df = self._enhance_email_data(df)
        
        # Create embeddings for similarity search
        index, all_chunks = embed_dataframe(df)
        
        # Process each email through extraction pipeline
        results = []
        
        for index, row in df.iterrows():
            try:
                # Convert row to dict for processing
                email_row = row.to_dict()
                
                # Add email index for tracking
                email_row['email_index'] = str(index)
                
                # Run enhanced extraction pipeline
                extraction_result = self._run_enhanced_extraction(
                    email_row=email_row,
                    faiss_index=index,
                    all_chunks=all_chunks,
                    email_index=str(index)
                )
                
                # Format result
                result = {
                    'email_index': str(index),
                    'subject': email_row.get('Subject', ''),
                    'sender': email_row.get('From', ''),
                    'date': email_row.get('Date', ''),
                    'filter_score': email_row.get('Filter-Score', 0),
                    'extraction_result': extraction_result,
                    'status': 'processed',
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Add to extraction history
                self.extraction_history.append({
                    'email_id': email_row.get('Message-ID', ''),
                    'subject': email_row.get('Subject', ''),
                    'extraction_timestamp': datetime.now().isoformat(),
                    'status': 'processed'
                })
                
            except Exception as e:
                # Handle processing errors
                result = {
                    'email_index': str(index),
                    'subject': email_row.get('Subject', 'Unknown'),
                    'sender': email_row.get('From', 'Unknown'),
                    'date': email_row.get('Date', ''),
                    'status': 'error',
                    'message': f'Processing failed: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
        
        return results
    
    def _enhance_email_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance email data with additional context and metadata"""
        
        # Add urgency detection
        df['urgency_level'] = df.apply(self._detect_urgency, axis=1)
        
        # Add email type classification
        df['email_type'] = df.apply(self._classify_email_type, axis=1)
        
        # Add sender importance score
        df['sender_importance'] = df.apply(self._calculate_sender_importance, axis=1)
        
        # Add content complexity score
        df['content_complexity'] = df.apply(self._calculate_content_complexity, axis=1)
        
        # Add thread context (if available)
        df['thread_context'] = df.apply(self._get_thread_context, axis=1)
        
        return df
    
    def _detect_urgency(self, row) -> str:
        """Detect urgency level of email"""
        content = f"{row.get('Subject', '')} {row.get('content', '')}".lower()
        
        high_urgency = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
        medium_urgency = ['deadline', 'due today', 'due tomorrow', 'priority']
        low_urgency = ['when convenient', 'no rush', 'low priority']
        
        if any(word in content for word in high_urgency):
            return 'high'
        elif any(word in content for word in medium_urgency):
            return 'medium'
        elif any(word in content for word in low_urgency):
            return 'low'
        else:
            return 'normal'
    
    def _classify_email_type(self, row) -> str:
        """Classify email type"""
        content = f"{row.get('Subject', '')} {row.get('content', '')}".lower()
        
        if any(word in content for word in ['meeting', 'call', 'conference', 'discussion']):
            return 'meeting'
        elif any(word in content for word in ['task', 'todo', 'action item', 'deadline']):
            return 'task'
        elif any(word in content for word in ['project', 'initiative', 'campaign']):
            return 'project'
        elif any(word in content for word in ['review', 'approve', 'sign']):
            return 'approval'
        elif any(word in content for word in ['update', 'status', 'progress']):
            return 'update'
        else:
            return 'general'
    
    def _calculate_sender_importance(self, row) -> float:
        """Calculate sender importance score"""
        sender = row.get('From', '').lower()
        
        # Check for important domains
        important_domains = ['company.com', 'boss.com', 'manager.com']
        if any(domain in sender for domain in important_domains):
            return 0.9
        
        # Check for C-level or management indicators
        management_indicators = ['ceo', 'cto', 'cfo', 'vp', 'director', 'manager', 'lead']
        if any(indicator in sender for indicator in management_indicators):
            return 0.8
        
        # Check for team members
        team_indicators = ['team', 'group', 'department']
        if any(indicator in sender for indicator in team_indicators):
            return 0.6
        
        return 0.3
    
    def _calculate_content_complexity(self, row) -> float:
        """Calculate content complexity score"""
        body = row.get('content', '')
        
        # Count sentences
        sentences = len(re.split(r'[.!?]+', body))
        
        # Count words
        words = len(body.split())
        
        # Count paragraphs
        paragraphs = len([p for p in body.split('\n\n') if p.strip()])
        
        # Calculate complexity score
        complexity = (sentences * 0.1 + words * 0.01 + paragraphs * 0.2) / 10
        return min(complexity, 1.0)
    
    def _get_thread_context(self, row) -> str:
        """Get thread context if available"""
        # This would require additional Gmail API calls to get thread information
        # For now, return empty string
        return ""
    
    def _run_enhanced_extraction(self, email_row: Dict[str, Any], 
                               faiss_index, all_chunks, email_index: str) -> Dict[str, Any]:
        """Run enhanced extraction with additional context"""
        
        # Add enhanced context to the email row
        enhanced_context = self._create_enhanced_context(email_row)
        email_row['enhanced_context'] = enhanced_context
        
        # Run the standard extraction pipeline
        result = run_extraction_pipeline(
            email_row=email_row,
            faiss_index=faiss_index,
            all_chunks=all_chunks,
            email_index=email_index
        )
        
        # Enhance the result with additional metadata
        enhanced_result = self._enhance_extraction_result(result, email_row)
        
        return enhanced_result
    
    def _create_enhanced_context(self, email_row: Dict[str, Any]) -> str:
        """Create enhanced context for extraction"""
        context_parts = []
        
        # Add urgency context
        urgency = email_row.get('urgency_level', 'normal')
        if urgency != 'normal':
            context_parts.append(f"URGENCY: {urgency.upper()}")
        
        # Add email type context
        email_type = email_row.get('email_type', 'general')
        context_parts.append(f"EMAIL TYPE: {email_type.upper()}")
        
        # Add sender importance
        sender_importance = email_row.get('sender_importance', 0.3)
        if sender_importance > 0.7:
            context_parts.append("SENDER: HIGH IMPORTANCE")
        elif sender_importance > 0.5:
            context_parts.append("SENDER: MEDIUM IMPORTANCE")
        
        # Add content complexity
        complexity = email_row.get('content_complexity', 0.3)
        if complexity > 0.7:
            context_parts.append("CONTENT: COMPLEX")
        elif complexity < 0.3:
            context_parts.append("CONTENT: SIMPLE")
        
        return " | ".join(context_parts)
    
    def _enhance_extraction_result(self, result: Dict[str, Any], 
                                 email_row: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance extraction result with additional metadata"""
        
        # Add email metadata to result
        enhanced_result = result.copy()
        enhanced_result['email_metadata'] = {
            'urgency_level': email_row.get('urgency_level', 'normal'),
            'email_type': email_row.get('email_type', 'general'),
            'sender_importance': email_row.get('sender_importance', 0.3),
            'content_complexity': email_row.get('content_complexity', 0.3),
            'filter_score': email_row.get('Filter-Score', 0),
            'enhanced_context': email_row.get('enhanced_context', '')
        }
        
        # Add processing metadata
        enhanced_result['processing_metadata'] = {
            'extraction_timestamp': datetime.now().isoformat(),
            'processing_version': 'enhanced_v1.0',
            'email_source': 'gmail_api'
        }
        
        return enhanced_result
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of extraction history"""
        if not self.extraction_history:
            return {'total_processed': 0, 'success_rate': 0}
        
        total_processed = len(self.extraction_history)
        successful = len([h for h in self.extraction_history if h['status'] == 'processed'])
        success_rate = (successful / total_processed) * 100
        
        return {
            'total_processed': total_processed,
            'successful': successful,
            'success_rate': success_rate,
            'last_extraction': self.extraction_history[-1]['extraction_timestamp'] if self.extraction_history else None
        }

def create_enhanced_extractor(gmail_service) -> GmailEnhancedExtractor:
    """Create an enhanced Gmail extractor"""
    return GmailEnhancedExtractor(gmail_service) 