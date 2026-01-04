"""
Task Manager Module
Handles task extraction, scheduling, and productivity integration
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from nlp_processor import ExtractedTask


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class StructuredTask:
    """Enhanced task structure for productivity tools"""
    id: str
    title: str
    description: str
    assignee: Optional[str]
    creator: str
    due_date: Optional[datetime]
    created_date: datetime
    priority: TaskPriority
    status: TaskStatus
    tags: List[str]
    source_email_id: str
    confidence: float
    estimated_duration: Optional[int] = None  # minutes
    dependencies: List[str] = None  # task IDs
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['due_date'] = self.due_date.isoformat() if self.due_date else None
        data['created_date'] = self.created_date.isoformat()
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data
    
    def to_calendar_event(self) -> Dict:
        """Convert to calendar event format"""
        return {
            'title': self.title,
            'description': self.description,
            'start': self.due_date.isoformat() if self.due_date else None,
            'end': (self.due_date + timedelta(hours=1)).isoformat() if self.due_date else None,
            'allDay': False,
            'tags': self.tags,
            'priority': self.priority.value
        }
    
    def to_todo_item(self) -> Dict:
        """Convert to todo list format"""
        return {
            'task': self.title,
            'description': self.description,
            'due': self.due_date.strftime('%Y-%m-%d') if self.due_date else None,
            'priority': self.priority.value,
            'assignee': self.assignee,
            'status': self.status.value,
            'tags': self.tags
        }


class TaskManager:
    """Manages task extraction, enhancement, and export"""
    
    def __init__(self):
        self.tasks: Dict[str, StructuredTask] = {}
        self.task_counter = 0
    
    def process_extracted_tasks(self, extracted_tasks: List[ExtractedTask], 
                              email_id: str, sender_email: str) -> List[StructuredTask]:
        """Convert extracted tasks to structured tasks"""
        structured_tasks = []
        
        for task in extracted_tasks:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{email_id}"
            
            # Enhance task with additional processing
            enhanced_task = self._enhance_task(task, task_id, email_id, sender_email)
            structured_tasks.append(enhanced_task)
            self.tasks[task_id] = enhanced_task
        
        return structured_tasks
    
    def _enhance_task(self, task: ExtractedTask, task_id: str, 
                     email_id: str, sender_email: str) -> StructuredTask:
        """Enhance extracted task with additional metadata"""
        
        # Generate title from description
        title = self._generate_task_title(task.task_description)
        
        # Determine priority
        priority = self._map_priority(task.priority)
        
        # Extract tags from description
        tags = self._extract_tags(task.task_description)
        
        # Estimate duration based on task type
        duration = self._estimate_duration(task.task_description)
        
        return StructuredTask(
            id=task_id,
            title=title,
            description=task.task_description,
            assignee=task.assignee,
            creator=sender_email,
            due_date=task.due_date,
            created_date=datetime.now(),
            priority=priority,
            status=TaskStatus.PENDING,
            tags=tags,
            source_email_id=email_id,
            confidence=task.confidence,
            estimated_duration=duration
        )
    
    def _generate_task_title(self, description: str) -> str:
        """Generate concise task title from description"""
        # Take first few words and clean up
        words = description.split()[:6]
        title = " ".join(words)
        
        # Remove common prefixes
        prefixes_to_remove = ['please', 'could you', 'can you', 'need to', 'should']
        for prefix in prefixes_to_remove:
            if title.lower().startswith(prefix):
                title = title[len(prefix):].strip()
                break
        
        # Capitalize first letter
        title = title[0].upper() + title[1:] if title else "Task"
        
        # Add ellipsis if truncated
        if len(words) > 6:
            title += "..."
        
        return title
    
    def _map_priority(self, priority_str: str) -> TaskPriority:
        """Map string priority to enum"""
        mapping = {
            'low': TaskPriority.LOW,
            'medium': TaskPriority.MEDIUM,
            'high': TaskPriority.HIGH,
            'urgent': TaskPriority.URGENT
        }
        return mapping.get(priority_str.lower(), TaskPriority.MEDIUM)
    
    def _extract_tags(self, description: str) -> List[str]:
        """Extract relevant tags from task description"""
        tags = []
        description_lower = description.lower()
        
        # Common task categories
        tag_keywords = {
            'meeting': ['meeting', 'call', 'conference', 'discussion'],
            'review': ['review', 'check', 'verify', 'validate'],
            'document': ['document', 'report', 'file', 'presentation'],
            'email': ['email', 'send', 'reply', 'forward'],
            'deadline': ['deadline', 'due', 'urgent', 'asap'],
            'follow-up': ['follow up', 'follow-up', 'check back'],
            'approval': ['approve', 'sign', 'authorize', 'confirm'],
            'research': ['research', 'investigate', 'analyze', 'study']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _estimate_duration(self, description: str) -> Optional[int]:
        """Estimate task duration in minutes"""
        description_lower = description.lower()
        
        # Duration keywords and their estimates
        duration_map = {
            'quick': 15,
            'brief': 30,
            'short': 30,
            'meeting': 60,
            'call': 30,
            'review': 45,
            'research': 120,
            'analysis': 180,
            'report': 240,
            'presentation': 180
        }
        
        for keyword, duration in duration_map.items():
            if keyword in description_lower:
                return duration
        
        # Default estimate based on description length
        word_count = len(description.split())
        if word_count < 5:
            return 15
        elif word_count < 10:
            return 30
        else:
            return 60
    
    def get_tasks_by_assignee(self, assignee: str) -> List[StructuredTask]:
        """Get all tasks assigned to a person"""
        return [task for task in self.tasks.values() 
                if task.assignee == assignee]
    
    def get_tasks_by_priority(self, priority: TaskPriority) -> List[StructuredTask]:
        """Get tasks by priority level"""
        return [task for task in self.tasks.values() 
                if task.priority == priority]
    
    def get_overdue_tasks(self) -> List[StructuredTask]:
        """Get tasks that are overdue"""
        now = datetime.now()
        return [task for task in self.tasks.values() 
                if task.due_date and task.due_date < now 
                and task.status == TaskStatus.PENDING]
    
    def get_upcoming_tasks(self, days: int = 7) -> List[StructuredTask]:
        """Get tasks due in the next N days"""
        now = datetime.now()
        future = now + timedelta(days=days)
        
        return [task for task in self.tasks.values() 
                if task.due_date and now <= task.due_date <= future
                and task.status == TaskStatus.PENDING]
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            return True
        return False
    
    def export_to_json(self, filename: str) -> None:
        """Export tasks to JSON file"""
        tasks_data = [task.to_dict() for task in self.tasks.values()]
        with open(filename, 'w') as f:
            json.dump(tasks_data, f, indent=2, default=str)
    
    def export_to_calendar(self, filename: str) -> None:
        """Export tasks as calendar events"""
        events = [task.to_calendar_event() for task in self.tasks.values() 
                 if task.due_date]
        
        with open(filename, 'w') as f:
            json.dump(events, f, indent=2, default=str)
    
    def export_to_todo_list(self, filename: str) -> None:
        """Export as todo list"""
        todo_items = [task.to_todo_item() for task in self.tasks.values()]
        
        with open(filename, 'w') as f:
            json.dump(todo_items, f, indent=2, default=str)
    
    def generate_task_report(self) -> Dict:
        """Generate task analytics report"""
        total_tasks = len(self.tasks)
        if total_tasks == 0:
            return {"message": "No tasks found"}
        
        # Status distribution
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(1 for task in self.tasks.values() 
                                            if task.status == status)
        
        # Priority distribution
        priority_counts = {}
        for priority in TaskPriority:
            priority_counts[priority.value] = sum(1 for task in self.tasks.values() 
                                                if task.priority == priority)
        
        # Assignee distribution
        assignee_counts = {}
        for task in self.tasks.values():
            assignee = task.assignee or "unassigned"
            assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
        
        # Overdue tasks
        overdue_count = len(self.get_overdue_tasks())
        upcoming_count = len(self.get_upcoming_tasks())
        
        return {
            "total_tasks": total_tasks,
            "status_distribution": status_counts,
            "priority_distribution": priority_counts,
            "assignee_distribution": assignee_counts,
            "overdue_tasks": overdue_count,
            "upcoming_tasks": upcoming_count,
            "completion_rate": status_counts.get("completed", 0) / total_tasks * 100
        }