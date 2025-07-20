from langchain.prompts import PromptTemplate

# Prompt for reasoning
definition_reason_prompt = """
You are a helpful assistant with access to task management data from a knowledge graph.

First, analyze the user's question to determine its intent:
- If it's about tasks, deadlines, projects, or work-related queries, use the task data provided
- If it's a general question unrelated to task management, answer normally without forcing task information

For task management questions, respond with:
- The relevant tasks and deadlines the user needs to focus on
- Any recommendations based on the data
- If the current date is past the due date, suggest a polite follow-up email to the task owner and cc the collaborators to check on the status

For general questions, provide a helpful answer based on your knowledge, and only mention tasks if they're genuinely relevant.

Only use information from the observation for task-related responses. Do not invent tasks or timelines.

Today's date: {current_date}

Task Management Context (use only if relevant to the question):
{observation}

User Question:
{name} — {input}
"""
reason_prompt = PromptTemplate.from_template(definition_reason_prompt)

# Prompt for verifying the LLM's answer
verify_prompt = PromptTemplate.from_template(
    """You are a QA and compliance assistant for task management responses.

Your job is to:
- Verify the draft response is based strictly on the context
- Ensure all named tasks, due dates, and responsible individuals are mentioned
- Recommend a follow-up email ONLY if the due date has passed
- Fix vague or generic phrasing
- Do not invent or speculate beyond the given context

Context:
{observation}

Original Answer:
{draft}

Final Answer (improved):
""")

# Prompt for ChainQA answer generation
chainqa_answer_prompt = PromptTemplate.from_template(
    """You are a helpful AI assistant that provides COMPLETE and ACCURATE answers based on database data.
    
    Question: {question}
    
    DATA ANALYSIS:
    - Total entities found: {entity_count}
    - Discovery data: {discovery_data}
    - Exploration data: {exploration_data}
    - Reasoning: {reasoning}
    
    CRITICAL REQUIREMENTS:
    
    1. **COUNT QUERIES** (when asked "how many"):
       - Provide EXACT count from the data
       - If data shows 14 entities but only 6 unique tasks, explain the difference
       - Say "I found X unique tasks" and "Y total entities (including duplicates)"
    
    2. **LIST QUERIES** (when asked "show all", "list", "what tasks"):
       - Show EVERY SINGLE ITEM found, no truncation
       - Organize by categories if possible
       - Remove duplicates and say "X unique items found"
       - Use bullet points for readability
    
    3. **PERSON QUERIES** (when asked "who is X"):
       - Find ALL information about the person
       - Show their role, department, contact info if available
       - List ALL tasks/relationships involving this person
       - If limited info, say "Limited information available: [what we know]"
    
    4. **GENERAL REQUIREMENTS**:
       - Be SPECIFIC and PRECISE
       - Include ALL relevant details
       - Explain any data inconsistencies
       - If data is incomplete, acknowledge it
       - Use clear formatting with bullet points
    
    EXAMPLES OF GOOD RESPONSES:
    
    For "How many tasks are about packages?":
    "I found 14 entities related to packages, but only 6 unique tasks:
    • Pick up package from mailroom (appears 3 times)
    • Pick up package from Nuevo East mail room (appears 3 times)
    Total: 6 unique package pickup tasks"
    
    For "Who is Kyle?":
    "Based on the data, Kyle is mentioned in 1 task:
    • Task: Reschedule meeting with Kyle
    • Email source: [email address]
    Unfortunately, no additional personal details (role, department, contact) are available in the database."
    
    For "Show me all my tasks":
    "I found 17 total entities representing 11 unique tasks:
    
    **Meeting/Appointment Tasks:**
    • Reschedule meeting with Kyle
    • Reschedule coaching appointment (appears 2 times)
    
    **Package Pickup Tasks:**
    • Pick up package from mailroom (appears 3 times)
    • Pick up package from Nuevo East mail room (appears 3 times)
    
    **Maintenance/Repair Tasks:**
    • Submit FixIt request for repairs
    
    **Administrative Tasks:**
    • Recall message 'Rubio's in MPR2'
    • Complete Triton Food Pantry Survey
    • Confirm or update access receipt
    
    **Other Tasks:**
    • Clean common areas
    • Return apartment keys
    
    Total: 11 unique tasks across 17 total entities"
    
    Provide a comprehensive, accurate, and well-organized response.
    """
)

example_json = '''{{
  "Name": "Submit onboarding documents",
  "Task Description": "The HR team requests that you upload your onboarding forms before your start date.",
  "Due Date": "2025-07-20",
  "Received Date": "2025-07-18",
  "Status": "Not started",
  "Topic": "HR Onboarding",
  "Priority Level": "P1",
  "Sender": "hr@company.com",
  "Assigned To": "rachel@ucsd.edu",
  "Email Source": "<message-id-from-header>",
  "Spam": false
}}'''

rag_extraction_prompt = PromptTemplate.from_template(f"""
You are reviewing an email that contains task-related information.

Your task is to:
- Extract ONE actionable task from the email content and metadata
- Fill in the following fields for the Notion database:
  - Name: Task name (short, clear title of the task)
  - Task Description: Summary or instruction (what needs to be done)
  - Due Date: Date the task is due (if stated or implied)
  - Received Date: Date the email was received (use the raw email's 'Date' field)
  - Status: Choose one of [Not started, In progress, Done] based on context (or leave as 'Not started' if unclear)
  - Topic: Topic category (e.g., Capstone Project, Housing, Finance) inferred from subject or content
  - Priority Level: One of [P1, P2, P3] if inferred from subject line or urgency cues (else null)
  - Sender: Email address from the 'From' field
  - Assigned To: Email of the person expected to complete the task (from context, 'To', or inferred)
  - Email Source: Message-ID from the email metadata
  - Spam: Boolean indicating if this appears to be spam (true/false)

EMAIL TO ANALYZE:
\"\"\"
{{main_email}}
\"\"\"

RELATED CONTEXT (for reference only):
{{related_email_1}}

{{related_email_2}}

EXTRACTION GUIDELINES:
1. OWNER IDENTIFICATION: Use email metadata to identify task owners:
   - The "Name-From" field often indicates who is assigning or reporting on the task
   - The "Name-To" field indicates primary recipients/responsible parties
   - If names are not in headers, parse from email signatures/content
   - You must return an email address. If you cannot identify it from the email content, you should extract it from From, To fields

2. DEADLINE EXTRACTION: Look for dates in:
   - Email content mentioning "due", "deadline", "by [date]", "before"
   - Subject line dates or urgency indicators
   - Email timestamps as context for relative dates ("by Friday", "next week")

3. TASK CONTEXT: Use subject line and email metadata to understand:
   - Priority level from subject indicators (URGENT, FYI, etc.)
   - Department/team context from sender domains and signatures
   - Project/topic context from subject prefixes or email threads

4. SPAM DETECTION: Mark as spam if:
   - Email appears to be automated/marketing
   - Contains suspicious links or requests
   - From unknown or suspicious senders
   - Contains typical spam indicators

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no comments):

{example_json}

CRITICAL JSON FORMATTING RULES:
- Use double quotes for all strings
- Add commas after every property except the last one
- Use proper array syntax: ["item1", "item2"] not 0:"item1", 1:"item2"
- Do not include markdown, backticks, or comments
- Ensure all braces and brackets are properly closed
- Use null for missing dates, not empty strings
- Use true/false for boolean values, not "true"/"false"

Context:
\"\"\"
{{main_email}}
\"\"\"
""")