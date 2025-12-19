from langchain_core.prompts import PromptTemplate

# Prompt for reasoning
reason_prompt = PromptTemplate.from_template("""
You are a helpful assistant with access to task management data from a knowledge graph.

First, analyze the user's question to determine its intent:
- If it's about tasks, deadlines, topics, or work-related queries, use the task data provided
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
""")

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

# Prompt for RAG-based task extraction from email content
example_json = '''{{
  "task_name": "Ensure off-peak deals are entered correctly in the deal blotter by checking settings and deal entry methods.",
  "task_description": "The email discusses issues with the default settings for traders' deal blotters and seeks a solution for correctly entering off-peak deals.",
  "topic": "Deal Blotter",
  "due_date": "2001-01-25",
  "received_date": "2001-01-18",
  "status": "not started",
  "priority_level": "Medium",
  "sender": "Kate Symes",
  "assigned_to": "Duong Luu",
  "message_id": "<26322156.1075841888052.JavaMail.evans@thyme>",
  "spam": false,
  "validation_status": "llm",
  "confidence_score": 0.85
}}'''

rag_extraction_prompt = PromptTemplate.from_template(
    (
        "You are reviewing an email that contains task-related information.\n"
        "\n"
        "Your task is to:\n"
        "- Extract ONE actionable task from the main email\n"
        "- Use all available email metadata (sender, recipients, dates, subject) to enrich the task details\n"
        "- Identify the task owner from email participants (from_name/to_name_cc_name fields)\n"
        "- Extract or infer task deadlines from the email content and context\n"
        "- Return FLAT structured JSON (not nested)\n"
        "\n"
        "EMAIL TO ANALYZE:\n"
        "{main_email}\n"
        "\n"
        "RELATED CONTEXT (for reference only):\n"
        "{related_email_1}\n"
        "\n"
        "{related_email_2}\n"
        "\n"
        "EXTRACTION GUIDELINES:\n"
        "1. MESSAGE_ID: Use the exact Message-ID from email metadata\n"
        "   - Copy the Message-ID field exactly as shown in the email metadata\n"
        "   - Do not use \"Unknown\" or placeholder values\n"
        "\n"
        "2. TOPIC: Extract a meaningful topic name for the task\n"
        "   - Use subject line, content context, or business domain\n"
        "   - Examples: \"Finance\", \"Trading\", \"Operations\", \"HR\"\n"
        "\n"
        "3. TASK CONTEXT: Use subject line and email metadata to understand:\n"
        "   - Priority level from subject indicators (URGENT, FYI, etc.)\n"
        "   - Department/team context from sender domains and signatures\n"
        "   - Topic context from subject prefixes or email threads\n"
        "\n"
        "4. OWNER IDENTIFICATION: Use email metadata to identify task owners:\n"
        "   - The \"from_name\" field often indicates who is assigning or reporting on the task\n"
        "   - The \"to_name\" field indicates primary recipients/responsible parties\n"
        "   - If names are not in headers, parse from email signatures/content\n"
        "   - You must return a name. If you cannot identify it from the email content, you should extract it from from_name, to_name fields\n"
        "\n"
        "5. SENDER/ASSIGNED_TO: Use email participants:\n"
        "   - sender: Use the from_name (person's name, not email)\n"
        "   - assigned_to: Use the to_name (person's name, not email) or person mentioned in content\n"
        "\n"
        "6. STATUS: Use one of these exact values:\n"
        "   - \"not started\" (default for new tasks)\n"
        "   - \"in progress\" (if task mentions ongoing work)\n"
        "   - \"completed\" (if task mentions completion)\n"
        "\n"
        "7. DATE EXTRACTION: Look for dates in:\n"
        "   - Email content mentioning \"due\", \"deadline\", \"by [date]\", \"before\"\n"
        "   - Subject line dates or urgency indicators\n"
        "   - Email timestamps as context for relative dates (\"by Friday\", \"next week\")\n"
        "   - received_date: Use the Date field from email metadata\n"
        "   - due_date: Extract from content or infer from context\n"
        "\n"
        "OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no comments):\n"
        "\n"
        "{example_json}\n"
        "\n"
        "CRITICAL JSON FORMATTING RULES:\n"
        "- Use double quotes for all strings\n"
        "- Add commas after every property except the last one\n"
        "- Use proper array syntax: [\"item1\", \"item2\"] not 0:\"item1\", 1:\"item2\"\n"
        "- Do not include markdown, backticks, or comments\n"
        "- Ensure all braces and brackets are properly closed\n"
        "- Use null for missing dates, not empty strings\n"
        "\n"
        "Context:\n"
        "{main_email}\n"
    )
)

