from langchain.prompts import PromptTemplate

# Prompt for reasoning
reason_prompt = PromptTemplate.from_template("""
You are a helpful assistant supporting task management.
Given a user's question and a factual observation (task data from a knowledge graph),
your job is to respond clearly with:
- The tasks and the deadlines the user needs to focus on
- Any recommendation you have based on the data
- If the current date is past the due date, suggest a polite follow-up email to the task owner and cc the collaborators to check on the status, otherwise, you don't need to suggest a follow-up email.
Only use information from the observation. Do not invent tasks or timelines.

Today's date: {current_date}

Context:
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
  "Topic": {{
    "name": "Interview Scheduling",
    "tasks": [
      {{
        "email_index": "<message123@gmail.com>",
        "task": {{
          "name": "Schedule interview with Rachel Wang",
          "summary": "Interview scheduled for Thursday at 12:00 PM PT.",
          "start_date": "2025-06-26",
          "due_date": "2025-06-26",
          "owner": {{
            "name": "Shiqi Wang",
            "role": "Hiring Manager",
            "department": "Engineering",
            "organization": "CCM Chase"
          }},
          "collaborators": [
            {{
              "name": "Daniel Griffiths",
              "role": "Team Lead",
              "department": "Engineering",
              "organization": "CCM Chase"
            }},
            {{
              "name": "Aishwarya Vyas", 
              "role": "HR Coordinator",
              "department": "Human Resources",
              "organization": "CCM Chase"
            }}
          ]
        }}
      }}
    ]
  }}
}}'''

rag_extraction_prompt = PromptTemplate.from_template(f"""
You are analyzing emails to extract structured task information.
Each email contains comprehensive metadata and content.

EMAIL FORMAT EXPLANATION:
- Message-ID: Unique email identifier (use this as email_index)
- Date: When the email was sent
- From/To/Cc/Bcc: Participants with names and email addresses
- Subject: Email subject line
- Email Content: The actual message body

ORGANIZATION EXTRACTION TIPS:
- Check email domains (@company.com) to identify organizations
- Look for company signatures, titles, and department mentions
- Extract role information from email signatures or content
- Use "Unknown" only when information is genuinely unavailable

Your task is to:
- Assign a specific topic name (1–3 words)
- Extract topic metadata (start date, due date, description, owners)
- Extract one task from each email (use Message-ID as email_index)
- Identify organizations from email domains and signatures
- Return valid structured JSON like this:

{example_json}

CRITICAL JSON FORMATTING RULES:
- Use double quotes for all strings
- Add commas after every property except the last one
- Use proper array syntax: ["item1", "item2"] not 0:"item1", 1:"item2"
- Do not include markdown, backticks, or comments
- Ensure all braces and brackets are properly closed
- For email_index: Use the exact Message-ID provided
- For organizations: Extract from email domains (e.g., @google.com → "Google")

Context:
\"\"\"
{{main_email}}
\"\"\"
""")
