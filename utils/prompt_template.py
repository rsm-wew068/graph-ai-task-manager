from langchain.prompts import PromptTemplate

# Prompt for reasoning
reason_prompt = PromptTemplate.from_template("""
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
    "name": "Deal Blotter",
    "tasks": [
      {{
        "email_index": "<26322156.1075841888052.JavaMail.evans@thyme>",
        "task": {{
          "name": "Ensure off-peak deals are entered correctly in the deal blotter by checking settings and deal entry methods.",
          "summary": "The email discusses issues with the default settings for traders' deal blotters and seeks a solution for correctly entering off-peak deals.",
          "start_date": "2001-01-18",
          "due_date": "2001-01-25",
          "owner": {{
            "name": "Kate Symes",
            "role": "Senior Power Trader",
            "department": "Trading",
            "organization": "Enron"
          }},
          "collaborators": [
            {{
              "name": "Duong Luu",
              "role": "Trader",
              "department": "Southwest Desk",
              "organization": "Enron"
            }},
            {{
              "name": "Will Smith",
              "role": "Trader",
              "department": "Southwest Desk",
              "organization": "Enron"
            }}
          ]
        }}
      }},
      {{
        "email_index": "<4004888.1075841931363.JavaMail.evans@thyme>",
        "task": {{
          "name": "Ensure off-peak deals are entered correctly in the deal blotter by verifying settings and entries.",
          "summary": "The email discusses issues with the default settings for the deal blotter, specifically concerning the entry of off-peak deals and the inclusion of Sundays and holidays.",
          "start_date": "2001-01-18",
          "due_date": "2001-02-01",
          "owner": {{
            "name": "Kate Symes",
            "role": "Senior Power Trader",
            "department": "Trading",
            "organization": "Enron"
          }},
          "collaborators": [
            {{
              "name": "Unknown",
              "role": "Unknown",
              "department": "Unknown",
              "organization": "Unknown"
            }}
          ]
        }}
      }}
    ]
  }}
}}'''

rag_extraction_prompt = PromptTemplate.from_template(f"""
You are reviewing an email that contains task-related information.

Your task is to:
- Extract ONE actionable task from the main email
- Use all available email metadata (sender, recipients, dates, subject) to
  enrich the task details
- Identify the task owner from email participants (From, To, Cc fields)
- Extract or infer task deadlines from the email content and context
- Return valid structured JSON

EMAIL TO ANALYZE:
\"\"\"
{{main_email}}
\"\"\"

RELATED CONTEXT (for reference only):
{{related_email_1}}

{{related_email_2}}

EXTRACTION GUIDELINES:
1. EMAIL_INDEX: Message-ID is email_index
   - Use the Message-ID field from email metadata as the email_index value

2. OWNER IDENTIFICATION: Use email metadata to identify task owners:
   - The "Name-From" field often indicates who is assigning or reporting on the task
   - The "Name-To" field indicates primary recipients/responsible parties
   - If names are not in headers, parse from email signatures/content
   - You must return a name. If you cannot identify it from the email content, you should extract it from Name-From, Name-To fields

3. DEADLINE EXTRACTION: Look for dates in:
   - Email content mentioning "due", "deadline", "by [date]", "before"
   - Email sent date is the start date if not mentioned in the content
   - Subject line dates or urgency indicators
   - Email timestamps as context for relative dates ("by Friday", "next week")

3. TASK CONTEXT: Use subject line and email metadata to understand:
   - Priority level from subject indicators (URGENT, FYI, etc.)
   - Department/team context from sender domains and signatures
   - Project/topic context from subject prefixes or email threads

4. COLLABORATORS: Identify from From/To/Cc/Bcc fields and email content mentions

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no comments):

{example_json}

CRITICAL JSON FORMATTING RULES:
- Use double quotes for all strings
- Add commas after every property except the last one
- Use proper array syntax: ["item1", "item2"] not 0:"item1", 1:"item2"
- Do not include markdown, backticks, or comments
- Ensure all braces and brackets are properly closed
- Use null for missing dates, not empty strings

Context:
\"\"\"
{{main_email}}
\"\"\"
""")
