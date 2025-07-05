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
You are an expert email task extraction assistant. Your job is to analyze
email metadata and content to extract structured task information.

INSTRUCTIONS:
1. Read the email metadata (Message-ID, From, To, etc.) and content carefully
2. Extract exactly ONE task from the email (if any exists)
3. Use the Message-ID as the email_index (critical for data integrity)
4. Infer organizations from email domains (e.g., @company.com → "Company")
5. Extract people information from email headers (From, To, Cc, Bcc)
6. Assign a topic name (1-3 words) that groups related tasks
7. Return valid JSON following the exact structure shown below

EMAIL DATA TO ANALYZE:
\"\"\"
{{main_email}}
\"\"\"

EXTRACTION RULES:
- email_index: Use the exact Message-ID from email metadata
- Organizations: Extract from email domains (@company.com → "Company")
- People names: Use Name-From, Name-To fields when available, otherwise
  parse from email addresses
- Dates: Infer from email content or use email Date if no specific dates
  mentioned
- If no clear task exists, create a minimal task for the email's main purpose

REQUIRED JSON FORMAT:
{example_json}

CRITICAL JSON FORMATTING RULES:
- Use the exact Message-ID for email_index (found in EMAIL METADATA section)
- Use double quotes for all strings
- Add commas after every property except the last one
- Use proper array syntax: ["item1", "item2"] not 0:"item1", 1:"item2"
- Do not include markdown, backticks, or comments
- Ensure all braces and brackets are properly closed
- Extract organizations from email domains in From/To/Cc/Bcc fields

RESPOND WITH ONLY THE JSON - NO OTHER TEXT.
""")

# Update the example to show how to use Message-ID
example_json = '''{{
  "Topic": {{
    "name": "Deal Blotter Management",
    "tasks": [
      {{
        "email_index": "<26322156.1075841888052.JavaMail.evans@thyme>",
        "task": {{
          "name": "Configure off-peak deal entry settings in deal blotter",
          "summary": "Email discusses issues with default settings for " +
                     "traders' deal blotters and seeks solution for " +
                     "correctly entering off-peak deals including " +
                     "Sundays and holidays.",
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
            }}
          ]
        }}
      }}
    ]
  }}
}}'''