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
You are reviewing a group of emails that belong to the same topic.

Each email contains exactly one task.

Your task is to:
- Assign a specific topic name (1–3 words)
- Extract topic metadata (start date, due date, description, owner(s), collaborators)
- Extract one task from each email (email_index, name, summary)
- Return valid structured JSON like this:

{example_json}

Guidelines:
- Each email = one task.
- Use double quotes.
- Do not include markdown, backticks, or comments.

Context:
\"\"\"
{{main_email}}
\"\"\"
""")