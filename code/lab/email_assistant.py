from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)

def classify_email(email_text):
    classification_prompt = ChatPromptTemplate.from_template("""
    You are an HR assistant. Classify the following email into one of these categories:
    - leave_request
    - salary_query
    - job_opening
    - policy_feedback

    Only return the category name, nothing else.

    Email:
    {email_text}
    """)
    prompt = classification_prompt.format_messages(email_text=email_text)
    result = model.invoke(prompt)
    return result.content.strip()


response_templates = {
    "leave_request": ChatPromptTemplate.from_template("""
        You are an HR assistant. Reply to the following leave request email politely.
        Include approval steps, policy reminders, and any required forms.
        Email: {email_text}
    """),
    "salary_query": ChatPromptTemplate.from_template("""
        You are an HR assistant. Reply to the following salary query email with clarity.
        Mention payroll timelines, escalation options, and contact info.
        Email: {email_text}
    """),
    "job_opening": ChatPromptTemplate.from_template("""
        You are an HR assistant. Reply to the following job opening inquiry.
        Include details on how to apply, eligibility, and expected timelines.
        Email: {email_text}
    """),
    "policy_feedback": ChatPromptTemplate.from_template("""
        You are an HR assistant. Reply to the following feedback on company policy.
        Thank them, acknowledge suggestions, and mention the review process.
        Email: {email_text}
    """)
}


def generate_reply(category, email_text):
    if category not in response_templates:
        return "Sorry, I couldn't determine the appropriate reply template."
    prompt = response_templates[category].format_messages(email_text=email_text)
    result = model.invoke(prompt)
    return result.content.strip()


examples = {
    "leave_request": "Hi HR, I need to take 3 days off next week due to a family emergency.",
    "salary_query": "Hello, I noticed my salary for this month hasn't been credited yet. Can you check?",
    "job_opening": "Dear HR, I saw a posting for the Data Analyst role. Could you share more details?",
    "policy_feedback": "I think our remote work policy could be more flexible. Here's my suggestion..."
}

for example_category, email in examples.items():
    print("="*60)
    print(f" Input Email:\n{email}")
    predicted_category = classify_email(email)
    print(f"\n Classified as: {predicted_category}")
    reply = generate_reply(predicted_category, email)
    print(f"\n Auto-Reply:\n{reply}")
    print("="*60, "\n")