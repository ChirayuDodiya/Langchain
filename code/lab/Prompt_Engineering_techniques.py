from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1,max_tokens=1000)
parser = StrOutputParser()

# 1. Zero-shot Prompting
print("Zero-shot Prompting:")
prompt_zero = ChatPromptTemplate.from_template(
    "Suggest a One project idea in one line in the field of machine learning for college students."
)
chain_zero = prompt_zero | llm | parser
print(chain_zero.invoke({}))

# 2. One-shot Prompting
print("\nOne-shot Prompting:")
prompt_one = ChatPromptTemplate.from_template(
    "Rewrite the following text into a professional LinkedIn post:\n"
    "Text: Just won the hackathon, feeling great!\n"
    "Post: Excited to share that my team and I won the university hackathon this weekend! It was a great experience solving real-world problems.\n\n"
    "Text: {student_text}\nPost:"
)
chain_one = prompt_one | llm | parser
print(chain_one.invoke({
    "student_text": "Finished my internship, learned a lot!"
}))

# 3. Few-shot Prompting
print("\nFew-shot Prompting:")
prompt_few = ChatPromptTemplate.from_template(
    "Summarize the following college announcements in one line:\n"
    "Announcement: The library will be closed on Sunday due to maintenance.\nSummary: Library closed Sunday for maintenance.\n"
    "Announcement: All final year students must submit their thesis by March 25.\nSummary: Final year thesis due by March 25.\n"
    "Announcement: {announcement}\nSummary:"
)
chain_few = prompt_few | llm | parser
print(chain_few.invoke({
    "announcement": "The seminar on AI Ethics will be held on Tuesday at 10 AM in Hall 2."
}))

# 4. Chain of Thought Prompting (no input)
print("\nChain of Thought Prompting:")
prompt_cot = ChatPromptTemplate.from_template(
    "Q: If a pencil costs 2 rupees and a notebook costs 10 rupees, what is the total cost for 3 pencils and 2 notebooks?\n"
    "A: Let's calculate step by step.\n"
    "Cost of 3 pencils = 3 x 2 = 6 rupees\n"
    "Cost of 2 notebooks = 2 x 10 = 20 rupees\n"
    "Total cost = 6 + 20 = 26 rupees\n\n"
    "Q: If an eraser costs 5 rupees and a sharpener costs 15 rupees, what is the total cost for 4 erasers and 1 sharpener?\n"
    "A:"
)
chain_cot = prompt_cot | llm | parser
print(chain_cot.invoke({}))

# 5. Role Prompting
print("\nRole Prompting:")
prompt_role = ChatPromptTemplate.from_template(
    "You are a friendly and helpful math teacher.\n"
    "Explain how to find the area of a circle to a 10-year-old student."
)
chain_role = prompt_role | llm | parser
print(chain_role.invoke({}))

# 6. Output Formatting (JSON-style)
print("\nOutput Formatting (JSON):")
prompt_json = ChatPromptTemplate.from_template(
    "Extract the following information from the sentence and return as JSON:\n"
    "Sentence: {sentence}\n"
    "For Example, 'Bob sold a car' should return: \n"
    "person: Bob, action: sold, object: car\n"
    "Fields: person, action, object"
)
chain_json = prompt_json | llm | parser
print(chain_json.invoke({"sentence": "Alice bought a new bicycle."}))

# 7. Avoiding Hallucinations
print("\nHallucination Avoidance:")
prompt_hall = ChatPromptTemplate.from_template(
    "Answer the following question only if you are sure. If you are unsure, say 'I don't know.'\n"
    "Question: {question}"
)
chain_hall = prompt_hall | llm | parser
print(chain_hall.invoke({"question": "what is my name?"}))
