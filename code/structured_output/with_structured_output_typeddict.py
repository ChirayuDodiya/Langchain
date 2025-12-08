from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.1)

#schema
class Review(TypedDict):
    # summary: str ->valid
    summary: Annotated[str,"A brief summary if the review"]
    # sentiment: str
    sentiment: Annotated[str,"Return sentiment of the review either negative, positive or neutral"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hpping for a software update to fix this.""")

print(result)