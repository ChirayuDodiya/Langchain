from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

# def word_count(text):
#     return len(text.split())

prompt=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    # 'word_count':RunnableLambda(word_count)
    'word_count':RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'AI'}))