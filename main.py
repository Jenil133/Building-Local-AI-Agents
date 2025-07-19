from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Langchain is a framework for building applications with LLMs.# Ollama is a tool for running LLMs locally.
# Chroma is a vector database for storing and retrieving embeddings.
# This code snippet demonstrates how to use these tools together.

model = OllamaLLM(model="llama3.2:latest")

template = """
Your are expert in answerting questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is a question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n==========================")
    question = input("Enter a question ('q' to quit): ")
    print("\n\n")
    if question.lower() == 'q':
        break

    # reviews = retriver.invoke({"query": question})
    # result = chain.invoke({"reviews": reviews, "question": question})
    docs = retriever.invoke(question)
    reviews_text = "\n\n".join([doc.page_content for doc in docs])  # Convert to string
    result = chain.invoke({"reviews": reviews_text, "question": question})
    print(result)
