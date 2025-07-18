from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Load persisted brain
vectordb = Chroma(persist_directory="vector_db", embedding_function=OpenAIEmbeddings())
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

@app.route("/coach", methods=["POST"])
def coach():
    question = request.json.get("question")
    answer = qa_chain.run(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000)
