import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("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")

app = FastAPI()

# Allow access from iOS apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector DB
vectordb = Chroma(
    persist_directory="vector_db",
    embedding_function=OpenAIEmbeddings(api_key=api_key),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=api_key),
    retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
)

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query")
    profile = data.get("profile", {})

    if not query:
        return {"error": "Missing query"}

    wrapped_query = f"""
            You are a professional AI fitness and wellness coach.

            Use the following user profile details to tailor your advice:
            - Injuries: {profile.get("injuries", "None")}
            - Health conditions: {profile.get("conditions", "None")}
            - Equipment available: {", ".join(profile.get("equipment", []))}
            - Time per day: {profile.get("time", "Not specified")}
            - Fitness goal: {profile.get("goal", "None")}

            Guidelines:
            - Only suggest exercises using listed equipment.
            - Avoid exercises that may worsen listed injuries.
            - If asthma or similar conditions are present, suggest low-impact or alternative cardio.
            - Make the plan achievable based on available time.
            - Never repeat the profile info in your response. Just use it to inform your advice.

            Now answer this question from the user:
            \"{query}\"
            """

    response = qa_chain.invoke({"query": wrapped_query})

    sources = []
    if "source_documents" in response:
        sources = [doc.page_content for doc in response["source_documents"]]

    return {
        "response": response.get("result", "").strip() or "Here’s what the documents say:\n" + "\n\n".join(sources[:3]),
        "source_documents": sources
    }

@app.get("/ping")
async def ping():
    return {"msg": "pong"}