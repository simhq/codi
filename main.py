import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from urllib.parse import urlparse, urlunparse

# Load .env file
load_dotenv()

# Read API Key from .env or Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found! Please set it in the `.env` file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

VECTOR_STORE_PATH = "vector_store.faiss"

@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    return None

@st.cache_resource
def create_rag_chain(_vector_store):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    retriever = _vector_store.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    system_message = SystemMessage(content=(
        "Strictly use the retrieved context verbatim. Do not generate information beyond the provided context. Do not hallucinate."
        "If the answer is not found, tell the user that the answer is not found."
        "Ensure consistent font usage throughout the response."
    ))
    memory.chat_memory.add_message(system_message)

    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory, 
        return_source_documents=True,
        output_key="answer"
    )

def normalize_url(url):
    parsed_url = urlparse(url)
    return urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))

def enhance_prompt(prompt):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    enriched_prompt = llm.predict(f"In the context of Singapore Infocomm Media Development Authority, enrich this prompt for a more effective RAG search. If the prompt appears to be a name, the enriched prompt expand the search to include the possibility that this person is part of the senior management, or this person is part of the management in a Group in IMDA, otherwise keep the original prompt as it is: {prompt}. Output only the prompt.")
    return enriched_prompt

def validate_and_enhance_response(answer, sources):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    source_contents = [f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(sources)]
    sources_text = "\n\n".join(source_contents)
    
    validation_prompt = f"""Review and enhance this answer while adhering to these criteria:
    1. Accuracy: Verify the information matches the provided sources below
    2. Completeness: Ensure all relevant information from the sources is included
    3. Clarity: Make sure the response is well-structured and easy to understand
    4. Consistency: Check for any contradictions between the answer and sources
    5. Citations: Verify that all information comes from the provided sources
    6. If any information in the answer is not found in the sources, remove it
    7. If the answer contains information not supported by the sources, respond with "No answer found."

    Sources:
    {sources_text}

    Original answer: {answer}

    If the answer needs improvement, provide only the enhanced version that strictly uses information from the sources. If it's already optimal and matches the sources, return it unchanged.
    """
    
    enhanced_response = llm.predict(validation_prompt)
    return enhanced_response

def main():
    st.set_page_config(page_title="💬 Ask CODI 2.0", layout="wide")

    st.markdown("""
        🔬 Proof-of-Concept (POC):
         This is an experimental AI-powered assistant currently in POC phase. All responses are AI-generated and may contain errors. Always verify information with official sources.
         """)

    vector_store = load_vector_store()
    if not vector_store:
        st.warning("⚠️ No vector store found. Please ensure it is prepared before running the app!")
        return

    rag_chain = create_rag_chain(vector_store)

    st.title("💬 CODI 2.0")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "query" not in st.session_state:
        st.session_state["query"] = ""

    if st.button("🆕 New Query"):
        st.session_state["chat_history"] = []
        st.session_state["query"] = ""
        rag_chain.memory.clear()
        st.rerun()

    query = st.text_input("🔍 Ask CODI:", key="query")

    if query.strip():
        with st.spinner("🔎 Searching and Validating..."):
            try:
                enhanced_query = enhance_prompt(query)

                chat_history = []
                for entry in st.session_state["chat_history"]:
                    if entry["role"] == "You":
                        chat_history.append(HumanMessage(content=entry["message"]))
                    else:
                        chat_history.append(AIMessage(content=entry["answer"]))

                result = rag_chain({"question": enhanced_query, "chat_history": chat_history})

                answer = result.get("answer", "No answer found on the Corporate Website.")
                sources = result.get("source_documents", [])

                if sources:
                    answer = validate_and_enhance_response(answer, sources)

                if not sources:
                    answer = "No answer found on the Corporate Website."

                st.session_state["chat_history"].insert(0, {
                    "role": "You",
                    "message": query,
                    "answer": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    for entry in st.session_state["chat_history"]:
        query = entry["message"]
        answer = entry["answer"]
        sources = entry.get("sources", [])

        st.write(f"**🧑‍💻 You:** {query}")
        st.write(f"**🤖 Bot:** {answer}")

        if sources:
            unique_sources = set()
            displayed_sources = 0
            for doc in sources:
                source_url = doc.metadata.get('source', 'Unknown source')
                normalized_source = normalize_url(source_url).lower()
                if normalized_source != 'unknown' and normalized_source not in unique_sources:
                    unique_sources.add(normalized_source)
                    displayed_sources += 1
                    st.write(f"**📌 Source {displayed_sources}:** {source_url}")
                    if displayed_sources == 5:
                        break
        else:
            st.write("_No sources found for this response._")

        st.markdown("---")

if __name__ == "__main__":
    main()