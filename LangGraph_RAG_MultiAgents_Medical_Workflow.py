from langgraph.graph import StateGraph
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from typing import TypedDict

# -------------------------------
# STATE SCHEMA DEFINITION
# -------------------------------
class WorkflowState(TypedDict, total=False):
    document: str
    summary: str
    question: str
    entities: str
    answer: str

# -------------------------------
# SETUP EMBEDDINGS & VECTORSTORE
# -------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("medical_docs_db", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", k=5)

# -------------------------------
# LLM SETUP (Using Compatible Model)
# -------------------------------

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------------
# AGENT: Summarization
# -------------------------------
summarize_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a medical expert assistant. Summarize the following clinical note:

{input}

Return the summary in structured SOAP format.
"""
)
summarizer = summarize_prompt | llm

def SummarizationAgent(state):
    document = state["document"]
    summary = summarizer.invoke({"input": document})
    if isinstance(summary, dict) and "content" in summary:
        summary = summary["content"]
    return {"summary": summary}

# -------------------------------
# AGENT: Entity Extraction
# -------------------------------
ner_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Extract medical entities (e.g., ICD-10 codes, diagnoses, medications) from the following summary:

{input}

Return results in JSON.
"""
)
ner_agent = ner_prompt | llm

def EntityExtractionAgent(state):
    summary = state["summary"]
    entities = ner_agent.invoke({"input": summary})
    if isinstance(entities, dict) and "content" in entities:
        entities = entities["content"]
    return {"entities": entities}

# -------------------------------
# AGENT: Patient Query (with RAG)
# -------------------------------
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

def PatientQueryAgent(state):
    question = state["question"]
    result = rag_chain.invoke({"query": question})
    answer = result.get("result", result)
    return {"answer": answer}


# -------------------------------
# LANGGRAPH WORKFLOW
# -------------------------------
graph_builder = StateGraph(WorkflowState)
graph_builder.add_node("Summarizer", SummarizationAgent)
graph_builder.add_node("EntityExtractor", EntityExtractionAgent)
graph_builder.add_node("QnA", PatientQueryAgent)

graph_builder.set_entry_point("Summarizer")
graph_builder.add_edge("Summarizer", "EntityExtractor")
graph_builder.add_edge("Summarizer", "QnA")

compiled_graph = graph_builder.compile()

# -------------------------------
# INVOKE THE WORKFLOW
# -------------------------------
def run_medical_agent_workflow(document: str, question: str):
    input_data = {
        "document": document,
        "question": question
    }
    try:
        output = compiled_graph.invoke(input_data)
        return output
    except StopIteration:
        print("Error: Model or task not supported by current endpoint.")
        return {}

# -------------------------------
# TEST EXAMPLE
# -------------------------------
if __name__ == "__main__":
    # test_doc = """
    # Patient John Doe visited on 2024-03-15 complaining of frequent urination and fatigue. Labs showed elevated blood glucose. Diagnosed with Type 2 Diabetes. Prescribed Metformin 500mg daily.
    # """
    #test_question = "Why am I taking Metformin?"
    test_doc = """
    Patient John Door admitted on 2024-03-15 due to positive COVID-19 result. After nine days from ICU admission, the patient was successfully discharged from the hospital.
    """
    test_question = "What were the main improvements in the patient's medical condition?"

    results = run_medical_agent_workflow(test_doc, test_question)

    print("--- Summary ---")
    print(results.get("summary", "[No Summary Generated]") )

    print("\n--- Entities ---")
    print(results.get("entities", "[No Entities Extracted]"))

    print("\n--- Answer ---")
    print(results.get("answer", "[No Answer Retrieved]"))
