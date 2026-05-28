from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
import asyncio

from src.retriever.hybrid_retriever import HybridRetriever, qdrant_client
from src.retriever.reranker import reranker, get_reranked_documents

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
# llm = ChatMistralAI(model="mistral-small-2603",temperature=0)

searcher = HybridRetriever(
    collection_name="main_vector_store", qdrant_client=qdrant_client)


class RagState(TypedDict):

    query: str
    retrieved_results: List[Dict]
    reranked_results: List[Dict]
    answer: str


prompt_template = ChatPromptTemplate([
    (
        "system",
        """
        You are a highly accurate RAG AI assistant.

        Your job is to answer ONLY from the provided context.

        STRICT RULES:
        1. Do NOT use outside knowledge.
        2. If the answer is not present in the context, say:
        "I could not find the answer in the provided document."
        3. Every answer MUST include citations using page numbers, source and title.
        4. If information comes from multiple pages, cite all relevant pages.
        5. At the end, provide a confidence level:
        - HIGH → answer clearly supported by context
        - MEDIUM → partially supported or inferred from context
        - LOW → weak evidence or incomplete information
        6. Keep answers concise, factual, and grounded.
        7. Never hallucinate page numbers or facts.
        8. If page numbers,source or title are unavailable in context metadata, say:
        "<Page number / Source / Title> not available."

        RESPONSE FORMAT:

        Answer:
        <your answer>

        Citations:
        <title>
        <Page 3>
        <Page 5, Page 8>

        Confidence:
        <HIGH / MEDIUM / LOW>
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {query}
        """
    )
])


# building nodes
async def retriever_node(state: RagState):
    query = state['query']
    results = await asyncio.to_thread(searcher.search, query_text=query)
    return {'retrieved_results': results}


async def reranker_node(state: RagState):
    query = state['query']
    retrieved_results = state['retrieved_results']
    reranked_results = await asyncio.to_thread(
        get_reranked_documents, query=query, retrieved_results=retrieved_results, top_k=3)
    return {'reranked_results': reranked_results}


async def answer_node(state: RagState):

    query = state['query']
    context = state['reranked_results']

    context = "\n".join(
        f"""
    Source: {chunk['metadata'].get('file_name')}
    Page: {chunk['metadata'].get('page')}
    Content:
    {chunk['document']}
    """
        for chunk in context
    )
    final_prompt = prompt_template.invoke({
        'query': query,
        'context': context
    })

    response = await llm.ainvoke(final_prompt)
    return {'answer': response.content}

# building the graph


def build_graph():
    graph = StateGraph(RagState)

    graph.add_node('retriever_node', retriever_node)
    graph.add_node('reranker_node', reranker_node)
    graph.add_node('answer_node', answer_node)

    graph.add_edge(START, 'retriever_node')
    graph.add_edge('retriever_node', 'reranker_node')
    graph.add_edge('reranker_node', 'answer_node')
    graph.add_edge('answer_node', END)

    workflow = graph.compile()

    return workflow
