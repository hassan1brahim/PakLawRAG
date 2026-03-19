from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "../vectorstore_sections",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def query_vectorstore(query, k=3):
    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(query, k=k)

    print(f"\nQuery: {query}")
    print(f"\nTop {k} retrieved sections:\n")

    for i, doc in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Section ID: {doc.metadata.get('section_id')}")
        print(f"Source: {doc.metadata.get('source')}")
        print("-" * 80)
        print(doc.page_content[:1000])
        print()


if __name__ == "__main__":
    query = input("Enter your legal question: ")
    query_vectorstore(query)