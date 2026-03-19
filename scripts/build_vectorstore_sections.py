import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_sections(json_path="../output/ppc_sections.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        sections = json.load(f)
    return sections


def convert_to_documents(sections):
    documents = []

    for sec in sections:
        section_id = sec["section_id"]
        text = sec["text"]

        doc = Document(
            page_content=text,
            metadata={
                "section_id": section_id,
                "source": f"PPC Section {section_id}"
            }
        )

        documents.append(doc)

    return documents


def build_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    save_path = "../vectorstore_sections"
    Path(save_path).mkdir(exist_ok=True)

    vectorstore.save_local(save_path)

    return vectorstore


if __name__ == "__main__":
    sections = load_sections()
    documents = convert_to_documents(sections)

    print(f"Loaded {len(sections)} parsed sections")
    print(f"Converted {len(documents)} documents")

    vectorstore = build_vectorstore(documents)

    print("Vector store created successfully at ../vectorstore_sections")