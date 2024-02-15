import os

from constants import GOOGLE_API_KEY
# set the google API keys for vision APIs
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from ingest import extract_retriever

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def main(show_resources):
    # extract the retriever
    retriever = extract_retriever()

    # set up the prompt template
    prompt_template = (
    """You are a helpful assistant and you will use the provided context to answer user questions. If you can not answer a user question based on the provided context, inform the user.
    
    Context: {context}
    User: {question}
    Answer:"""
    )

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    # setup langchain pipeline
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt":prompt}
    )

    # interactive questions and answers loop
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # get answer from the chain
        res = qa.invoke(query)
        answer, docs = res["result"], res["source_documents"]

        # print the query and results
        print("\n\n> Question:")
        print(query)

        print("\n> Answer:")
        print(answer)

        # print the source documents
        if show_resources:
            # this is a flag used to print relevant resources for answers
            print("--------------------------------------------------------------------SOURCE DOCUMENTS-------------------------------------------------------------")
            for document in docs:
                print("|-"*50)
                print("\n\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("--------------------------------------------------------------------SOURCE DOCUMENTS-------------------------------------------------------------")

if __name__ == "__main__":
    main(show_resources=False)