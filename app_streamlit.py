import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from textwrap import fill


class interactive_document:
    """
    Class to receive a PDF file and prepare it for querying using an LLM.
    """

    def __init__(self, file_path: str):
        """
        Load the PDF given in 'file_path' and initialize the LLM and its prompt.
        """
        loader = PyPDFLoader(file_path)
        self.docs = loader.load()
        self.llm = ChatOpenAI(
            temperature=0.0,
            model="llama3.2:3b",
            openai_api_key="ollama",
            openai_api_base="http://localhost:11434/v1"
            )
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm, 
            self.prompt
            )

    @st.cache
    def vectorize(self):
        """
        Vectorize the PDF and set up retriever for querying.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,
            add_start_index=True
            )
        self.all_splits = text_splitter.split_documents(self.docs)
        self.vectorstore = Chroma.from_documents(
            documents=self.all_splits,
            embedding=OllamaEmbeddings(model="llama3.2:3b")
            )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr"
            )
        self.rag_chain = create_retrieval_chain(
            self.retriever, 
            self.question_answer_chain
            )
        
    def ask(self, query: str):
        """
        Query the PDF.
        """
        response = self.rag_chain.invoke({"input": query})
        self.answer = response["answer"]


# Initialize the streamlit app page
st.set_page_config(layout='wide')
st.title("PDF Chatter", width='content')
fileupload, questions, answers = st.columns(3)

uploaded_file = fileupload.file_uploader(
    "Choose a PDF file:",
    type="pdf",
    accept_multiple_files=False
    )

if uploaded_file is not None:
    with fileupload:
        file = interactive_document(uploaded_file.name)
        st.success("Document loaded!")
        with st.spinner(text="Processing your PDF...", show_time=True):
            file.vectorize()
        st.success("Document processed!")
        st.write("You can now chat with your PDF.")
        
with questions:
    st.text_input()
    # with st.form(key="query_form"):
    #     query = st.text_input("Your question:", value="")
    #     submit_button = st.form_submit_button(label="Ask")

with answers:
    if query != "":
        with st.spinner(text="Generating a response...", show_time=True):
            file.ask(query)
        st.write(fill(file.answer))

    ### need to figure out how to prevent vectorization from running when the query is submitted