import os
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
from langchain import OpenAI, PromptTemplate, VectorDBQA
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from preprocess import CalendarEvent, calendar_row_to_tsv, events_to_df

PERSIST_DIRECTORY = "db"


class DocRetrievalQNA:
    def __init__(self, embeddings, vectorstore, qa):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.qa = qa

    def ask(self, query):
        if self.qa is None:
            raise ValueError("No index loaded!")

        result = self.qa({"query": query})
        source_events = []
        for doc in result["source_documents"]:
            values = doc.page_content.split("\t")

            # HACK: fill in missing TSV values with empty strings
            if len(values) < len(CalendarEvent._fields):
                values.extend([""] * (len(CalendarEvent._fields) - len(values)))

            source_event = CalendarEvent(*tuple(values))
            source_events.append(source_event)

        df_source_events = events_to_df(source_events)

        return result["result"], df_source_events

    @classmethod
    def from_df(cls, df, prompt_context={}):
        embeddings = cls._get_embeddings()
        vectorstore = cls._index_documents(df, embeddings)
        prompt_template = cls._make_prompt_template(**prompt_context)
        qa = cls._chain_from_vectorstore(vectorstore, prompt_template)
        return cls(embeddings, vectorstore, qa)

    def save(self):
        self.vectorstore.persist()
        path = Path(PERSIST_DIRECTORY) / "qa.json"
        self.qa.save(path)

    @classmethod
    def load(cls, prompt_context={}):
        embeddings = cls._get_embeddings()
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings
        )
        prompt_template = cls._make_prompt_template(**prompt_context)
        qa = cls._chain_from_vectorstore(vectorstore, prompt_template)
        return cls(embeddings, vectorstore, qa)

    @staticmethod
    def _index_documents(df, embeddings):
        df = df.fillna("")
        # put all the content in one field as a tsv!
        df["content"] = df.apply(calendar_row_to_tsv, axis=1)

        loader = DataFrameLoader(df, page_content_column="content")
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        return Chroma.from_documents(
            texts,
            embeddings,
            include_metadata=True,
            persist_directory=PERSIST_DIRECTORY,
        )

    @staticmethod
    def _get_embeddings():
        return OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    @staticmethod
    def _make_prompt_template(name=None, emails=None):
        date_str = date.today().strftime("%Y-%m-%d")
        date_prompt = f"Today's date is {date_str}. "
        name_prompt = f"You are speaking to someone named {name}. " if name else ""
        email_prompt = (
            f"The person you're speaking to uses the following "
            f"email addresses: {emails}. "
            if emails
            else ""
        )
        tsv_header = "\t".join(CalendarEvent._fields)
        prompt_template = (
            date_prompt
            + name_prompt
            + email_prompt
            + (
                "Use the following calendar events to answer the question below. "
                f"The events are in a TSV format with the following header:\n"
                f"{tsv_header}\n\n"
                "If you don't know the answer, just say that you don't know, "
                "don't try to make up an answer. \n\n"
                "{context}\n\n"
                "Question: {question}\n"
                "Answer:\n"
            )
        )
        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    @staticmethod
    def _chain_from_vectorstore(vectorstore, prompt_template):
        qa = VectorDBQA.from_chain_type(
            llm=OpenAI(model_name="gpt-4", temperature=0.0),
            chain_type="stuff",
            k=25,
            vectorstore=vectorstore,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template, "verbose": True},
        )
        return qa


# A wrapper class that convert "event" objects into dataframes
# Then, calls a modular "QNA" class that does heavy lifting to index and query
class CalendarIndex:
    DF_PATH = Path(PERSIST_DIRECTORY) / "events.parquet"
    QNA_CLASS = DocRetrievalQNA

    def __init__(self, df, qna):
        self.df = df
        self.qna = qna

    @classmethod
    def from_events(cls, events: List[CalendarEvent]):
        df = pd.DataFrame(events, columns=CalendarEvent._fields)
        qna = cls.QNA_CLASS.from_df(df)
        return cls(df, qna)

    @classmethod
    def load(cls, prompt_context={}):
        df = pd.read_parquet(cls.DF_PATH)
        qna = cls.QNA_CLASS.load(prompt_context)
        return cls(df, qna)

    def save(self):
        self.df.to_parquet(self.DF_PATH)
        self.qna.save()

    def add_events(self, events: List[CalendarEvent]):
        self.df = pd.concat([self.df, events_to_df(events)])
        # TODO: add incrementally rather than re-creating
        self.qna = self.QNA_CLASS.from_df(self.df)

    def ask(self, query):
        return self.qna.ask(query)
