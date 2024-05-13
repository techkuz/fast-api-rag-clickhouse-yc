import clickhouse_connect

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.llms import YandexGPT
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_community.vectorstores import ClickhouseSettings, Clickhouse
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI()


class Query(BaseModel):
    text: str


@app.get("/clickhouse_connection")
def connection_status(
    host: str,
    port: int,
    user: str,
    password: str,
    ca_cert: str = '/etc/ssl/certs/ca-certificates.crt',
    secure: bool = True,
    verify: bool = True
) -> str:
    """Check if connection to ClickHouse works.

    Args:
        host (str): ClickHouse host
        port (int): ClickHouse port
        user (str): ClickHouse username
        password (str): ClickHouse password
        ca_cert (str, optional): CA certificate. Defaults to '/etc/ssl/certs/ca-certificates.crt'.
        secure (bool, optional): Whether to use SSL. Defaults to True.
        verify (bool, optional): Whether to verify SSL. Defaults to True.

    Returns:
        str: ClickHouse version
    """
    with clickhouse_connect.get_client(
            host=host, port=port, username=user,
            password=password, secure=secure, verify=verify, ca_cert=ca_cert) as ch_client:
        return ch_client.command('SELECT version()')


def get_s3csv_docs_loader(
    prefix: str,
    bucket: str,
    url: str,
    access_key_id: str,
    secret_access_key: str
) -> S3DirectoryLoader:
    """Load documents from S3 bucket.

    Args:
        prefix (str): S3 prefix
        bucket (str): S3 bucket
        url (str): S3 endpoint URL
        access_key_id (str): AWS access key ID
        secret_access_key (str): AWS secret access key

    Returns:
        S3DirectoryLoader: S3 directory loader
    """
    loader = S3DirectoryLoader(
        bucket=bucket,
        endpoint_url=url,
        prefix=prefix,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key
    )
    return loader


def split_docs(
    loader: S3DirectoryLoader,
    chunk_size: int,
    chunk_overlap: int
) -> list:
    """Split documents into chunks.

    Args:
        loader (S3DirectoryLoader): S3 directory loader
        chunk_size (int): Chunk size
        chunk_overlap (int): Chunk overlap

    Returns:
        list: Split documents
    """
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


def get_clickhouse_config(
    host: str,
    port: int,
    user: str,
    password: str
) -> ClickhouseSettings:
    """Get ClickHouse settings.

    Args:
        host (str): ClickHouse host
        port (int): ClickHouse port
        user (str): ClickHouse username
        password (str): ClickHouse password

    Returns:
        ClickhouseSettings: ClickHouse settings
    """
    ch_config = ClickhouseSettings(
        host=host, port=port, username=user, password=password)
    return ch_config


def get_yagpt_embeddings(
    folder_id: str,
    token: str
) -> YandexGPTEmbeddings:
    """Get Yandex GPT embeddings.

    Args:
        folder_id (str): Yandex GPT folder ID
        token (str): Yandex GPT IAM token

    Returns:
        YandexGPTEmbeddings: Yandex GPT embeddings
    """
    embeddings = YandexGPTEmbeddings(
        iam_token=token, folder_id=folder_id)
    return embeddings


@app.post("/embeddings/")
def load_embeddings(
    token: str,
    folder_id: str,
    ch_user: str,
    ch_password: str,
    ch_port: str,
    ch_host: str,
    s3_bucket: str,
    s3_url: str,
    s3_prefix: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    ch_ca_cert: str = '/etc/ssl/certs/ca-certificates.crt',
    ch_secure: bool = True,
    ch_verify: bool = True,
    split_chunk_size: int = 1000,
    split_chunk_overlap: int = 100
):
    """Load embeddings to ClickHouse.

    Args:
        token (str): Yandex GPT IAM token
        folder_id (str): Yandex GPT folder ID
        ch_user (str): ClickHouse username
        ch_password (str): ClickHouse password
        ch_port (str): ClickHouse port
        ch_host (str): ClickHouse host
        s3_bucket (str): S3 bucket
        s3_url (str): S3 endpoint URL
        s3_prefix (str): S3 prefix
        s3_access_key_id (str): AWS access key ID
        s3_secret_access_key (str): AWS secret access key
        ch_ca_cert (str, optional): CA certificate. Defaults to '/etc/ssl/certs/ca-certificates.crt'.
        ch_secure (bool, optional): Whether to use SSL. Defaults to True.
        ch_verify (bool, optional): Whether to verify SSL. Defaults to True.
        split_chunk_size (int, optional): Chunk size. Defaults to 1000.
        split_chunk_overlap (int, optional): Chunk overlap. Defaults to 100.

    """
    s3csv_loader = get_s3csv_docs_loader(
        prefix=s3_prefix,
        bucket=s3_bucket,
        url=s3_url,
        access_key_id=s3_access_key_id,
        secret_access_key=s3_secret_access_key
    )
    splitted_docs = split_docs(
        loader=s3csv_loader,
        chunk_size=split_chunk_size,
        chunk_overlap=split_chunk_overlap
    )
    embeddings = get_yagpt_embeddings(folder_id=folder_id, token=token)
    ch_config = get_clickhouse_config(
        host=ch_host,
        port=ch_port,
        user=ch_user,
        password=ch_password
    )
    # load data to clickhouse
    app.docsearch = Clickhouse.from_documents(
        splitted_docs,
        embeddings,
        config=ch_config,
        secure=ch_secure,
        verify=ch_verify,
        ca_cert=ch_ca_cert)


@app.get("/similar_docs")
def search_docs(
    query: Query,
    k: int = 2
):
    """Search for documents similar to query.

    Args:
        query (Query): Query
        k (int, optional): Number of results. Defaults to 2.

    Returns:
        list: Similar documents
    """
    app.docs = app.docsearch.similarity_search(query.text, k)
    return app.docs


@app.get("/emb_docs")
def read_docs():
    """Get documents (for debugging)

    Returns:
        list: Documents
    """
    return app.docs


@app.post("/llm")
def query_llm(
    token: str,
    folder_id: str,
    query: Query
):
    """Query Yandex GPT with query.

    Args:
        token (str): Yandex GPT IAM token
        folder_id (str): Yandex GPT folder ID
        query (Query): Query

    Returns:
        str: Response
    """
    prompt_template = "Ответь на вопрос {question}"
    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template
    )
    llm = YandexGPT(iam_token=token, folder_id=folder_id)
    chain = prompt | llm
    response = chain.invoke(query.text)

    return response


@app.post("/llm_rag")
def query_llm_rag(
    token: str,
    folder_id: str,
    query: Query
):
    """Query Yandex GPT with query using RAG prompt.

    Args:
        token (str): Yandex GPT IAM token
        folder_id (str): Yandex GPT folder ID
        query (Query): Query

    Returns:
        str: Response
    """
    # Промпт для обработки документов
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    # Промпт для языковой модели
    document_variable_name = "context"
    stuff_prompt_override = """
        Прими во внимание приложенные к вопросу тексты и дай ответ на вопрос.
        Текст:
        -----
        {context}
        -----
        Вопрос:
        {query}
    """
    prompt = PromptTemplate(
        template=stuff_prompt_override,
        input_variables=["context", "query"]
    )

    llm = YandexGPT(iam_token=token, folder_id=folder_id)

    # Создаём цепочку
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )

    response = chain.invoke({'query': query.text,
                             'input_documents': app.docs})

    return response
