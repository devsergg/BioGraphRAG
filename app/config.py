from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "pain-trials"
    neo4j_uri: str
    neo4j_username: str = "neo4j"
    neo4j_password: str
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "biotech-graphrag"
    semantic_scholar_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
