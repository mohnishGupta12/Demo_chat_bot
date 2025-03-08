from pydantic import BaseModel , Field

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is AI?")
    collection_name: str = Field(..., example="abinbev_docs_v2")

class EmbedRequest(BaseModel):
    collection_name: str = Field(..., description="Collection name is required")
    directory: str = Field(..., description="Directory path is required")