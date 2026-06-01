from pydantic import BaseModel, Field

class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    id: int = Field(description="Id of the record")
    source: str = Field(description="source of the chunk")
    chunk_index: int = Field(description="index of the chunk")
    content: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")
    similarity: float = Field(description="Similarity")

    def as_result(self, document):
        metadata = {"id": document["id"], "source": document["source"], "chunk_index": document["chunk_index"], "similarity": document["similarity"]}
        return Result(page_content=self.content, metadata=metadata)


class Chunks(BaseModel):
    chunks: list[Chunk]