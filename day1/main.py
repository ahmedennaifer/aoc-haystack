import os

from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.utils import Secret
from haystack_integrations.components.rankers.cohere import CohereRanker

load_dotenv()


if __name__ == "__main__":
    fetcher = LinkContentFetcher()
    converter = HTMLToDocument()
    reranker = CohereRanker(api_key=Secret.from_token(os.environ["COHERE_API_KEY"]))
    splitter = DocumentSplitter(split_by="sentence", split_length=10)
    template = """Given the information below, answer the query. Only use the provided context to generate the answer and output the used document links
                Context:
                {% for document in documents %}
                    {{ document.content }}
                    URL: {{ document.meta.url }}
                {% endfor %}

                Question: {{ query }}
                Answer:"""

    prompt_builder = PromptBuilder(template=template)
    generator = OpenAIGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url="https://api.groq.com/openai/v1",
        model="llama3-70b-8192",
        generation_kwargs={"max_tokens": 512},
    )
    pipeline = Pipeline()
    pipeline.add_component(name="fetcher", instance=fetcher)
    pipeline.add_component(name="converter", instance=converter)
    pipeline.add_component(name="splitter", instance=splitter)
    pipeline.add_component(name="reranker", instance=reranker)
    pipeline.add_component(name="prompt_builder", instance=prompt_builder)
    pipeline.add_component(name="generator", instance=generator)

    pipeline.connect("fetcher", "converter")
    pipeline.connect("converter", "splitter")
    pipeline.connect("splitter", "reranker")
    pipeline.connect("reranker", "prompt_builder")
    pipeline.connect("prompt_builder", "generator")

    urls = [
        "https://haystack.deepset.ai/blog/extracting-metadata-filter",
        "https://haystack.deepset.ai/blog/query-expansion",
        "https://haystack.deepset.ai/blog/query-decomposition",
        "https://haystack.deepset.ai/cookbook/metadata_enrichment",
    ]

    # query = "What is the difference between metadata filtering and metadata enrichment?"
    query = "Which methods can I use to transform query for better retrieval?"
    # query = "How can I use metadata to improve retrieval?"
    # query = "What's preprocessing?" # Should return no answer

    result = pipeline.run(
        data={
            "fetcher": {"urls": urls},
            "prompt_builder": {"query": query},
            "reranker": {"query": query},
        }
    )
    print(result["generator"]["replies"][0])


"""
You can use three methods to transform a query for better retrieval:

1. **Extract Metadata from Queries to Improve Retrieval**: Use LLMs t       o extract metadata from queries to use as filters that improve retrieval in RAG applications. This method improves the quality of the retrieved documents by narrowing down the search space based on concrete metadata such as domain, source, date, or topic.

2. **Query Expansion**: Expand keyword queries to improve recall and provide more context to RAG. This method increases the number of results, thereby increasing recall. It involves generating a certain number of similar queries based on the user query, which helps to cover d       ifferent aspects of the original query.

3. **Query Decomposition**: Decompose queries that are multiples in disguise and have an LLM reason about the final answer. This method involves breaking down complex queries into smaller sub-questions that can be answered independently, and then reasoning about the final ans       wer based on each sub-answer.

Used document links:
- https://haystack.deepset.ai/blog/query-decomposition
- https://haystack.deepset.ai/blog/query-expansion
- https://haystack.deepset.ai/blog/extracting-metadata-filter
"""
