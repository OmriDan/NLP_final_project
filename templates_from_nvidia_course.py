from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


"""
In order, this zero-shot classification chain:

Takes in a dictionary with two required keys, input and options.
Passes it through the zero-shot prompt to get the input to our LLM.
Passes that string to the model to get the result.
Task: Pick out several models that you think would be good for this kind of task and see how well they perform! Specifically:

Try to find models that are predictable across multiple examples. If the format is always easy to parse and extremely
predictable, then the model is probably ok.
Try to find models that are also fast! This is important because internal reasoning generally happens behind the hood
before the external response gets generated. Thereby, it is a blocking process which can slow down start of "user-facing"
generation, making your system feel sluggish.
"""
## Feel free to try out some more models and see if there are better lightweight options
## https://build.nvidia.com
instruct_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2")

sys_msg = (
    "Choose the most likely topic classification given the sentence as context."
    " Only one word, no explanation.\n[Options : {options}]"
)

## One-shot classification prompt with heavy format assumptions.
zsc_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_msg),
    ("user", "[[The sea is awesome]]"),
    ("assistant", "boat"),
    ("user", "[[{input}]]"),
])

## Roughly equivalent as above for <s>[INST]instruction[/INST]response</s> format
zsc_prompt = ChatPromptTemplate.from_template(
    f"{sys_msg}\n\n"
    "[[The sea is awesome]][/INST]boat</s><s>[INST]"
    "[[{input}]]"
)

zsc_chain = zsc_prompt | instruct_llm | StrOutputParser()

def zsc_call(input, options=["car", "boat", "airplane", "bike"]):
    return zsc_chain.invoke({"input" : input, "options" : options}).split()[0]

print("-" * 80)
print(zsc_call("Should I take the next exit, or keep going to the next one?"))

print("-" * 80)
print(zsc_call("I get seasick, so I think I'll pass on the trip"))

print("-" * 80)
print(zsc_call("I'm scared of heights, so flying probably isn't for me"))


## Should use KnowledgeBase from LangChain for our project

"""
Query Embedding
Purpose: Designed for embedding shorter-form or question-like material, such as a simple statement or a question.
Method: Utilizes embed_query for embedding each query individually.
Role in Retrieval: Acts as the "key" creator to enable search (query process) in a document retrieval framework.
Usage Pattern: Embedded dynamically, as needed, for comparison against a pre-processed collection of document embeddings.

Document Embedding
Purpose: Tailored for longer-form or response-like content, including document chunks or paragraphs.
Method: Employs embed_documents for batch processing of documents.
Role in Retrieval: Acts as the "value" creator to make the searchable content for the retrieval system.
Usage Pattern: Typically embedded en masse as a pre-processing step, creating a repository of document embeddings for future querying.

If using embeddings, nice experiment can be cross-similarity matrix of the questions

Facebook AI Similarity Search (FAISS) is a library for efficient similarity search and clustering of dense vectors.
It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
It also includes supporting code for evaluation and parameter tuning.
https://python.langchain.com/docs/integrations/vectorstores/faiss/

Use ragas for RAG evaluation (LLM-as-a-judge)
https://arxiv.org/abs/2306.05685
https://docs.ragas.io/en/stable/

"""