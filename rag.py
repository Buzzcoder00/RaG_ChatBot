from dotenv import load_dotenv
load_dotenv()
from google import genai
client = genai.Client()
from google.genai import types

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

db = lancedb.connect("./embeddings")
model = (
    get_registry()
    .get("sentence-transformers")
    .create(name="BAAI/bge-small-en-v1.5", device="cpu")
)

class Words(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

table = db.create_table("words", schema=Words)
table.add(
    [
        {"text": "Honey never spoils and can last for thousands of years."},
        {"text": "Bananas are berries but strawberries are not."},
        {"text": "Wombat poop is cube-shaped to stop it rolling away."},
        {"text": "A day on Venus is longer than a year on Venus."},
        {"text": "The heart of a shrimp is located in its head."},
    ]
)






while True:
    query=input("Enter your query:")

    if query.lower() in ("exit","quit"):
        break

    actual = table.search(query).limit(2).to_list()
    context_joined=",".join(context["text"] for context in actual)
    
    response = client.models.generate_content(
    model="gemini-3.5-flash",
    config=types.GenerateContentConfig(
       system_instruction="You are a helpful assistant that provides information based on the following context: " + context_joined),
    contents=f"""user: {query},
                context: {context_joined}
                answer: """

    )
    print(response.text)