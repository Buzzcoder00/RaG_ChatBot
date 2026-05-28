from dotenv import load_dotenv
load_dotenv()

from google import genai

client = genai.Client()


while True:
    query=input("Enter your query:")

    if query.lower() in ("exit","quit"):
        break
    
    response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=query
    )
    print(response.text)