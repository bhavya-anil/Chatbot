from langchain_groq import ChatGroq
from pypdf import PdfReader
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Get API key from .env
api_key = os.getenv("GROQ_API_KEY")

# ---------- Load PDF ----------
reader = PdfReader("file.pdf")

pdf_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text

pdf_text = pdf_text[:8000]

# ---------- LLM Setup ----------
llm = ChatGroq(
    api_key=api_key,
    model_name="groq/compound"
)

# ---------- System Prompt ----------
history = [
    {
        "role": "system",
        "content": f"""
Answer using only the PDF content.
If not found, say 'Not found in PDF'.

PDF Content:
{pdf_text}
"""
    }
]

print("PDF Chatbot Ready! Type 'exit' to quit.\n")

# ---------- Chat Loop ----------
while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    history.append({"role": "user", "content": user_input})
    response = llm.invoke(history)
    history.append({"role": "assistant", "content": response.content})

    print("\nAI:", response.content, "\n")
