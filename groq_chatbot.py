import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


SYSTEM_PROMPT = """
You are an AI Health and Lifestyle Assistant inside a Lifestyle Disease
Prediction application.

Your job is to explain the application's prediction results and provide
general lifestyle information in simple, understandable language.

Important rules:
- Do not claim to diagnose a disease.
- Do not replace a doctor or other qualified healthcare professional.
- Do not prescribe medicines or give medication dosages.
- Explain prediction results as estimates from a machine-learning model.
- Encourage professional medical advice when symptoms are serious,
  persistent, or concerning.
- Give practical, general lifestyle suggestions about nutrition,
  physical activity, sleep, stress management, and healthy habits.
- Be concise, friendly, and easy to understand.
"""


def chat_with_groq(message, context=None, history=None):
    """
    Send a user message to Groq with optional prediction context
    and previous conversation history.
    """

    if not client:
        return (
            "Groq is not configured yet. "
            "Please add your GROQ_API_KEY to the .env file."
        )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Here is the user's current prediction information. "
                    "Use it only to explain the result and provide general "
                    "lifestyle guidance:\n\n"
                    f"{context}"
                )
            }
        )

    if history:
        messages.extend(history[-10:])

    messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=700
        )

        return response.choices[0].message.content

    except Exception as error:
        print(f"Groq API error: {error}")

        return (
            "Sorry, I couldn't connect to the AI Health Assistant right now. "
            "Please check your Groq API configuration and try again."
        )