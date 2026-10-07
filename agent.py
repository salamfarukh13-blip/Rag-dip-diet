"""
agent.py
Dual-LLM RAG pipeline:
  1. IBM Granite  → DIP math analysis + compliance check
  2. Groq (openai/gpt-oss-120b) → structured, conversational response
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

IBM_APIKEY     = os.getenv("IBM_WATSONX_APIKEY", "")
IBM_PROJECT_ID = os.getenv("IBM_WATSONX_PROJECT_ID", "")
IBM_URL        = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")


# ─────────────────────────────────────────────
#  IBM Granite — DIP math & compliance check
# ─────────────────────────────────────────────
def call_ibm_granite(prompt: str) -> str:
    """
    Call IBM Granite via ibm-watsonx-ai.
    Falls back gracefully if credentials are not configured.
    """
    if not IBM_APIKEY or not IBM_PROJECT_ID:
        return "[IBM Granite] Credentials not configured — skipping Granite analysis step."

    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        model = ModelInference(
            model_id="ibm/granite-13b-instruct-v2",
            credentials={"apikey": IBM_APIKEY, "url": IBM_URL},
            project_id=IBM_PROJECT_ID,
            params={
                GenParams.MAX_NEW_TOKENS: 512,
                GenParams.TEMPERATURE: 0.3,
                GenParams.STOP_SEQUENCES: ["\n\n\n"],
            },
        )
        response = model.generate_text(prompt=prompt)
        return response.strip() if response else "[IBM Granite] Empty response."
    except Exception as exc:
        return f"[IBM Granite Error] {exc}"


# ─────────────────────────────────────────────
#  Groq — structured, conversational formatting
# ─────────────────────────────────────────────
def call_groq_model(granite_analysis: str, user_query: str) -> str:
    """
    Send Granite's raw analysis to Groq's openai/gpt-oss-120b for a polished reply.
    """
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    system_prompt = (
        "You are a friendly, expert DIP Diet nutritionist assistant. "
        "You receive a raw technical analysis from IBM Granite and the user's original question. "
        "Your job is to format the analysis into a clean, warm, highly structured response. "
        "Use clear headings, bullet points, and bold text where helpful. "
        "Always include the calculated food quantities with exact gram values. "
        "Be encouraging and practical."
    )

    user_message = (
        f"## User Question\n{user_query}\n\n"
        f"## IBM Granite Technical Analysis\n{granite_analysis}\n\n"
        "Please format this into a clear, helpful, and well-structured response for the user."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.6,
        max_tokens=1024,
    )
    return completion.choices[0].message.content.strip()


# ─────────────────────────────────────────────
#  Primary RAG Agent Pipeline
# ─────────────────────────────────────────────
def run_rag_agent(user_query: str, weight: float, conditions: str, context: str) -> str:
    """
    Full pipeline:
      1. Build a structured prompt with RAG context + user profile.
      2. Send to IBM Granite for DIP math analysis.
      3. Send Granite output to Groq for a polished, conversational reply.
    """
    fruit_grams  = weight * 10
    veggie_grams = weight * 5

    granite_prompt = f"""You are a DIP Diet expert assistant. Use the following knowledge context to answer the user's question accurately.

=== DIP DIET KNOWLEDGE CONTEXT ===
{context}

=== USER PROFILE ===
- Body Weight: {weight} kg
- Health Conditions: {conditions if conditions else 'None mentioned'}
- Calculated Daily Targets:
  * Morning Fruits (Breakfast): {fruit_grams:.0f} grams (formula: {weight} kg × 10)
  * Raw Vegetables per meal (Plate 1): {veggie_grams:.0f} grams (formula: {weight} kg × 5)

=== USER QUESTION ===
{user_query}

=== YOUR TASK ===
1. Calculate exact quantities using the DIP formulas above.
2. Check compliance with DIP rules (timing, food types, plate order).
3. Provide specific recommendations based on the user's health conditions.
4. Highlight any DIP rules that are relevant to this query.

Answer:"""

    print("[agent] Calling IBM Granite for DIP analysis...")
    granite_output = call_ibm_granite(granite_prompt)
    print(f"[agent] Granite output length: {len(granite_output)} chars")

    print("[agent] Sending to Groq for structured formatting...")
    final_response = call_groq_model(granite_output, user_query)
    print("[agent] Pipeline complete.")

    return final_response
