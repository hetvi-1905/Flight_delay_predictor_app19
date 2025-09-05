# models/llm_explainer.py
from groq import Groq

# Hardcoded Groq API key (replace with your actual key)
GROQ_API_KEY = "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"

# Initialize client
client = Groq(api_key=GROQ_API_KEY)

def generate_explanation(carrier_name, airport_name, target_month, avg_delay, breakdown):
    # Prepare breakdown in readable format
    breakdown_str = ", ".join([f"{k} ({v}%)" for k, v in breakdown.items()])

    # Prompt for Groq LLM
    prompt = f"""
    You are an aviation delay analysis assistant. 
    Summarize forecast results for passengers and airlines.

    Carrier: {carrier_name}
    Airport: {airport_name}
    Month: {target_month}
    Avg Delay: {avg_delay} minutes
    Breakdown: {breakdown_str}

    Write a clear explanation highlighting main contributors to the delay.
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You explain flight delays in simple terms."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=400
    )

    return response.choices[0].message.content.strip()
