# services/parser.py
import json
from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def parse_incident_text(text: str, operator_name: str, machine_id: str, context: dict = None):
    """
    Uses Groq Llama 3 to parse transcript into a structured JSON object.
    If context is provided, it uses the new text to fill in the blanks.
    """
    if context:
        # We have existing context, so we are filling in the blanks.
        system_prompt = f"""
        You are an AI assistant helping an operator complete an incident report.
        Here is the partially filled report:
        {json.dumps(context, indent=2)}

        The operator has provided new information in the following transcript.
        Your task is to extract the missing information from the new transcript and use it to update the report.
        Return the single, final, and complete JSON object. Do not change the existing values unless the new transcript explicitly corrects them.

        New Transcript: "{text}"

        Return only the final, complete JSON object.
        """
        user_content = "Based on the new transcript, please complete the incident report."

    else:
        # This is the first pass, parsing from scratch.
        system_prompt = f"""
        You are an expert AI assistant for Caterpillar machinery incident logging.
        Your task is to parse the user's transcribed voice note into a structured JSON object.

        The operator is '{operator_name}' and the machine ID is '{machine_id}'. These are already known.

        From the transcript, extract the following fields:
        - "location": Where the incident occurred.
        - "incident_type": Classify into one of: "Mechanical", "Electrical", "Hydraulic", "Safety", "Operational", "Warning Light", "Unknown".
        - "severity": Classify into one of: "Low", "Medium", "High", "Critical".
        - "description": A concise summary of what the operator said happened.
        - "actions_taken": What the operator did in response.

        RULES:
        1. Return ONLY a valid JSON object.
        2. If information for a field is not in the transcript, set its value to `null`.
        3. The final JSON should include the known operator_name and machine_id.
        """
        user_content = f"Here is the incident transcript: \"{text}\""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model="llama3-70b-8192",
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Error during parsing: {e}")
        return {"error": "Failed to parse text with LLM."}