import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Test GenAI client
try:
    from google import genai as genai_pkg
    api_key = os.environ.get('GENAI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if api_key:
        client = genai_pkg.Client(api_key=api_key)
        print("✓ GenAI client initialized successfully")
    else:
        print("✗ No API key found")
        exit(1)
except Exception as e:
    print(f"✗ GenAI client initialization failed: {e}")
    exit(1)

def test_conversation_context():
    # Test data
    question = "What is AI?"
    conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "Can you explain machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI that allows systems to learn from data."}
    ]

    # Build conversation context
    context = ""
    if conversation_history and len(conversation_history) > 0:
        context = "\n\nPrevious conversation:\n"
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if content.strip():  # Only include non-empty messages
                context += f"{role.title()}: {content}\n"

    print(f"Question: {question}")
    print(f"Conversation history length: {len(conversation_history)}")
    print(f"Context built: {bool(context.strip())}")
    print(f"Context:\n{context}")

    prompt = f"""You are a helpful AI tutor. You have access to the conversation history below.

{context}
Current question: {question}

Please provide a helpful, contextual response that builds on the previous conversation if applicable. If this is the first message or there's no relevant context, just answer the question directly. Keep your answer clear and educational."""

    print(f"\nPrompt:\n{prompt}")

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={
                "temperature": 0.1,
                "top_p": 0.5,
                "max_output_tokens": 300,
            }
        )

        # Extract answer
        answer = None
        if hasattr(response, 'text') and response.text:
            answer = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content
                    if hasattr(content, 'parts') and content.parts:
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                answer = part.text
                                break
                        if answer:
                            break
                    elif hasattr(content, 'text') and content.text:
                        answer = content.text
                        break

        if answer:
            print(f"\n✓ AI Response: {answer}")
        else:
            print("\n✗ No answer extracted from response")

    except Exception as e:
        print(f"\n✗ Error generating response: {e}")

if __name__ == "__main__":
    test_conversation_context()
