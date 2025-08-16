import logging
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NOTE: this will work great with ChatGPT, depending on the model, the prompt will need to change.
SYSTEM_PROMPT = """
You are a Content Analyst and a Q&A Data Generator. Your sole task is to analyze provided text chunks and generate a list of question-and-answer pairs based *only* on the information present in each chunk.

Your primary objective is to create a high-quality dataset for a knowledge base, focusing on extracting specific, factual information.

**Core Instructions:**
1.  Read and thoroughly understand the provided `[CHUNKED_TEXT]`.
2.  Identify all key concepts, facts, and relationships within the text.
3.  For each key concept, generate multiple relevant and non-redundant question-and-answer pairs. Strive to generate as many questions as possible.
4.  **Critical Constraint:** The questions generated must specifically emphasize and relate to the topic of **[TOPIC_OF_EMPHASIS]**. Prioritize creating questions that are directly relevant to this topic.
5.  Ensure the `answer` is a concise, direct, and complete response found entirely within the provided text.
6.  Format the final output as a single, valid JSON array of objects.

**Output Format Specification:**
- The output MUST be a JSON array.
- Each element in the array MUST be a JSON object with two keys: `"question"` and `"answer"`.
- The values for both keys must be strings.
- Do NOT include any additional text, explanations, or code outside of the final JSON array.

**Example Input/Output (for reference):**
* This section is crucial for a technical task like this. Insert a few examples here to demonstrate the exact format.
* **Example Input:**
    ```text
    [CHUNKED_TEXT]
    The Roman Republic was a period of ancient Roman civilization beginning with the overthrow of the Roman Kingdom in 509 BCE and ending in 27 BCE with the establishment of the Roman Empire. Its government was characterized by elected magistrates and a complex system of checks and balances. The city of Rome itself was founded in 753 BCE.
    ```
* **Example Output (with [TOPIC_OF_EMPHASIS] = "Government"):**
    ```json
    [
      {
        "question": "What was the system of government during the Roman Republic?",
        "answer": "Its government was characterized by elected magistrates and a complex system of checks and balances."
      }
    ]
    ```

**Ready to begin. Here is your text:**

@@CHUNKED_TEXT@@

---
**Here is [TOPIC_OF_EMPHASIS]:**
**@@TOPIC_OF_EMPHASIS@@**
"""


def get_system_prompt(chunk: str, topic: str) -> str:
    """
    Generates a system prompt by replacing placeholders in a predefined template.

    Args:
        chunk (str): The text content to insert into the system prompt
        topic (str): The topic to emphasize, inserted into the system prompt (e.g. 2025 Tesla Cybertruck)

    Returns:
        str: The system prompt string with the placeholders replaced by the provided chunk and topic.
    """
    logging.info(f"Generating system prompt for topic: {topic}")
    prompt = SYSTEM_PROMPT
    prompt = prompt.replace("@@CHUNKED_TEXT@@", chunk)
    prompt = prompt.replace("@@TOPIC_OF_EMPHASIS@@", topic)
    return prompt


def get_openai_completion(
    client: openai.Client,
    chunk: str,
    topic: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 150,
) -> str:
    """
    Connects to the OpenAI LLM API, sends a prompt, and returns the generated text.

    Args:
        prompt (str): The text prompt to send to the model.
        model (str): The name of the model to use (e.g., "gpt-3.5-turbo", "gpt-4o").
        temperature (float): Controls the randomness of the output. Higher values
                             (e.g., 0.8) make the output more random, while lower
                             values (e.g., 0.2) make it more focused and deterministic.
        max_tokens (int): The maximum number of tokens (words/characters) in the
                          generated response.

    Returns:
        str: The generated text from the LLM, or an error message if the API call fails.
    """
    logging.info(f"Getting OpenAI completion with model: {model}")
    prompt = get_system_prompt(chunk, topic)

    try:
        # Call the chat completions API
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Check if the response is valid and return the content
        if response.choices and response.choices[0].message:
            logging.info("Successfully received response from OpenAI API.")
            return response.choices[0].message.content.strip()  # type: ignore
        else:
            logging.error("Error: Empty or invalid response from the API.")
            raise Exception("Error: Empty or invalid response from the API.")
    except Exception as e:
        logging.error(f"An error occurred during OpenAI API call: {e}")
        raise e
