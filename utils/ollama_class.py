import os
from ollama import AsyncClient
import ast
import re


class OllamaIntegration:
    def __init__(self, model="gemma3:4b", host=None, master_prompt=""):
        """
        Initialize the OllamaIntegration class for interacting with the Ollama server.

        :param model: The Ollama model to use (default: "llama3.2-vision:11b" for image processing).
        :param host: The Ollama server host URL. Defaults to OLLAMA_HOST env variable or "http://localhost:11434".
        :param master_prompt: The base prompt used for all requests, customizable for different image types.
        """
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        # Master prompt can be updated to tailor instructions for different image types
        self.master_prompt = master_prompt
        # Initialize async client for non-blocking API calls
        self.client = AsyncClient(host=self.host)

    def set_master_prompt(self, prompt):
        """
        Update the master prompt to customize instructions for different image types.

        The master prompt defines core instructions for the Ollama model, e.g., extracting text from
        patient evaluation tables. It can be updated dynamically to handle various image layouts.

        :param prompt: The new master prompt to use for text extraction.
        """
        self.master_prompt = prompt

    async def generate_response(self, additional_data=None, images=None, stream=False):
        """
        Generate a response using the Ollama model, supporting both text and image inputs.

        This method combines the master prompt with optional additional data (text) and images (base64-encoded).
        For vision models like llama3.2-vision, images should be provided as a list of base64 strings.
        The master_prompt can be updated via set_master_prompt to handle different image types.

        :param additional_data: Optional text data to append to the prompt (e.g., metadata).
        :param images: Optional list of base64-encoded images for vision models.
        :param stream: If True, stream the response; if False, return the complete response.
        :return: The generated response as a string, or None if streaming.
        """
        # Combine master prompt with additional data if provided
        full_prompt = self.master_prompt
        if additional_data:
            if isinstance(additional_data, dict):
                additional_data = str(additional_data)
            full_prompt += f"\n\nAdditional data:\n{additional_data}"

        if stream:
            await self._stream_and_print_response(full_prompt, images)
            return None
        else:
            return await self._generate_response(full_prompt, images)

    async def _generate_response(self, prompt, images=None):
        """
        Generate a complete response asynchronously.

        :param prompt: The combined prompt (master_prompt + additional_data).
        :param images: Optional list of base64-encoded images for vision models.
        :return: The extracted text as a string.
        """
        try:
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                images=(
                    images if images else []
                ),  # Ensure images is a list, even if None
            )
            return response.get("response", "")
        except Exception as e:
            print(f"Error in Ollama API call: {str(e)}")
            return None

    async def _stream_and_print_response(self, prompt, images=None):
        """
        Stream the response chunk by chunk and print it.

        :param prompt: The combined prompt (master_prompt + additional_data).
        :param images: Optional list of base64-encoded images for vision models.
        """
        try:
            async for chunk in await self.client.generate(
                model=self.model,
                prompt=prompt,
                images=images if images else [],
                stream=True,
            ):
                print(chunk["response"], end="", flush=True)
            print()  # Newline after streaming
        except Exception as e:
            print(f"Error streaming Ollama response: {str(e)}")

    def parse_response_to_dict(self, response):
        """
        Parse an Ollama response to extract a Python dictionary, removing system messages and Markdown.

        The response may contain a dictionary wrapped in text (e.g., "Here is the Python dictionary...")
        and Markdown code blocks (```python ... ```). This method extracts the dictionary string and
        converts it to a Python dictionary.

        :param response: The raw response string from the Ollama model.
        :return: A Python dictionary if successful, None if parsing fails.
        """
        if not response or not isinstance(response, str):
            print("Error: Empty or invalid response")
            return None

        try:
            # Extract the dictionary string using regex
            # Match content between ```python and ```, or standalone dictionary
            dict_pattern = r"\{[\s\S]*?\}"
            match = re.search(dict_pattern, response)
            if not match:
                print("Error: No dictionary found in response")
                return None

            dict_str = match.group(0)

            # Parse the dictionary string to a Python dictionary
            parsed_dict = ast.literal_eval(dict_str)
            if not isinstance(parsed_dict, dict):
                print("Error: Parsed response is not a dictionary")
                return None

            return parsed_dict

        except (ValueError, SyntaxError) as e:
            print(f"Error: Failed to parse dictionary from response: {str(e)}")
            print("Raw response:", response)
            return None
