import asyncio
import time
import base64
from PIL import Image
import io
import os
from prompts import patient_info_prompt
from ollama_class import OllamaIntegration


async def execute_patient_prompt(ollama_instance, image_path):
    """
    Test function to process an image using Ollama's llama3.2-vision model and extract patient information.

    :param ollama_instance: Instance of OllamaIntegration class.
    :param image_path: Path to the image file (e.g., PNG or JPEG).
    :return: Parsed Python dictionary if successful, None if an error occurs.
    """
    try:
        # Verify the image file exists and is readable
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' does not exist.")
            return None
        if not os.access(image_path, os.R_OK):
            print(f"Error: No read permission for image file '{image_path}'.")
            return None

        # Validate and encode the image file
        try:
            with Image.open(image_path) as img:
                # Convert image to RGB if necessary (e.g., for PNG with transparency)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Save image to a bytes buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                # Encode to base64
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error: Invalid or corrupted image file '{image_path}': {str(e)}")
            return None

        # Set the master prompt for patient information extraction
        ollama_instance.set_master_prompt(patient_info_prompt)

        # Generate response with the base64-encoded image
        response = await ollama_instance.generate_response(images=[image_base64])

        # Parse the response to a dictionary
        patient_info_dict = ollama_instance.parse_response_to_dict(response)

        # Print results
        print("Raw Response:")
        print(response)
        if patient_info_dict:
            print("Parsed Patient Info Dictionary:")
            print(patient_info_dict)
            print("Type:", type(patient_info_dict))
        else:
            print("Failed to parse dictionary; check raw response above.")

        return patient_info_dict

    except PermissionError as e:
        print(f"Error: Permission denied accessing '{image_path}': {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing image or calling Ollama: {str(e)}")
        return None


if __name__ == "__main__":
    # Initialize the OllamaIntegration class with the llama3.2-vision model
    ollama_integration = OllamaIntegration(host="http://localhost:11434")

    # Path to the test image
    test_image_path = "./test_patient_info.png"  # Ensure this path is correct

    # Measure execution time
    start = time.time()

    # Run the test
    result = asyncio.run(execute_patient_prompt(ollama_integration, test_image_path))

    # Calculate and print execution time
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
