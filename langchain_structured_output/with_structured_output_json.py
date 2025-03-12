from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Annotated, Literal

load_dotenv()

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Define schema using json
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


# Generate structured output using Pydantic schema directly
structured_output = model.with_structured_output(json_schema)

# Invoke the model
result = structured_output.invoke("""
Overview
The Samsung Galaxy S24 Ultra is a powerhouse smartphone that continues Samsung's legacy 
of premium devices. Packed with cutting-edge features, a stunning display, and powerful 
performance, this phone is designed for tech enthusiasts and professionals alike. 
But is it worth the high price tag? Let’s dive into the details.
                                  
Pros:
Gorgeous AMOLED Display: The 6.8-inch QHD+ Dynamic AMOLED 2X screen offers stunning colors, deep blacks, and a 120Hz adaptive refresh rate.
Outstanding Camera System: The 200MP primary sensor, improved telephoto lenses, and AI-enhanced photography produce exceptional shots in any lighting.                                

""")

# Print results using dot notation
print(result) 
print("\n")
