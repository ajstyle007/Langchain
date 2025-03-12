from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Annotated, Literal

load_dotenv()

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Define schema using Pydantic BaseModel (not TypedDict)
class Review(BaseModel):

    Device : str

    key_themes : list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    Summary : str = Field(description="A brief summary of review")
    Sentiment : Literal["pos", "neg"] = Field(description="return sentiment whether it is negative, positive or neutral")
    pros : Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons : Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name : Optional[str] = Field(default=None, description="Write the name of the reviewer")

# Generate structured output using Pydantic schema directly
structured_output = model.with_structured_output(Review)

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
print(result)  # Prints the structured Pydantic object
print("\n")
print(result.Device)
print(result.name)
print(result.key_themes)
print(result.Summary)
print(result.Sentiment)
print(result.pros)
print(result.cons)
