import csv
import os

import pymupdf
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

rate_limiters = InMemoryRateLimiter(
    requests_per_second=0.05, check_every_n_seconds=0.1, max_bucket_size=10
)


class ExpertiseArea(BaseModel):
    area_of_expertise: str = Field(
        description="Best suitable area of expertise chosen from provided options."
    )


# these categories have to be carefully selected and defined
expertise_areas = [
    "Financial Industry",
    "Healthcare",
    "Beatuy",
    "Automotive",
    "Education",
    "Engineering",
    "Public Sector",
]


results = {}

for cv in os.listdir("cvs"):
    pdf = pymupdf.open(os.path.join("cvs", cv))

    pages = []

    for page in pdf:
        pages.append(page.get_text())

    client = ChatGroq(temperature=0, rate_limiter=rate_limiters)
    client = client.with_structured_output(ExpertiseArea, method="json_mode")

    parser = PydanticOutputParser(pydantic_object=ExpertiseArea)

    chat_completion = client.invoke(
        f"What is the area of expertise for this candidate {'/n/n'.join(pages)}. Choose from given options {', '.join(expertise_areas)}. Answer should only contain the chosen option. Respond in JSON with the following structure: {parser.get_format_instructions()}"
    )

    results[cv] = chat_completion

with open("cv_categories.csv", "w+", newline="") as f:
    w = csv.DictWriter(f, results.keys())
    w.writeheader()
    w.writerow(results)
