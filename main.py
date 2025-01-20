import os

import pymupdf
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class QualificationResponse(BaseModel):
    expert_name: str = Field(description="Name of the expert.")
    is_qualified: bool = Field(
        description="Is the candidate qualified for the job and does he meet the requirements for the position. Yes or no."
    )
    reasonings: str = Field(
        description="Give a short (maximum 50 words) but precise explanation of why the candidate is qualified for the position."
    )
    years_of_relevant_experience: int = Field(
        description="Calculate the candidates years of experience in relevant the relevant fields for the job. Do not sum up experience in the same field in that occurs in the same time period."
    )
    suitability_score: float = Field(
        description="Estimate how qualified and suited the candidate is for the job. Consider relevant criteria that were submitted with the prompt. Range the suitability between zero (lowest) to 100 (highest)"
    )


position_input = input("Describe the position: ")
industry_input = input("Which industry is relevant for the project? ")
experience_input = input("How many years of experience in the industry are required? ")
optional_input = input("Would you like to add any additional information? ")

prompt = f"Position: {position_input}; Industry: {industry_input}; Work experience: {experience_input}; Optional: {optional_input}"

# prompt = input("What kind of expert are you looking for? ")
results = []
all_cvs = []  # nested list -> includes list objects for every single cv

for cv in os.listdir("cvs"):
    pages = []
    pdf = pymupdf.open(os.path.join("cvs", cv))

    for page in pdf:
        pages.append(page.get_text())

    client = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    client = client.with_structured_output(QualificationResponse, method="json_mode")

    parser = PydanticOutputParser(pydantic_object=QualificationResponse)

    prompt = f"Treat the following prompt as criteria to look for an expert for an open job {prompt}. Rate the following CV if it is matches the criteria: {'/n/n'.join(pages)}. Respond in JSON with the following structure: {parser.get_format_instructions()}"
    chat_completion = client.invoke(prompt)
    # print(prompt)
    if chat_completion.is_qualified:
        results.append(chat_completion)

top_results = sorted(results, key=lambda x: x.suitability_score, reverse=True)[:5]

for result in top_results:
    result = result.__dict__
    print(f"Expert Name: {result['expert_name']}")
    print(f"Qualified: {'Yes' if result['is_qualified'] else 'No'}")
    print(f"Reasonings: {result['reasonings']}")
    print(f"Years of Relevant Experience: {result['years_of_relevant_experience']}")
    print(f"Suitability Score: {result['suitability_score']}/100")
    print("-" * 50)
