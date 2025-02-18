## Language packages
import json
import torch
import re
import csv
from typing import List, Dict, Any

## Make sure CUDA is available for this model.
assert torch.cuda.is_available(), "CUDA not available"

## 3rd Party packages
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
## Custom packages
from knowledgegraph import KnowledgeGraph

# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_NAME= "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    # token=hf_token
)
# model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    # token=hf_token
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2
    
)

llm = HuggingFacePipeline(pipeline=pipe)

class Relationship(BaseModel):
    source: str = Field(description="Name")
    target: str = Field(description="Name")
    type: str = Field(description="Type of relationship. E.g.: Friend, colleague, relative")
    
class Relationships(BaseModel):
    relationships: List[Relationship]

parser = PydanticOutputParser(pydantic_object=Relationships)

format_instructions = """
Adhere strictly to the following JSON format:
{
  "relationships": [
      { "source": "Name of source", "target": "Name of target", "type": "Type of the relationship"},
      { "source": "Name of source", "target": "Name of target", "type": "Type of the relationship"}      
  ]
}
"""

class BookWorm:
    def __init__(self):
        self.kg = KnowledgeGraph()
        self.kg.clear()
        self.llm = llm

        messages = [
            {"role": "assistant", "content": "You are an expert on reading novels. Read the given context and answer the question of the user. Use only the context provided."},
            {"role": "user", "content": "Find all people and their relationship to each other.\n\n{format_instructions}\n\n{context}\n\n"}
        ]
        template = tokenizer.apply_chat_template(messages, tokenize=False)
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context"],
            # partial_variables={"format_instructions": parser.get_format_instructions()}
            partial_variables={"format_instructions": format_instructions}
        )

        self.relation_chain = self.prompt | self.llm.bind(skip_prompt=True)

    def extract_relationships(self, context: str) -> List[Dict[str, str]]:
        output = self.relation_chain.invoke(input={"context": context})
            
        relationships = []

        try:
            results = re.findall(r'```json(.*?)```', output, re.DOTALL)

            if results:
                # relationships = parser.parse(results[-1])
                data = json.loads(results[-1])
                relationships = data.get("relationships", [])

                for relation in relationships:
                    source = relation["source"]
                    target = relation["target"]
                    relationship = relation["type"]
                    
                    self.kg.add_node(source, {"name": source})
                    self.kg.add_node(target, {"name": target})
                    self.kg.add_edge(source, target, relationship)

        except Exception as e:
            with open('data/outputs.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([context, output, e])
            
            print("Error: Could not parse LLM response as JSON\n")
            return []

        return relationships
    
    def graph_data(self):
        return self.kg.dump()
        
    def graph_summary(self) -> Dict[str, Any]:
        return {
            "node_count": len(self.kg.nodes),
            "edge_count": sum(len(edges) for edges in self.kg.edges.values()),
            "data": self.kg.dump()
        }