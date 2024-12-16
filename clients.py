import os
import weaviate
import openai

from weaviate.classes.init import Auth
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv

load_dotenv()

# OPEN AI
api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = "https://api.openai.com/v1"
model_kwargs = {
    "model": "gpt-4o-mini",
    "temperature": 1.2,
    "max_tokens": 500
}
smol_model_kwargs = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_tokens": 200
}

# ANTHROPIC
# api_key = os.getenv("ANTHROPIC_API_KEY")
# endpoint_url = "https://api.anthropic.com/v1"
# model_kwargs = {
#     "model": "claude-3-5-sonnet-20241022",
#     "temperature": 0.7,
#     "max_tokens": 500,
# }

# RUNPOD
# runpod_serverless_id = os.getenv("RUNPOD_SERVERLESS_ID")
# endpoint_url = f"https://api.runpod.ai/v2/{runpod_serverless_id}/openai/v1"
# model_kwargs = {
#     "model": "mistralai/Mistral-7B-Instruct-v0.3",
#     "temperature": 0.3,
#     "max_tokens": 500
# }

# Initialize clients
wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=Auth.api_key(wcd_api_key),
    skip_init_checks=True
)

openai_client = wrap_openai(openai.AsyncClient(api_key=api_key, base_url=endpoint_url))


__all__ = ['weaviate_client', 'openai_client', 'smol_model_kwargs', 'model_kwargs']