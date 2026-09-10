## **Note:** 
This folder contains all the learnings, research and practical handson work related to a Small Language Model for different use cases.

Temp:

# Context for LLM: SLM Fine-Tuning Data Pipeline (ADX to JSONL)

## Project Goal
I am building a pipeline to fine-tune a Small Language Model (SLM), specifically `sarvamai/sarvam-1` (2B parameters), using **Unsloth** and **Hugging Face**. The goal of the SLM is to perform intent extraction and search query building from multi-turn customer-bot conversations in various Indian languages (and code-mixed languages like Hinglish).

## Current Objective
I need a robust Python script to fetch raw data from **Azure Data Explorer (ADX)**, transform it into the specific chat-template format required for Supervised Fine-Tuning (SFT), filter out unsafe lengths, clean the JSON, and save it as a `.jsonl` file.

## Data Source Details (Azure Data Explorer)
*   **Authentication:** I will connect to ADX using an Azure App Registration (Client ID and Client Secret).
*   **Library to use:** `azure-kusto-data` (and its pandas integration).
*   **Placeholders:** Please use placeholders like `<ADX_CLUSTER_URI>`, `<ADX_DATABASE>`, `<ADX_TABLE>`, `<CLIENT_ID>`, and `<CLIENT_SECRET>` in the code so I can fill them in.

## Raw Data Structure (Assumed from ADX)
Assume the ADX table has at least two main columns:
1.  `conversation_text`: A single string containing the multi-turn chat, separated by newlines (e.g., `"Customer: Hello\nBot: Hi\nCustomer: I need a loan"`).
2.  `extracted_json`: A string or JSON object containing the LLM's extracted output. *(Note: The raw data might contain token count fields like `retrival_input_tokens`. The script MUST strip these out, as the SLM should not be trained to predict token counts).*

## Target Output Format (Crucial for Unsloth/SFTTrainer)
The final output must be a `.jsonl` file where every line is a valid JSON object representing one training example. It must strictly follow this structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert intent extraction assistant. Analyze the conversation and output ONLY valid JSON."
    },
    {
      "role": "user",
      "content": "<The entire multi-turn conversation string goes here as a SINGLE message>"
    },
    {
      "role": "assistant",
      "content": "<The cleaned, pretty-printed JSON string goes here>"
    }
  ]
}
