import argparse

# Import the CRAG class from your modularized codebase
from src.crag import CRAG  # Adjust the import path if your CRAG class is in a different module
from dotenv import load_dotenv
load_dotenv()


def validate_args(args):
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")
    if args.temperature < 0 or args.temperature > 1:
        raise ValueError("temperature must be between 0 and 1.")
    return args

def parse_args():
    parser = argparse.ArgumentParser(description="CRAG Process for Document Retrieval and Query Answering.")
    parser.add_argument("--file_path", type=str, default="data/Understanding_Climate_Change (1).pdf",
                        help="Path to the document file to encode (supports PDF, TXT, CSV, JSON, DOCX, XLSX).")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash",
                        help="Language model to use (default: gemini-1.5-flash).")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Maximum tokens to use in LLM responses (default: 1000).")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature to use for LLM responses (default: 0).")
    parser.add_argument("--query", type=str, default="What are the main causes of climate change?",
                        help="Query to test the CRAG process.")
    parser.add_argument("--lower_threshold", type=float, default=0.3,
                        help="Lower threshold for score evaluation (default: 0.3).")
    parser.add_argument("--upper_threshold", type=float, default=0.7,
                        help="Upper threshold for score evaluation (default: 0.7).")
    parser.add_argument("--web_search_enabled", type=str, default=None,
                        help="Set to 'true' to enable web search, 'false' to disable, or leave unset to use env var.")

    args = parser.parse_args()
    # Convert web_search_enabled to bool or None
    if args.web_search_enabled is not None:
        val = args.web_search_enabled.strip().lower()
        if val == 'true':
            args.web_search_enabled = True
        elif val == 'false':
            args.web_search_enabled = False
        else:
            args.web_search_enabled = None
    return validate_args(args)

def main(args):
    crag = CRAG(
        file_path=args.file_path,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold,
        web_search_enabled=args.web_search_enabled
    )
    response = crag.run(args.query)
    print(f"Query: {args.query}")
    print(f"Answer: {response}")

if __name__ == "__main__":
    main(parse_args())