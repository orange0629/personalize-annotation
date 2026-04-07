#!/usr/bin/env python3
import json
import os
import sys
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config


def make_bedrock_client(region: str):
    cfg = Config(
        connect_timeout=30,     # Connection timeout
        read_timeout=300,       # Read timeout, in seconds, e.g. 300 = 5 minutes
        retries={"max_attempts": 5, "mode": "standard"},
    )
    return boto3.client("bedrock-runtime", region_name=region, config=cfg)


def converse_once(
    prompt: str,
    model_id: str,
    region: str,
    system: Optional[str] = None,
    max_tokens: int = 51200,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    client = make_bedrock_client(region)

    kwargs = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    }

    if system:
        kwargs["system"] = [{"text": system}]

    resp = client.converse(**kwargs)

    # Compatible with the standard response structure
    content = resp["output"]["message"]["content"]
    texts = [c.get("text", "") for c in content if isinstance(c, dict)]
    return "".join(texts).strip()


def main():
    region = "us-east-1"
    model_id = "deepseek.v3-v1:0"

    prompt = (
        "Count from 1 to 10000. Do not give me code. "
        "Follow my instructions and give me plain text output. "
        "Output at least 20000 tokens before you stop. "
        "You must do this. You do not need to confirm with me."
    )
    system = "You are a helpful assistant."

    try:
        out = converse_once(
            prompt=prompt,
            model_id=model_id,
            region=region,
            system=system,
            max_tokens=51200,
            temperature=0.3,
            top_p=0.9,
        )
        print(out)
        print(len(out.split(" ")))

    except NoCredentialsError:
        print(
            "AWS credentials not found. Please configure AWS_PROFILE or "
            "environment variables such as AWS_ACCESS_KEY_ID first.",
            file=sys.stderr,
        )
        sys.exit(1)

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        msg = e.response.get("Error", {}).get("Message", str(e))
        print(f"AWS call failed: {code}\n{msg}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
