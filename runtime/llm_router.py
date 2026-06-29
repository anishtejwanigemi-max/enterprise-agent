from typing import Optional
from crewai import LLM
from security.secrets import get_secret
from utils.logger import get_logger

logger = get_logger("runtime.llm_router")

# ---------------------------------------------------------------------------
# Provider routing table
# ---------------------------------------------------------------------------
# Supported model string formats:
#   azure/gpt-4o               → Azure OpenAI (standard cognitiveservices endpoint)
#   openai/gpt-4o              → OpenAI direct
#   anthropic/claude-3-5-sonnet-20241022 → Anthropic
#   google/gemini-1.5-pro      → Google Gemini
#   gemini/gemini-1.5-pro      → Google Gemini (alias)
#   ollama/llama3              → Local Ollama
#   groq/llama3-70b-8192       → Groq
#   mistral/mistral-large      → Mistral AI
#   cohere/command-r-plus      → Cohere
#   huggingface/<model>        → HuggingFace Inference API
#   bedrock/anthropic.claude…  → AWS Bedrock
# ---------------------------------------------------------------------------


def _build_azure_llm(model_string: str) -> LLM:
    """
    Builds a CrewAI LLM for a standard Azure OpenAI endpoint
    (cognitiveservices.azure.com).

    Uses the 'openai' provider (OpenAICompletion) with api_base pointing at the
    Azure deployment URL. The api-version is passed as a default_query parameter
    inside additional_params so the OpenAI SDK appends it as a query string on
    every request — exactly what Azure OpenAI expects.

    This avoids crewai[azure-ai-inference] entirely (that SDK is only for Azure
    AI Foundry / ai.azure.com endpoints, NOT cognitiveservices.azure.com).

    Required env vars:
        AZURE_OPENAI_API_KEY     – your Azure resource key
        AZURE_OPENAI_ENDPOINT    – e.g. https://ai-native-dev-llm.cognitiveservices.azure.com
        AZURE_OPENAI_DEPLOYMENT  – deployment name (e.g. gpt-4o); auto-derived
                                   from the model slug after '/' if not set
        AZURE_OPENAI_API_VERSION – defaults to 2024-08-01-preview

    Usage in agents.yaml:  llm: azure/gpt-4o
    """
    api_key = get_secret("AZURE_OPENAI_API_KEY")
    endpoint = (
        get_secret("AZURE_OPENAI_ENDPOINT")
        or "https://ai-native-dev-llm.cognitiveservices.azure.com"
    )
    api_version = get_secret("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"

    parts = model_string.split("/", 1)
    model_slug = parts[1] if len(parts) > 1 else "gpt-4o"
    deployment = get_secret("AZURE_OPENAI_DEPLOYMENT") or model_slug

    endpoint = endpoint.rstrip("/")
    azure_base_url = f"{endpoint}/openai/deployments/{deployment}"

    logger.info(
        f"Azure OpenAI → base_url={azure_base_url}  api_version={api_version}"
    )

    # Pass default_query directly as a top-level kwarg.
    # LLM.__new__ forwards all kwargs straight to OpenAICompletion, which has
    # default_query as a proper Pydantic field. The OpenAI SDK client receives
    # it and appends ?api-version=... to every request URL — exactly what
    # Azure OpenAI requires.
    # Do NOT use api_version or additional_params — both end up in
    # Completions.create() which rejects them with "unexpected keyword argument".
    return LLM(
        model=f"openai/{deployment}",
        api_key=api_key,
        api_base=azure_base_url,
        default_query={"api-version": api_version},
    )


def get_llm(model_string: str) -> LLM:
    """
    Instantiates and returns a CrewAI-compatible LLM object.

    Falls back in this order for unknown/failing providers:
      1. Azure OpenAI (if AZURE_OPENAI_API_KEY is set)
      2. OpenAI direct (if OPENAI_API_KEY is set)
      3. Stub LLM (keeps process alive; inference calls will fail with a clear message)
    """
    logger.info(f"Routing LLM for model string: '{model_string}'")

    parts = model_string.split("/", 1)
    provider = parts[0].lower() if len(parts) > 1 else "openai"

    # ── Azure OpenAI ─────────────────────────────────────────────────────────
    if provider == "azure":
        try:
            return _build_azure_llm(model_string)
        except Exception as e:
            logger.error(f"Azure LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── OpenAI direct ────────────────────────────────────────────────────────
    if provider == "openai":
        api_key = get_secret("OPENAI_API_KEY")
        # No OpenAI key but Azure key present → transparently re-route
        if not api_key and get_secret("AZURE_OPENAI_API_KEY"):
            logger.info("No OPENAI_API_KEY found; routing to Azure OpenAI.")
            azure_model = f"azure/{parts[1]}" if len(parts) > 1 else "azure/gpt-4o"
            try:
                return _build_azure_llm(azure_model)
            except Exception as e:
                logger.error(f"Azure transparent fallback failed: {e}")
                return _fallback_llm(model_string)
        try:
            return LLM(model=model_string, api_key=api_key)
        except Exception as e:
            logger.error(f"OpenAI LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── Anthropic ────────────────────────────────────────────────────────────
    if provider == "anthropic":
        try:
            return LLM(model=model_string, api_key=get_secret("ANTHROPIC_API_KEY"))
        except Exception as e:
            logger.error(f"Anthropic LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── Google / Gemini ──────────────────────────────────────────────────────
    if provider in ("google", "gemini"):
        api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        try:
            return LLM(model=model_string, api_key=api_key)
        except Exception as e:
            logger.error(f"Google/Gemini LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── Ollama (local) ───────────────────────────────────────────────────────
    if provider == "ollama":
        base_url = get_secret("OLLAMA_BASE_URL") or "http://localhost:11434"
        try:
            return LLM(model=model_string, api_key="ollama", api_base=base_url)
        except Exception as e:
            logger.error(f"Ollama LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── Groq ─────────────────────────────────────────────────────────────────
    if provider == "groq":
        try:
            return LLM(model=model_string, api_key=get_secret("GROQ_API_KEY"))
        except Exception as e:
            logger.error(f"Groq LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── Mistral ──────────────────────────────────────────────────────────────
    if provider == "mistral":
        try:
            return LLM(model=model_string, api_key=get_secret("MISTRAL_API_KEY"))
        except Exception as e:
            logger.error(f"Mistral LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── Cohere ───────────────────────────────────────────────────────────────
    if provider == "cohere":
        try:
            return LLM(model=model_string, api_key=get_secret("COHERE_API_KEY"))
        except Exception as e:
            logger.error(f"Cohere LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── HuggingFace Inference API ────────────────────────────────────────────
    if provider == "huggingface":
        try:
            return LLM(model=model_string, api_key=get_secret("HUGGINGFACE_API_KEY"))
        except Exception as e:
            logger.error(f"HuggingFace LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── AWS Bedrock ──────────────────────────────────────────────────────────
    if provider == "bedrock":
        aws_region = get_secret("AWS_DEFAULT_REGION") or "us-east-1"
        try:
            return LLM(model=model_string, aws_region_name=aws_region)
        except Exception as e:
            logger.error(f"Bedrock LLM init failed: {e}")
            return _fallback_llm(model_string)

    # ── Unknown provider ─────────────────────────────────────────────────────
    logger.warning(f"Unknown provider '{provider}' — attempting smart fallback.")
    return _fallback_llm(model_string)


def _fallback_llm(original_model: str) -> LLM:
    """
    Smart fallback:
      1. Azure OpenAI if AZURE_OPENAI_API_KEY is set
      2. OpenAI if OPENAI_API_KEY is set
      3. Stub LLM (won't crash on init; will fail at inference with a clear message)
    """
    if get_secret("AZURE_OPENAI_API_KEY"):
        logger.warning(f"Falling back from '{original_model}' → Azure OpenAI.")
        try:
            return _build_azure_llm("azure/gpt-4o")
        except Exception as e:
            logger.error(f"Azure fallback failed: {e}")

    openai_key = get_secret("OPENAI_API_KEY")
    if openai_key:
        logger.warning(f"Falling back from '{original_model}' → OpenAI gpt-4o.")
        try:
            return LLM(model="openai/gpt-4o", api_key=openai_key)
        except Exception as e:
            logger.error(f"OpenAI fallback failed: {e}")

    logger.error(
        "No API keys found. Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY."
    )
    return LLM(model="openai/gpt-4o", api_key="NO_KEY_CONFIGURED")
