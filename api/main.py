import asyncio
import json
import logging
import os
import time
import pathlib
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union, AsyncGenerator, Tuple
from uuid import uuid4
from jwt.jwks_client import PyJWKClient
import uvicorn
import jwt # type: ignore
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

try:
    import vertexai
    from vertexai.generative_models import (
        GenerativeModel,
        Part as VertexPart,
        Content as VertexContent,
        GenerationConfig as VertexGenerationConfig,
        HarmCategory as VertexHarmCategory,
        HarmBlockThreshold as VertexHarmBlockThreshold,
        Tool as VertexTool,
        ToolConfig as VertexToolConfig,
        GenerationResponse,
    )
    from vertexai.generative_models import GenerationResponse as _SDKGenerationResponseStreamChunk
    from google.api_core import exceptions as google_api_core_exceptions
    from vertexai import rag
    VERTEX_AI_SDK_AVAILABLE = True
except ImportError:
    VERTEX_AI_SDK_AVAILABLE = False
    class _PlaceholderSDKType:
        def __init__(self, *args: Any, **kwargs: Any): pass
        def __getattr__(self, name: str) -> Any:
            if name in ("GoogleAPIError", "InvalidArgument", "PermissionDenied", "ResourceExhausted", "DeadlineExceeded"):
                return type(name + "Placeholder", (Exception,), {})
            raise RuntimeError(f"Vertex AI SDK not available. Cannot access '{name}'.")

    GenerativeModel = _PlaceholderSDKType #type: ignore
    VertexPart = _PlaceholderSDKType #type: ignore
    VertexContent = _PlaceholderSDKType #type: ignore
    VertexGenerationConfig = _PlaceholderSDKType #type: ignore
    VertexHarmCategory = _PlaceholderSDKType #type: ignore
    VertexHarmBlockThreshold = _PlaceholderSDKType #type: ignore
    VertexTool = _PlaceholderSDKType #type: ignore
    VertexToolConfig = _PlaceholderSDKType #type: ignore
    GenerationResponse = _PlaceholderSDKType #type: ignore
    _SDKGenerationResponseStreamChunk = _PlaceholderSDKType #type: ignore
    google_api_core_exceptions = _PlaceholderSDKType() # type: ignore

    class RagPlaceholder:
        Retrieval = _PlaceholderSDKType #type: ignore
        VertexRagStore = _PlaceholderSDKType #type: ignore
        RagResource = _PlaceholderSDKType #type: ignore
        RagRetrievalConfig = _PlaceholderSDKType #type: ignore
    rag = RagPlaceholder() #type: ignore

load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
VERTEX_PROJECT_ID_ENV = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION")

VERTEX_GEMINI_MODEL_NAME = os.getenv("VERTEX_GEMINI_MODEL_NAME", "gemini-2.0-flash-001").replace("-latest", "")

RAG_CORPUS_NAME = os.getenv("RAG_CORPUS_NAME")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# LLM_SYSTEM_INSTRUCTION = os.getenv("RAG_FT_SYSTEM_INSTRUCTION_FOR_API", "").strip() # Removed: Will be loaded from JSON
SYSTEM_INSTRUCTION_JSON_PATH = os.getenv("SYSTEM_INSTRUCTION_JSON_PATH", "data/config/instructions.json")
LLM_SYSTEM_INSTRUCTION_FROM_JSON: Optional[str] = None # Will hold instruction from JSON

GEMINI_SAFETY_SETTINGS_THRESHOLD = os.getenv("GEMINI_SAFETY_SETTINGS_THRESHOLD", "BLOCK_MEDIUM_AND_ABOVE")
GENERATIVE_MODEL_API_TIMEOUT_SECONDS = int(os.getenv("GENERATIVE_MODEL_API_TIMEOUT_SECONDS", "90"))
STREAMING_RESPONSE_TIMEOUT_SECONDS = int(os.getenv("STREAMING_RESPONSE_TIMEOUT_SECONDS", "500"))
WEBSOCKET_GENERATION_TIMEOUT_SECONDS = int(os.getenv("WEBSOCKET_GENERATION_TIMEOUT_SECONDS", "120"))
DISABLE_SAFETY_SETTINGS_FOR_FINETUNED_MODEL = os.getenv("DISABLE_SAFETY_SETTINGS_FOR_FINETUNED_MODEL", "False").lower() == "true"

DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
INTERACTION_LOG_PATH = os.getenv("INTERACTION_LOG_PATH")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_JWT_AUD = os.getenv("SUPABASE_JWT_AUD", "authenticated")

REQUEST_TIMESTAMPS: Dict[str, List[float]] = {}
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "15"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
SDK_THREAD_FINISH_TIMEOUT_SECONDS = 10

def setup_logger(name: str, level: int, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.propagate = False
    return logger

app_logger = setup_logger("app_logger", logging.DEBUG if DEBUG_MODE else logging.INFO)
interaction_logger = setup_logger(
    "interaction_logger",
    logging.INFO,
    INTERACTION_LOG_PATH
)

gcp_project_configured: Optional[str] = None
vertex_ai_client_wrapper_instance: Optional[Any] = None
vertex_ai_configured_successfully = False
llm_model_operational = False
_supabase_jwk_client: Optional[PyJWKClient] = None
rag_retrieval_tool_global: Optional[VertexTool] = None

class ChatMessageInput(BaseModel):
    role: Literal["user", "model"]
    text: str = Field(..., min_length=1)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    conversation_history: Optional[List[ChatMessageInput]] = None
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response.")

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    debug_info: Optional[Dict[str, Any]] = None

def _ensure_vertex_contents(contents_list: List[Any]) -> Optional[List[VertexContent]]:
    if not contents_list or not VERTEX_AI_SDK_AVAILABLE: return None
    vertex_content_list: List[VertexContent] = []
    for item in contents_list:
        if isinstance(item, VertexContent):
            vertex_content_list.append(item)
            continue
        try:
            current_role = item.get('role', 'user') if isinstance(item, dict) else getattr(item, 'role', 'user')
            parts_data = item.get('parts', []) if isinstance(item, dict) else getattr(item, 'parts', [])
            vertex_parts = []
            for p_item in parts_data:
                text_content = None
                if isinstance(p_item, dict) and 'text' in p_item:
                    text_content = p_item['text']
                elif isinstance(p_item, str):
                    text_content = p_item
                if text_content and text_content.strip():
                    if VERTEX_AI_SDK_AVAILABLE:
                        vertex_parts.append(VertexPart.from_text(text_content))
            if vertex_parts:
                if VERTEX_AI_SDK_AVAILABLE:
                    vertex_content_list.append(VertexContent(role=current_role, parts=vertex_parts))
        except Exception as e:
            app_logger.error(f"Error processing item for VertexContent: {item} - {e}", exc_info=DEBUG_MODE)
    return vertex_content_list if vertex_content_list else None

class VertexAIClientWrapper:
    def __init__(self, project: str, location: str):
        if not VERTEX_AI_SDK_AVAILABLE:
            raise RuntimeError("Vertex AI SDK is not available. Please install google-cloud-aiplatform.")
        try:
            vertexai.init(project=project, location=location)
        except Exception as e:
            raise RuntimeError(f"Vertex AI SDK initialization failed: {e}") from e

    def generate_content(
        self, model_name: str, contents: List[VertexContent],
        generation_config: Optional[Union[VertexGenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[VertexHarmCategory, VertexHarmBlockThreshold]] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List[VertexTool]] = None,
        tool_config: Optional[VertexToolConfig] = None,
        stream: bool = False,
        request_options: Optional[Dict[str, Any]] = None,
        session_id_for_log: Optional[str] = "unknown_sid"
    ) -> Union[GenerationResponse, AsyncGenerator[_SDKGenerationResponseStreamChunk, None]]:
        if not VERTEX_AI_SDK_AVAILABLE:
            raise RuntimeError("Vertex AI SDK not available for generate_content call.")
        model_init_kwargs: Dict[str, Any] = {}
        if system_instruction and system_instruction.strip():
            model_init_kwargs["system_instruction"] = VertexContent(role="system", parts=[VertexPart.from_text(system_instruction)])
        gen_model_instance = GenerativeModel(model_name=model_name, **model_init_kwargs)
        vertex_contents_final = _ensure_vertex_contents(contents)
        if not vertex_contents_final:
            raise ValueError("Invalid or empty contents provided for LLM generation.")
        vertex_generation_config_obj: Optional[VertexGenerationConfig] = None
        if isinstance(generation_config, VertexGenerationConfig):
            vertex_generation_config_obj = generation_config
        elif isinstance(generation_config, dict):
            known_gc_keys = {"temperature", "top_p", "top_k", "candidate_count", "max_output_tokens", "stop_sequences"}
            filtered_gc_dict = {k: v for k, v in generation_config.items() if k in known_gc_keys and v is not None}
            if filtered_gc_dict:
                vertex_generation_config_obj = VertexGenerationConfig(**filtered_gc_dict)
        try:
            app_logger.info(f"[WRAPPER SID:{session_id_for_log}] Attempting generate_content for model {model_name} WITH request_options if provided. Options: {request_options}")
            response = gen_model_instance.generate_content(
                contents=vertex_contents_final,
                generation_config=vertex_generation_config_obj,
                safety_settings=safety_settings,
                tools=tools,
                tool_config=tool_config,
                stream=stream,
                request_options=request_options
            )
            app_logger.info(f"[WRAPPER SID:{session_id_for_log}] Call to generate_content WITH request_options for model {model_name} successful.")
            return response
        except TypeError as e:
            if "request_options" in str(e).lower() and \
               ("unexpected keyword argument" in str(e).lower() or "got an unexpected keyword argument" in str(e).lower()):
                app_logger.warning(f"[WRAPPER SID:{session_id_for_log}] SDK Error (model: {model_name}): {e}. Retrying without 'request_options'.")
                app_logger.info(f"[WRAPPER SID:{session_id_for_log}] Attempting generate_content for model {model_name} WITHOUT request_options (retry).")
                response = gen_model_instance.generate_content(
                    contents=vertex_contents_final,
                    generation_config=vertex_generation_config_obj,
                    safety_settings=safety_settings,
                    tools=tools,
                    tool_config=tool_config,
                    stream=stream
                )
                app_logger.info(f"[WRAPPER SID:{session_id_for_log}] Call to generate_content WITHOUT request_options (retry) for model {model_name} successful.")
                return response
            else:
                app_logger.error(f"[WRAPPER SID:{session_id_for_log}] TypeError (non-request_options) calling Vertex AI (model: {model_name}): {e}", exc_info=DEBUG_MODE)
                raise
        except Exception as e:
            app_logger.error(f"[WRAPPER SID:{session_id_for_log}] Error (non-TypeError) calling Vertex AI (model: {model_name}): {e}", exc_info=DEBUG_MODE)
            raise

bearer_scheme = HTTPBearer()
async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> Dict[str, Any]:
    token = creds.credentials
    if not (SUPABASE_URL or SUPABASE_JWT_SECRET):
        app_logger.error("Authentication service (Supabase URL or JWT Secret) is not configured.")
        raise HTTPException(status_code=500, detail="Authentication service not configured.")
    try:
        unverified_header = jwt.get_unverified_header(token)
    except jwt.exceptions.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token header: {e}")
    alg = unverified_header.get("alg")
    decode_options = {"verify_signature": True, "verify_exp": True}
    try:
        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                app_logger.error("Auth: Supabase JWT Secret for HS256 not configured.")
                raise HTTPException(status_code=500, detail="Auth (HS256 secret missing).")
            payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience=SUPABASE_JWT_AUD or None, options=decode_options)
        elif alg == "RS256":
            if not _supabase_jwk_client:
                app_logger.error("Auth: Supabase JWKS client for RS256 not initialized.")
                raise HTTPException(status_code=500, detail="Auth (RS256 JWKS client not ready).")
            signing_key = _supabase_jwk_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(token, signing_key.key, algorithms=["RS256"], audience=SUPABASE_JWT_AUD or None, options=decode_options)
        else:
            raise HTTPException(status_code=401, detail=f"Unsupported token algorithm: {alg}.")
        return {"user_id": payload.get("sub"), "email": payload.get("email"), "role": payload.get("role")}
    except jwt.ExpiredSignatureError: raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidAudienceError: raise HTTPException(status_code=401, detail="Invalid token audience.")
    except jwt.MissingRequiredClaimError as e: app_logger.warning(f"Token missing claim: {e} (alg: {alg})", exc_info=DEBUG_MODE); raise HTTPException(status_code=401, detail=f"Token missing claim: {e}.")
    except jwt.InvalidTokenError as e: app_logger.warning(f"Invalid token: {e} (alg: {alg})", exc_info=DEBUG_MODE); raise HTTPException(status_code=401, detail=f"Token validation error: {e}.")
    except Exception as e: app_logger.error(f"Unexpected token validation error: {e} (alg: {alg})", exc_info=DEBUG_MODE); raise HTTPException(status_code=500, detail="Error during token validation.")

async def rate_limit_dependency(request: Request):
    user_id_for_key = getattr(request.state, "user_id_for_rate_limit", None)
    client_key = user_id_for_key or (request.client.host if request.client else "unknown_ip")
    current_time = time.monotonic()
    REQUEST_TIMESTAMPS.setdefault(client_key, [])
    timestamps = [ts for ts in REQUEST_TIMESTAMPS[client_key] if ts > current_time - RATE_LIMIT_WINDOW]
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        wait_time = (timestamps[0] + RATE_LIMIT_WINDOW) - current_time if timestamps else RATE_LIMIT_WINDOW
        raise HTTPException(status_code=429, detail=f"Too Many Requests. Try again in {max(0, int(wait_time)) + 1} seconds.")
    timestamps.append(current_time)
    REQUEST_TIMESTAMPS[client_key] = timestamps

def _get_vertex_safety_threshold_enum_val() -> Union[VertexHarmBlockThreshold, str]:
    if not VERTEX_AI_SDK_AVAILABLE: return GEMINI_SAFETY_SETTINGS_THRESHOLD.upper()
    mapping = {"BLOCK_NONE": VertexHarmBlockThreshold.BLOCK_NONE, "BLOCK_ONLY_HIGH": VertexHarmBlockThreshold.BLOCK_ONLY_HIGH, "BLOCK_MEDIUM_AND_ABOVE": VertexHarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "BLOCK_LOW_AND_ABOVE": VertexHarmBlockThreshold.BLOCK_LOW_AND_ABOVE}
    default_threshold = VertexHarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    enum_val = mapping.get(GEMINI_SAFETY_SETTINGS_THRESHOLD.upper(), default_threshold)
    if isinstance(enum_val, str): app_logger.warning(f"Safety threshold string '{GEMINI_SAFETY_SETTINGS_THRESHOLD}' mapped to string '{enum_val}', not enum. Defaulting."); return default_threshold
    return enum_val

def get_vertex_safety_settings() -> Optional[Dict[VertexHarmCategory, VertexHarmBlockThreshold]]:
    if not VERTEX_AI_SDK_AVAILABLE or not hasattr(VertexHarmCategory, "HARM_CATEGORY_HARASSMENT"): return None
    threshold = _get_vertex_safety_threshold_enum_val()
    if isinstance(threshold, str): app_logger.error(f"Could not map safety threshold string '{threshold}' to a valid VertexHarmBlockThreshold enum."); return None
    categories = [VertexHarmCategory.HARM_CATEGORY_HARASSMENT, VertexHarmCategory.HARM_CATEGORY_HATE_SPEECH, VertexHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, VertexHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT]
    return {category: threshold for category in categories}

CHAT_SESSIONS_DIR = pathlib.Path(os.getenv("CHAT_SESSIONS_BASE_PATH", "/app/data/chat_sessions"))

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global gcp_project_configured, vertex_ai_client_wrapper_instance, vertex_ai_configured_successfully
    global llm_model_operational, _supabase_jwk_client, rag_retrieval_tool_global
    global LLM_SYSTEM_INSTRUCTION_FROM_JSON, SYSTEM_INSTRUCTION_JSON_PATH

    try: CHAT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True); app_logger.info(f"Chat sessions directory ensured at: {CHAT_SESSIONS_DIR}")
    except OSError as e: app_logger.error(f"Could not create chat sessions directory {CHAT_SESSIONS_DIR}: {e}", exc_info=DEBUG_MODE)

    # Load system instruction from JSON
    try:
        instruction_file_path = pathlib.Path(SYSTEM_INSTRUCTION_JSON_PATH)
        instruction_file_full_path = instruction_file_path.resolve() # Get absolute path for logging
        app_logger.info(f"Attempting to load system instruction from: {instruction_file_full_path}")

        if instruction_file_path.is_file():
            # Ensure parent directory exists for the config file if a relative path like 'config/...' is used
            # and the script is run from a different working directory.
            # However, for reading, generally, this is not needed, but helpful if we were to write.
            # For now, simply check if the path is absolute or relative to CWD.
            app_logger.info(f"Found instruction file at: {instruction_file_full_path}")
            with open(instruction_file_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                LLM_SYSTEM_INSTRUCTION_FROM_JSON = config_data.get("system_instruction", "").strip()
                if LLM_SYSTEM_INSTRUCTION_FROM_JSON:
                    app_logger.info(f"Successfully loaded system instruction from JSON: {SYSTEM_INSTRUCTION_JSON_PATH}")
                    app_logger.debug(f"System instruction content: '{LLM_SYSTEM_INSTRUCTION_FROM_JSON[:100]}...'")
                else:
                    app_logger.warning(f"System instruction key 'system_instruction' not found or empty in JSON: {SYSTEM_INSTRUCTION_JSON_PATH}. No system instruction from JSON will be used.")
                    LLM_SYSTEM_INSTRUCTION_FROM_JSON = "" # Ensure it's an empty string
        else:
            app_logger.warning(f"System instruction JSON file not found at: {instruction_file_full_path} (relative path from CWD: {SYSTEM_INSTRUCTION_JSON_PATH}). No system instruction from JSON will be used.")
            LLM_SYSTEM_INSTRUCTION_FROM_JSON = "" # Explicitly set to empty if file not found
    except json.JSONDecodeError as e:
        resolved_path_for_error = pathlib.Path(SYSTEM_INSTRUCTION_JSON_PATH).resolve()
        app_logger.error(f"Error decoding system instruction JSON file {resolved_path_for_error}: {e}", exc_info=DEBUG_MODE)
        LLM_SYSTEM_INSTRUCTION_FROM_JSON = "" # Explicitly set to empty on error
    except Exception as e:
        resolved_path_for_error = pathlib.Path(SYSTEM_INSTRUCTION_JSON_PATH).resolve()
        app_logger.error(f"An unexpected error occurred while loading system instruction from JSON {resolved_path_for_error}: {e}", exc_info=DEBUG_MODE)
        LLM_SYSTEM_INSTRUCTION_FROM_JSON = "" # Explicitly set to empty on error


    gcp_project_configured = VERTEX_PROJECT_ID_ENV
    if not gcp_project_configured:
        try: from google.auth import default as google_auth_default; _, gcp_project_configured = google_auth_default(scopes=['https://www.googleapis.com/auth/cloud-platform']); app_logger.info(f"GCP Project ID determined via default credentials: {gcp_project_configured}" if gcp_project_configured else "GCP Project ID not found by default creds.")
        except Exception as e: app_logger.warning(f"Could not determine GCP Project ID via default credentials: {e}")
    if not gcp_project_configured: app_logger.error("CRITICAL: GCP Project ID could not be determined.")

    if VERTEX_AI_SDK_AVAILABLE and gcp_project_configured and VERTEX_LOCATION:
        try:
            vertex_ai_client_wrapper_instance = VertexAIClientWrapper(project=gcp_project_configured, location=VERTEX_LOCATION)
            vertex_ai_configured_successfully = True; app_logger.info(f"Vertex AI Client initialized for project '{gcp_project_configured}' in '{VERTEX_LOCATION}'.")
            llm_model_operational = bool(VERTEX_GEMINI_MODEL_NAME)
            if not llm_model_operational: app_logger.error("VERTEX_GEMINI_MODEL_NAME not set.")
            else: app_logger.info(f"Using LLM model: {VERTEX_GEMINI_MODEL_NAME}")
        except Exception as e: app_logger.error(f"Failed to initialize VertexAIClientWrapper: {e}", exc_info=DEBUG_MODE); vertex_ai_configured_successfully = False; llm_model_operational = False
    else:
        missing = [item for item, cond in [("Vertex AI SDK", VERTEX_AI_SDK_AVAILABLE), ("GCP Project ID", gcp_project_configured), ("Vertex Location", VERTEX_LOCATION)] if not cond]
        app_logger.warning(f"Vertex AI features disabled. Missing: {', '.join(missing) if missing else 'configuration details'}.")
        vertex_ai_configured_successfully = False; llm_model_operational = False

    if SUPABASE_URL:
        try: jwks_uri = SUPABASE_URL.rstrip('/') + "/auth/v1/.well-known/jwks.json"; _supabase_jwk_client = PyJWKClient(jwks_uri, cache_jwk_set=True, lifespan=3600); app_logger.info(f"Supabase JWKS Client initialized for URL: {jwks_uri}")
        except Exception as e: app_logger.error(f"Failed to initialize Supabase JWK client (RS256): {e}", exc_info=DEBUG_MODE)
    if not SUPABASE_JWT_SECRET and not _supabase_jwk_client: app_logger.warning("Auth: Neither Supabase JWT Secret (HS256) nor Supabase URL (RS256 JWKS) configured.")
    elif SUPABASE_JWT_SECRET: app_logger.info("Supabase JWT Secret (HS256) configured.")

    if vertex_ai_configured_successfully and VERTEX_AI_SDK_AVAILABLE and RAG_CORPUS_NAME:
        if not RAG_CORPUS_NAME.startswith("projects/"): app_logger.critical(f"RAG_CORPUS_NAME ('{RAG_CORPUS_NAME}') must be a full resource name. RAG tool NOT created.")
        else:
            try:
                rag_retrieval = rag.Retrieval(source=rag.VertexRagStore(rag_resources=[rag.RagResource(rag_corpus=RAG_CORPUS_NAME)], rag_retrieval_config=rag.RagRetrievalConfig(top_k=RAG_TOP_K)))
                rag_retrieval_tool_global = VertexTool.from_retrieval(retrieval=rag_retrieval); app_logger.info(f"RAG Engine tool created for corpus: {RAG_CORPUS_NAME} with top_k={RAG_TOP_K}.")
            except Exception as e: app_logger.error(f"Failed to create RAG Engine tool for '{RAG_CORPUS_NAME}': {e}", exc_info=DEBUG_MODE)
    elif RAG_CORPUS_NAME: app_logger.warning(f"RAG_CORPUS_NAME ('{RAG_CORPUS_NAME}') set, but Vertex AI not configured. RAG tool NOT created.")
    
    yield
    app_logger.info("Application shutdown.")

app = FastAPI(title="RAG Chat API with Vertex AI RAG Engine", version="2.3.4-json-instr", lifespan=lifespan, docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    app_logger.error(f"Request validation error for {request.url.path}: {exc.errors()}", exc_info=DEBUG_MODE)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors()), "message": "Invalid input format."})

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", tags=["Utility"], include_in_schema=False)
async def root(): return {"message": f"{app.title} v{app.version} is running."}

@app.get("/health", tags=["Utility"])
async def health_check():
    auth_ok = bool(_supabase_jwk_client or SUPABASE_JWT_SECRET)
    status = "healthy"
    if not (vertex_ai_configured_successfully and llm_model_operational and auth_ok): status = "degraded"
    if not (vertex_ai_configured_successfully and llm_model_operational): status = "unhealthy"
    return {"service_status": status, "version": app.version, "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            "details": {"gcp_project_id": gcp_project_configured or "N/A", "vertex_sdk_initialized": vertex_ai_configured_successfully,
                        "llm_model": VERTEX_GEMINI_MODEL_NAME if llm_model_operational else "N/A", "llm_operational": llm_model_operational,
                        "rag_tool_configured": bool(RAG_CORPUS_NAME), "rag_tool_active": bool(rag_retrieval_tool_global),
                        "auth_service_configured": auth_ok, "debug_mode": DEBUG_MODE,
                        "system_instruction_source": f"JSON ('{SYSTEM_INSTRUCTION_JSON_PATH}')" if LLM_SYSTEM_INSTRUCTION_FROM_JSON else "None",
                        "system_instruction_loaded": bool(LLM_SYSTEM_INSTRUCTION_FROM_JSON)                       
                        }}

@app.get("/api/sessions/", tags=["Session Management"], dependencies=[Depends(rate_limit_dependency)])
async def list_sessions(current_user: Dict[str, Any] = Depends(get_current_user), offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
    user_id = current_user.get("user_id");
    if not user_id: raise HTTPException(status_code=400, detail="User ID not found in token.")
    user_session_dir = CHAT_SESSIONS_DIR / str(user_id)
    if not user_session_dir.exists(): return {"sessions": [], "pagination": {"total": 0, "offset": offset, "limit": limit, "has_more": False}}
    try: session_files = sorted([f for f in user_session_dir.glob("*.jsonl") if f.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)
    except OSError as e: app_logger.error(f"Error listing session files for user {user_id}: {e}", exc_info=DEBUG_MODE); raise HTTPException(status_code=500, detail="Error accessing session data.")
    metadata = []
    for f_path in session_files:
        try: created_at = time.gmtime(f_path.stat().st_ctime); metadata.append({"id": f_path.stem, "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", created_at), "title": f"Chat on {time.strftime('%Y-%m-%d', created_at)}"})
        except Exception as e: app_logger.warning(f"Could not process session file {f_path}: {e}")
    paginated = metadata[offset : offset + limit]
    return {"sessions": paginated, "pagination": {"total": len(metadata), "offset": offset, "limit": limit, "has_more": (offset + limit) < len(metadata)}}

@app.get("/api/sessions/{session_id}", tags=["Session Management"], dependencies=[Depends(rate_limit_dependency)])
async def get_session_messages(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user.get("user_id");
    if not user_id: raise HTTPException(status_code=400, detail="User ID not found.")
    session_file = CHAT_SESSIONS_DIR / str(user_id) / f"{pathlib.Path(session_id).name}.jsonl"
    if not session_file.is_file(): raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    messages: List[Dict[str,str]] = []
    try:
        with session_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    msg = json.loads(line)
                    if isinstance(msg.get("role"), str) and isinstance(msg.get("text"), str): messages.append(ChatMessageInput(**msg).model_dump())
                    else: app_logger.warning(f"Skipping malformed message in session {session_id} (user {user_id}, line {i}).")
                except (json.JSONDecodeError, Exception) as e: app_logger.warning(f"Skipping invalid line in session {session_id} (user {user_id}, line {i}): {e}")
        return messages
    except IOError as e: app_logger.error(f"Failed to read session {session_file}: {e}", exc_info=DEBUG_MODE); raise HTTPException(status_code=500, detail="Error reading session file.")
    except Exception as e: app_logger.error(f"Unexpected error loading session {session_id}: {e}", exc_info=DEBUG_MODE); raise HTTPException(status_code=500, detail="Internal error retrieving session.")

def _persist_chat_messages(user_id: str, session_id: str, question: str, model_answer: str, log_entry: Dict[str, Any]) -> None:
    user_dir = CHAT_SESSIONS_DIR / str(user_id)
    try:
        user_dir.mkdir(parents=True, exist_ok=True)
        file_path = user_dir / f"{pathlib.Path(session_id).name}.jsonl"
        answer_to_log = model_answer
        if not model_answer.strip():
            if log_entry.get("error_detail"): answer_to_log = str(log_entry.get("error_detail", "Error"))
            elif log_entry.get("llm_prompt_feedback_block_reason", "NONE") != "NONE": answer_to_log = f"Blocked: {log_entry['llm_prompt_feedback_block_reason']}"
            elif log_entry.get("llm_finish_reason") not in ["STOP", "NONE", "UNKNOWN"]: answer_to_log = f"Issue: {log_entry['llm_finish_reason']}"
        with file_path.open("a", encoding="utf-8") as sf:
            sf.write(json.dumps({"role": "user", "text": question}) + "\n")
            sf.write(json.dumps({"role": "model", "text": answer_to_log.strip()}) + "\n")
    except Exception as e: app_logger.error(f"Error persisting chat history for session {session_id}: {e}", exc_info=DEBUG_MODE)

async def _stream_llm_response_generator(
    llm_api_args: Dict[str, Any], session_id: str, user_id: str, question: str, client_ip: str
) -> AsyncGenerator[str, None]:
    app_logger.info(f"[SID:{session_id}] STREAMER: Entered. User: {user_id}, IP: {client_ip}")
    start_time = time.monotonic()
    full_response_text = ""
    error_sent_to_client = False
    log_entry: Dict[str, Any] = {"session_id": session_id, "user_id": user_id, "query_length": len(question),
                                 "query_preview": question[:50] + ("..." if len(question) > 50 else ""),
                                 "rag_active": bool(llm_api_args.get("tools")), "llm_model": llm_api_args.get("model_name"),
                                 "stream_request": True, "client_ip": client_ip, "error_detail": None,
                                 "llm_prompt_feedback_block_reason": "NONE", "llm_finish_reason": "UNKNOWN"}
    response_queue: asyncio.Queue[Tuple[Optional[_SDKGenerationResponseStreamChunk], Optional[BaseException]]] = asyncio.Queue()
    try:
        loop_for_thread = asyncio.get_running_loop()
        app_logger.info(f"[SID:{session_id}] STREAMER: Using event loop ID {id(loop_for_thread)} for thread callbacks.")
    except RuntimeError:
        app_logger.error(f"[SID:{session_id}] STREAMER: Could not get running loop! Falling back to get_event_loop().")
        loop_for_thread = asyncio.get_event_loop()

    sdk_thread_task_handle: Optional[asyncio.Task] = None

    def put_item_threadsafe_callback(item_tuple: Tuple[Optional[Any], Optional[BaseException]]):
        item_data, item_error = item_tuple
        app_logger.info(f"[SID:{session_id}] SDK THREAD CALLBACK (Loop ID {id(asyncio.get_event_loop()) if loop_for_thread.is_running() else 'N/A not running'}): "
                        f"Preparing to put. Item type: {type(item_data).__name__ if item_data else 'None'}, "
                        f"Error type: {type(item_error).__name__ if item_error else 'None'}. "
                        f"Queue size before put: {response_queue.qsize()}")
        try:
            response_queue.put_nowait(item_tuple)
            app_logger.info(f"[SID:{session_id}] SDK THREAD CALLBACK: Successfully put item. "
                            f"Queue size after put: {response_queue.qsize()}")
        except Exception as e_put:
            app_logger.error(f"[SID:{session_id}] SDK THREAD CALLBACK: EXCEPTION during response_queue.put_nowait: {type(e_put).__name__} - {e_put}", exc_info=True)

    def sdk_iterate_in_thread():
        app_logger.info(f"[SID:{session_id}] SDK THREAD (OS Thread ID: {threading.get_ident()}, Loop ID for callbacks: {id(loop_for_thread)}): Entered function.")
        thread_op_start = time.monotonic()
        try:
            if not vertex_ai_client_wrapper_instance:
                app_logger.error(f"[SID:{session_id}] SDK THREAD: Vertex AI client not initialized.")
                loop_for_thread.call_soon_threadsafe(put_item_threadsafe_callback, (None, RuntimeError("Vertex AI client not initialized.")))
                return
            
            current_args = {**llm_api_args, "request_options": {"timeout": GENERATIVE_MODEL_API_TIMEOUT_SECONDS}, "session_id_for_log": session_id}
            app_logger.info(f"[SID:{session_id}] SDK THREAD: Calling VertexAIClientWrapper.generate_content. Model: {current_args.get('model_name')}")
            
            sdk_call_start = time.monotonic()
            sdk_stream_iterator = vertex_ai_client_wrapper_instance.generate_content(**current_args)
            app_logger.info(f"[SID:{session_id}] SDK THREAD: Got iterator in {time.monotonic() - sdk_call_start:.3f}s. Iterating stream.")
            
            count = 0; first_chunk_time = None; iter_start_time = time.monotonic()
            for chunk in sdk_stream_iterator:
                if first_chunk_time is None: first_chunk_time = time.monotonic(); app_logger.info(f"[SID:{session_id}] SDK THREAD: First chunk at {first_chunk_time - iter_start_time:.3f}s (total {first_chunk_time - sdk_call_start:.3f}s from call).")
                count += 1; app_logger.debug(f"[SID:{session_id}] SDK THREAD: Got chunk {count}. Preview: {str(getattr(chunk.candidates[0].content.parts[0], 'text', 'N/A'))[:30] if DEBUG_MODE and chunk.candidates and chunk.candidates[0].content.parts else 'N/A'}")
                loop_for_thread.call_soon_threadsafe(put_item_threadsafe_callback, (chunk, None))
            
            app_logger.info(f"[SID:{session_id}] SDK THREAD: Finished {count} chunks in {time.monotonic() - iter_start_time:.3f}s.")
        except Exception as e:
            app_logger.error(f"[SID:{session_id}] SDK THREAD: EXCEPTION after {time.monotonic() - thread_op_start:.3f}s: {type(e).__name__} - {e}", exc_info=DEBUG_MODE)
            loop_for_thread.call_soon_threadsafe(put_item_threadsafe_callback, (None, e))
        finally:
            app_logger.info(f"[SID:{session_id}] SDK THREAD: Finally block after {time.monotonic() - thread_op_start:.3f}s. Signalling StopAsyncIteration.")
            loop_for_thread.call_soon_threadsafe(put_item_threadsafe_callback, (None, StopAsyncIteration()))

    try:
        yield f"data: {json.dumps({'type': 'processing', 'session_id': session_id, 'message': 'Processing your request...'})}\n\n"
        app_logger.info(f"[SID:{session_id}] STREAMER: Creating and scheduling SDK thread task.")
        sdk_thread_task_handle = asyncio.create_task(
            asyncio.to_thread(sdk_iterate_in_thread),
            name=f"sdk_thread_for_sid_{session_id}"
        )
        app_logger.info(f"[SID:{session_id}] STREAMER: SDK thread task scheduled: {sdk_thread_task_handle.get_name()}")

        final_sdk_response_obj = None; stream_had_text = False
        while True:
            app_logger.info(f"[SID:{session_id}] STREAMER: Waiting for queue item (timeout: {STREAMING_RESPONSE_TIMEOUT_SECONDS}s). Queue size: {response_queue.qsize()}")
            try:
                chunk, error = await asyncio.wait_for(response_queue.get(), timeout=STREAMING_RESPONSE_TIMEOUT_SECONDS)
                app_logger.info(f"[SID:{session_id}] STREAMER: Got from queue. Chunk: {chunk is not None}, Error: {type(error).__name__ if error else 'None'}. Queue size: {response_queue.qsize()}")
            except asyncio.TimeoutError:
                log_entry["error_detail"] = f"Streaming response queue timed out after {STREAMING_RESPONSE_TIMEOUT_SECONDS}s."
                app_logger.error(f"[SID:{session_id}] STREAMER: {log_entry['error_detail']}")
                if not error_sent_to_client: yield f"data: {json.dumps({'type': 'error', 'session_id': session_id, 'message': log_entry['error_detail']})}\n\n"; error_sent_to_client = True
                return
            
            if isinstance(error, StopAsyncIteration): app_logger.info(f"[SID:{session_id}] STREAMER: StopAsyncIteration received."); break
            if error: app_logger.error(f"[SID:{session_id}] STREAMER: Error from queue: {error}", exc_info=DEBUG_MODE and isinstance(error, Exception)); raise error

            if chunk and VERTEX_AI_SDK_AVAILABLE:
                final_sdk_response_obj = chunk
                if chunk.candidates:
                    stream_had_text = True
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            full_response_text += part.text
                            yield f"data: {json.dumps({'type': 'chunk', 'session_id': session_id, 'data': part.text})}\n\n"
        
        app_logger.info(f"[SID:{session_id}] STREAMER: Loop finished. Processing final response state.")
        if final_sdk_response_obj and VERTEX_AI_SDK_AVAILABLE:
            prompt_fb = getattr(final_sdk_response_obj, 'prompt_feedback', None)
            candidates = getattr(final_sdk_response_obj, 'candidates', [])
            log_entry["llm_prompt_feedback_block_reason"] = str(getattr(prompt_fb, 'block_reason', "NONE")).upper()
            if prompt_fb and prompt_fb.block_reason != 0:
                msg = getattr(prompt_fb, 'block_reason_message', "Request blocked.")
                log_entry["error_detail"] = msg; app_logger.warning(f"[SID:{session_id}] STREAMER: Prompt blocked: {msg}")
                if not stream_had_text and not error_sent_to_client:
                    yield f"data: {json.dumps({'type': 'error', 'session_id': session_id, 'message': msg})}\n\n"
                    error_sent_to_client = True
                full_response_text = full_response_text or msg
            elif not stream_had_text and not full_response_text.strip():
                msg = "AI model returned no content."
                log_entry["error_detail"] = msg; app_logger.warning(f"[SID:{session_id}] STREAMER: {msg}")
                if not error_sent_to_client:
                    yield f"data: {json.dumps({'type': 'error', 'session_id': session_id, 'message': msg})}\n\n"
                    error_sent_to_client = True
                full_response_text = full_response_text or msg
            elif candidates and candidates[0]:
                finish_reason_obj = getattr(candidates[0], 'finish_reason', None)
                if finish_reason_obj and hasattr(finish_reason_obj, 'name'):
                    finish_reason = str(finish_reason_obj.name).upper()
                else:
                    finish_reason = "UNKNOWN"
                log_entry["llm_finish_reason"] = finish_reason; app_logger.info(f"[SID:{session_id}] STREAMER: Finish reason: {finish_reason}")
                if finish_reason in ["SAFETY", "RECITATION"] and not error_sent_to_client and not stream_had_text:
                    msg = f"Response blocked: {finish_reason.lower()}."
                    log_entry["error_detail"] = log_entry.get("error_detail") or msg 
                    yield f"data: {json.dumps({'type': 'error', 'session_id': session_id, 'message': msg})}\n\n"
                    error_sent_to_client = True
                    full_response_text = full_response_text or msg

            usage_meta = getattr(final_sdk_response_obj, 'usage_metadata', None)
            if usage_meta: log_entry["llm_token_counts"] = {"prompt": usage_meta.prompt_token_count, "candidates": usage_meta.candidates_token_count, "total": usage_meta.total_token_count}
        elif not final_sdk_response_obj and not log_entry.get("error_detail"):
            msg = "AI model provided no response object."
            log_entry["error_detail"] = msg; app_logger.warning(f"[SID:{session_id}] STREAMER: {msg}")
            if not error_sent_to_client:
                yield f"data: {json.dumps({'type': 'error', 'session_id': session_id, 'message': msg})}\n\n"
                error_sent_to_client = True
        
        if full_response_text.strip() and not log_entry.get("error_detail"):
            app_logger.info(f"[SID:{session_id}] STREAMER: Yielding stream_end.")
            yield f"data: {json.dumps({'type': 'stream_end', 'session_id': session_id, 'message': 'Response stream complete.'})}\n\n"

    except Exception as e:
        log_entry["error_detail"] = log_entry.get("error_detail") or f"Main streamer error: {type(e).__name__} - {e}"
        app_logger.error(f"[SID:{session_id}] STREAMER: Unhandled error: {log_entry['error_detail']}", exc_info=DEBUG_MODE)
        if not error_sent_to_client:
            try:
                yield f"data: {json.dumps({'type': 'error', 'session_id': session_id, 'message': 'An internal server error occurred.'})}\n\n"
                error_sent_to_client = True
            except Exception as e_yield:
                app_logger.warning(f"[SID:{session_id}] STREAMER: Could not yield final error message: {e_yield}")
    finally:
        app_logger.info(f"[SID:{session_id}] STREAMER: Finally block. SDK task handle: {sdk_thread_task_handle if sdk_thread_task_handle else 'Not created'}")
        if sdk_thread_task_handle:
            if not sdk_thread_task_handle.done():
                app_logger.info(f"[SID:{session_id}] STREAMER: SDK task still running. Waiting for completion or timeout ({SDK_THREAD_FINISH_TIMEOUT_SECONDS}s).")
                try:
                    await asyncio.wait_for(sdk_thread_task_handle, timeout=SDK_THREAD_FINISH_TIMEOUT_SECONDS)
                    app_logger.info(f"[SID:{session_id}] STREAMER: SDK task completed in finally block.")
                except asyncio.TimeoutError:
                    app_logger.error(f"[SID:{session_id}] STREAMER: SDK thread task timed out in finally. Attempting to cancel.")
                    log_entry["error_detail"] = log_entry.get("error_detail") or "SDK thread cleanup timeout."
                    sdk_thread_task_handle.cancel()
                    try:
                        await sdk_thread_task_handle
                    except asyncio.CancelledError:
                        app_logger.info(f"[SID:{session_id}] STREAMER: SDK thread task successfully cancelled.")
                    except Exception as e_cancel:
                        app_logger.error(f"[SID:{session_id}] STREAMER: Exception while awaiting cancelled SDK task: {e_cancel}")
                except Exception as e:
                    app_logger.error(f"[SID:{session_id}] STREAMER: SDK thread task error during finally wait: {e}")
                    log_entry["error_detail"] = log_entry.get("error_detail") or f"SDK thread cleanup error: {e}"
            else:
                 app_logger.info(f"[SID:{session_id}] STREAMER: SDK task was already done.")
                 try:
                     sdk_thread_task_handle.result()
                 except Exception as e_task_done_err:
                     app_logger.error(f"[SID:{session_id}] STREAMER: SDK task had an exception: {e_task_done_err}")
                     log_entry["error_detail"] = log_entry.get("error_detail") or f"SDK thread task failed: {e_task_done_err}"
        
        log_entry["processing_time_ms"] = round((time.monotonic() - start_time) * 1000, 2)
        trimmed_response = full_response_text.strip()
        log_entry["llm_response_length"] = len(trimmed_response)
        log_entry["llm_response_preview"] = (trimmed_response[:70] + "...") if len(trimmed_response) > 70 else trimmed_response
        if not trimmed_response and log_entry.get("error_detail") and "Error" not in log_entry["llm_response_preview"]: log_entry["llm_response_preview"] = "Error"
        elif not trimmed_response and not log_entry.get("error_detail"): log_entry["llm_response_preview"] = "Empty"
        
        (interaction_logger.error if log_entry.get("error_detail") else interaction_logger.info)(json.dumps(log_entry, default=str))
        app_logger.info(f"[SID:{session_id}] STREAMER: Finished. Persisting messages.")
        if question: _persist_chat_messages(user_id, session_id, question, trimmed_response, log_entry)

@app.post("/chat", dependencies=[Depends(rate_limit_dependency)])
async def chat_endpoint(
    request_data: QueryRequest,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    start_time_req = time.monotonic()
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=403, detail="User identifier missing from token.")

    http_request.state.user_id_for_rate_limit = user_id
    client_ip = http_request.client.host if http_request.client else "N/A"


    if not (llm_model_operational and vertex_ai_client_wrapper_instance and VERTEX_GEMINI_MODEL_NAME):
        app_logger.error(f"Chat attempt failed: LLM service unavailable (Op: {llm_model_operational}, Client: {bool(vertex_ai_client_wrapper_instance)}, Model: {VERTEX_GEMINI_MODEL_NAME})")
        raise HTTPException(status_code=503, detail="LLM service is currently unavailable. Please try again later.")
    if not request_data.question or not request_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request_data.session_id or str(uuid4())

    vertex_chat_history: List[VertexContent] = []
    if request_data.conversation_history and VERTEX_AI_SDK_AVAILABLE:
        for msg_input in request_data.conversation_history:
            if msg_input.role in ["user", "model"] and msg_input.text and msg_input.text.strip():
                try:
                    vertex_chat_history.append(VertexContent(role=msg_input.role.lower(), parts=[VertexPart.from_text(msg_input.text)]))
                except Exception as e_hist:
                     app_logger.warning(f"Skipping invalid history item for SID {session_id} (User: {user_id}): {e_hist}", exc_info=DEBUG_MODE)

    full_llm_contents: List[VertexContent] = list(vertex_chat_history)
    if VERTEX_AI_SDK_AVAILABLE :
        user_question_part = VertexPart.from_text(request_data.question)
        full_llm_contents.append(VertexContent(role="user", parts=[user_question_part]))
    else:
        raise HTTPException(status_code=503, detail="LLM SDK not available.")


    gen_config_llm_dict = {"temperature": 0.3, "candidate_count": 1}

    final_gen_config: Union[VertexGenerationConfig, Dict[str, Any]]
    if VERTEX_AI_SDK_AVAILABLE:
        final_gen_config = VertexGenerationConfig(**gen_config_llm_dict)
    else:
        final_gen_config = gen_config_llm_dict

    safety_settings_llm = None
    if not DISABLE_SAFETY_SETTINGS_FOR_FINETUNED_MODEL:
        safety_settings_llm = get_vertex_safety_settings()

    # Use system instruction from JSON if available
    system_instruction_llm = LLM_SYSTEM_INSTRUCTION_FROM_JSON if LLM_SYSTEM_INSTRUCTION_FROM_JSON else None
    if system_instruction_llm:
        app_logger.info(f"[SID:{session_id}] Using system instruction loaded from JSON.")
    else:
        app_logger.info(f"[SID:{session_id}] No system instruction will be used (either not configured, file error, or empty in JSON).")


    llm_api_args = {
        "model_name": VERTEX_GEMINI_MODEL_NAME,
        "contents": full_llm_contents,
        "generation_config": final_gen_config,
        "safety_settings": safety_settings_llm,
        "system_instruction": system_instruction_llm,
        "stream": request_data.stream
    }
    if rag_retrieval_tool_global:
        llm_api_args["tools"] = [rag_retrieval_tool_global]

    if request_data.stream:
        if not VERTEX_AI_SDK_AVAILABLE:
            raise HTTPException(status_code=503, detail="Streaming not available due to SDK issue.")

        return StreamingResponse(
            _stream_llm_response_generator(llm_api_args, session_id, user_id, request_data.question, client_ip),
            media_type="text/event-stream"
        )

    non_stream_log_entry = {
        "session_id": session_id, "user_id": user_id,
        "query_length": len(request_data.question),
        "query_preview": request_data.question[:50] + ("..." if len(request_data.question)>50 else ""),
        "rag_active": bool(rag_retrieval_tool_global), "llm_model": VERTEX_GEMINI_MODEL_NAME,
        "stream_request": False,
        "client_ip": client_ip,
        "error_detail": None,
        "llm_prompt_feedback_block_reason": "NONE",
        "llm_finish_reason": "UNKNOWN",
        "system_instruction_used": bool(system_instruction_llm)
    }
    llm_response_text = "Error: Could not generate an AI response due to an internal issue."

    try:
        if not vertex_ai_client_wrapper_instance:
            app_logger.critical("Vertex AI client not initialized for non-streaming chat.")
            raise RuntimeError("Vertex AI client not initialized.")

        llm_api_args_non_stream = llm_api_args.copy()
        llm_api_args_non_stream["stream"] = False
        llm_api_args_non_stream["request_options"] = {"timeout": GENERATIVE_MODEL_API_TIMEOUT_SECONDS}
        llm_api_args_non_stream["session_id_for_log"] = session_id


        sdk_response_future = asyncio.to_thread(
            vertex_ai_client_wrapper_instance.generate_content, **llm_api_args_non_stream
        )
        sdk_response: GenerationResponse = await asyncio.wait_for(sdk_response_future, timeout=GENERATIVE_MODEL_API_TIMEOUT_SECONDS + 5)

        _prompt_feedback = getattr(sdk_response, 'prompt_feedback', None)
        non_stream_log_entry["llm_prompt_feedback_block_reason"] = str(getattr(_prompt_feedback, 'block_reason', "NONE")).upper()
        if _prompt_feedback and _prompt_feedback.block_reason != 0:
            message = getattr(_prompt_feedback, 'block_reason_message', "Request blocked by content policies.")
            llm_response_text = f"Request blocked by content policies: {message}"
            non_stream_log_entry["error_detail"] = llm_response_text
        elif not VERTEX_AI_SDK_AVAILABLE or not sdk_response.candidates:
             llm_response_text = "AI model returned no response candidates."
             non_stream_log_entry["error_detail"] = llm_response_text
        else:
            candidate = sdk_response.candidates[0]
            finish_reason_val = getattr(candidate, 'finish_reason', None)
            non_stream_log_entry["llm_finish_reason"] = str(getattr(finish_reason_val, 'name', "UNKNOWN")).upper()

            if finish_reason_val and finish_reason_val.name == "SAFETY":
                llm_response_text = "AI response content blocked due to safety policies."
                non_stream_log_entry["error_detail"] = llm_response_text
            elif finish_reason_val and finish_reason_val.name == "RECITATION":
                llm_response_text = "AI response content blocked due to recitation policy."
                non_stream_log_entry["error_detail"] = llm_response_text
            else:
                extracted_text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text') and p.text]
                extracted_text = "".join(extracted_text_parts).strip()

                if extracted_text:
                    llm_response_text = extracted_text
                    if finish_reason_val and finish_reason_val.name == "MAX_TOKENS":
                        llm_response_text += " (Note: Response may be truncated due to maximum token limit.)"
                elif finish_reason_val and finish_reason_val.name == "STOP" and not extracted_text:
                    llm_response_text = "No specific answer found based on the available context."
                else:
                    llm_response_text = f"AI model provided an empty or non-text response. (Finish Reason: {non_stream_log_entry['llm_finish_reason']})"
                    if not non_stream_log_entry["error_detail"]: non_stream_log_entry["error_detail"] = llm_response_text

            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                gm = candidate.grounding_metadata
                if hasattr(gm, 'retrieval_queries') and gm.retrieval_queries:
                    non_stream_log_entry["rag_retrieval_queries_count"] = len(gm.retrieval_queries)

            _usage_metadata = getattr(sdk_response, 'usage_metadata', None)
            if _usage_metadata:
                non_stream_log_entry["llm_token_counts"] = {
                    "prompt_token_count": getattr(_usage_metadata,'prompt_token_count',0),
                    "candidates_token_count": getattr(_usage_metadata,'candidates_token_count',0),
                    "total_token_count": getattr(_usage_metadata,'total_token_count',0),
                }

    except asyncio.TimeoutError:
        timeout_val = GENERATIVE_MODEL_API_TIMEOUT_SECONDS + 5
        non_stream_log_entry["error_detail"] = f"Timeout after {timeout_val} s for non-streaming LLM call."
        app_logger.warning(f"[SID:{session_id}] LLM request timeout: {non_stream_log_entry['error_detail']}")
        raise HTTPException(status_code=504, detail="AI service request timed out. Please try again.")
    except google_api_core_exceptions.DeadlineExceeded as e_google_deadline:
        error_message = getattr(e_google_deadline, 'message', str(e_google_deadline))
        non_stream_log_entry["error_detail"] = f"GoogleAPIDeadlineExceeded: {type(e_google_deadline).__name__} - {error_message}"
        app_logger.error(f"[SID:{session_id}] Google API DeadlineExceeded: {non_stream_log_entry['error_detail']}", exc_info=DEBUG_MODE)
        raise HTTPException(status_code=408, detail=f"AI service request timed out (SDK level): {error_message}")
    except google_api_core_exceptions.GoogleAPIError as e_google:
        error_message = getattr(e_google, 'message', str(e_google))
        non_stream_log_entry["error_detail"] = f"GoogleAPIError: {type(e_google).__name__} - {error_message}"
        status_code = getattr(e_google, 'code', 502)
        detail_msg = "An error occurred with the AI service."
        if isinstance(e_google, google_api_core_exceptions.InvalidArgument): detail_msg = f"AI service request failed due to an invalid argument: {error_message}"; status_code = 400
        elif isinstance(e_google, google_api_core_exceptions.PermissionDenied): detail_msg = "AI service request failed (permission issue)."; status_code = 500
        elif isinstance(e_google, google_api_core_exceptions.ResourceExhausted): detail_msg = f"AI service quota exceeded: {error_message}."; status_code = 429
        app_logger.error(f"[SID:{session_id}] Google API Error: {non_stream_log_entry['error_detail']}", exc_info=DEBUG_MODE)
        raise HTTPException(status_code=status_code if isinstance(status_code, int) else 502, detail=detail_msg)
    except Exception as e_unexp:
        non_stream_log_entry["error_detail"] = f"Unexpected error: {type(e_unexp).__name__} - {str(e_unexp)}"
        app_logger.error(f"[SID:{session_id}] Unexpected LLM error: {non_stream_log_entry['error_detail']}", exc_info=DEBUG_MODE)
        raise HTTPException(status_code=500, detail="An internal server error occurred while generating the AI response.")
    finally:
        non_stream_log_entry["processing_time_ms"] = round((time.monotonic() - start_time_req) * 1000, 2)
        llm_response_text_trimmed = llm_response_text.strip()
        non_stream_log_entry["llm_response_length"] = len(llm_response_text_trimmed)
        non_stream_log_entry["llm_response_preview"] = (llm_response_text_trimmed[:70] + "...") if len(llm_response_text_trimmed) > 70 else llm_response_text_trimmed
        if not llm_response_text_trimmed and non_stream_log_entry.get("error_detail"): non_stream_log_entry["llm_response_preview"] = "Error"
        elif not llm_response_text_trimmed: non_stream_log_entry["llm_response_preview"] = "Empty"
        (interaction_logger.error if non_stream_log_entry.get("error_detail") else interaction_logger.info)(json.dumps(non_stream_log_entry, default=str))

    if request_data.question:
        _persist_chat_messages(user_id, session_id, request_data.question, llm_response_text_trimmed, non_stream_log_entry)
    response_payload = ChatResponse(answer=llm_response_text, session_id=session_id)
    if DEBUG_MODE: response_payload.debug_info = {k:v for k,v in non_stream_log_entry.items() if k not in ['user_id']}
    return response_payload

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload_dev = os.getenv("API_RELOAD_DEV", "False").lower() == "true"
    log_level_uvicorn = "debug" if (DEBUG_MODE or reload_dev) else "info"
    app_module_name = pathlib.Path(__file__).stem
    uvicorn.run(f"{app_module_name}:app", host=host, port=port, reload=reload_dev, log_level=log_level_uvicorn)