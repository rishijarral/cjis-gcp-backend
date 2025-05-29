import asyncio
import json
import logging
import os
import time
import pathlib
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
            if name in ("GoogleAPIError", "InvalidArgument", "PermissionDenied", "ResourceExhausted"):
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

VERTEX_GEMINI_MODEL_NAME = os.getenv("VERTEX_GEMINI_MODEL_NAME", "gemini-2.0-flash").replace("-latest", "")

RAG_CORPUS_NAME = os.getenv("RAG_CORPUS_NAME")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

LLM_SYSTEM_INSTRUCTION = os.getenv("RAG_FT_SYSTEM_INSTRUCTION_FOR_API", "").strip()
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
        stream: bool = False
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
            response = gen_model_instance.generate_content(
                contents=vertex_contents_final,
                generation_config=vertex_generation_config_obj,
                safety_settings=safety_settings,
                tools=tools,
                tool_config=tool_config,
                stream=stream
            )
            return response
        except Exception as e:
            app_logger.error(f"Error calling Vertex AI generate_content (model: {model_name}): {e}", exc_info=DEBUG_MODE)
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
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience=SUPABASE_JWT_AUD or None,
                options=decode_options
            )
        elif alg == "RS256":
            if not _supabase_jwk_client:
                app_logger.error("Auth: Supabase JWKS client for RS256 not initialized.")
                raise HTTPException(status_code=500, detail="Auth (RS256 JWKS client not ready).")
            signing_key = _supabase_jwk_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=SUPABASE_JWT_AUD or None,
                options=decode_options
            )
        else:
            raise HTTPException(status_code=401, detail=f"Unsupported token algorithm: {alg}.")

        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("role")
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid token audience.")
    except jwt.MissingRequiredClaimError as e_claim_missing:
        app_logger.warning(f"Token missing required claim: {e_claim_missing} (alg: {alg})", exc_info=DEBUG_MODE)
        raise HTTPException(status_code=401, detail=f"Token missing claim: {e_claim_missing}.")
    except jwt.InvalidTokenError as e_invalid_token:
        app_logger.warning(f"Invalid token: {e_invalid_token} (alg: {alg})", exc_info=DEBUG_MODE)
        raise HTTPException(status_code=401, detail=f"Token validation error: {e_invalid_token}.")
    except Exception as e_unexp:
        app_logger.error(f"Unexpected token validation error: {e_unexp} (alg: {alg})", exc_info=DEBUG_MODE)
        raise HTTPException(status_code=500, detail="Error during token validation.")

async def rate_limit_dependency(request: Request):
    user_id_for_key = getattr(request.state, "user_id_for_rate_limit", None)
    client_key = user_id_for_key or (request.client.host if request.client else "unknown_ip")

    current_time = time.monotonic()
    REQUEST_TIMESTAMPS.setdefault(client_key, [])

    timestamps = [ts for ts in REQUEST_TIMESTAMPS[client_key] if ts > current_time - RATE_LIMIT_WINDOW]

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        wait_time = (timestamps[0] + RATE_LIMIT_WINDOW) - current_time if timestamps else RATE_LIMIT_WINDOW
        raise HTTPException(
            status_code=429,
            detail=f"Too Many Requests. Try again in {max(0, int(wait_time)) + 1} seconds."
        )

    timestamps.append(current_time)
    REQUEST_TIMESTAMPS[client_key] = timestamps

def _get_vertex_safety_threshold_enum_val() -> Union[VertexHarmBlockThreshold, str]:
    if not VERTEX_AI_SDK_AVAILABLE: return GEMINI_SAFETY_SETTINGS_THRESHOLD.upper()

    mapping = {
        "BLOCK_NONE": VertexHarmBlockThreshold.BLOCK_NONE,
        "BLOCK_ONLY_HIGH": VertexHarmBlockThreshold.BLOCK_ONLY_HIGH,
        "BLOCK_MEDIUM_AND_ABOVE": VertexHarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "BLOCK_LOW_AND_ABOVE": VertexHarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    }
    default_threshold = VertexHarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    enum_val = mapping.get(GEMINI_SAFETY_SETTINGS_THRESHOLD.upper(), default_threshold)

    if isinstance(enum_val, str): # Should not happen if VERTEX_AI_SDK_AVAILABLE is true
        app_logger.warning(f"Safety threshold string '{GEMINI_SAFETY_SETTINGS_THRESHOLD}' mapped to string '{enum_val}', not enum. Defaulting.")
        return default_threshold
    return enum_val

def get_vertex_safety_settings() -> Optional[Dict[VertexHarmCategory, VertexHarmBlockThreshold]]:
    if not VERTEX_AI_SDK_AVAILABLE or not hasattr(VertexHarmCategory, "HARM_CATEGORY_HARASSMENT"):
        return None

    threshold = _get_vertex_safety_threshold_enum_val()
    if isinstance(threshold, str): # Should indicate an issue if SDK was available
        app_logger.error(f"Could not map safety threshold string '{threshold}' to a valid VertexHarmBlockThreshold enum.")
        return None

    categories = [
        VertexHarmCategory.HARM_CATEGORY_HARASSMENT,
        VertexHarmCategory.HARM_CATEGORY_HATE_SPEECH,
        VertexHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        VertexHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
    ]
    return {category: threshold for category in categories}

CHAT_SESSIONS_DIR = pathlib.Path(os.getenv("CHAT_SESSIONS_BASE_PATH", "/app/data/chat_sessions"))

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global gcp_project_configured, vertex_ai_client_wrapper_instance, vertex_ai_configured_successfully
    global llm_model_operational, _supabase_jwk_client, rag_retrieval_tool_global

    try:
        CHAT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        app_logger.info(f"Chat sessions directory ensured at: {CHAT_SESSIONS_DIR}")
    except OSError as e:
        app_logger.error(f"Could not create chat sessions directory {CHAT_SESSIONS_DIR}: {e}", exc_info=DEBUG_MODE)

    gcp_project_configured = VERTEX_PROJECT_ID_ENV
    if not gcp_project_configured:
        try:
            from google.auth import default as google_auth_default # Local import
            _, gcp_project_configured = google_auth_default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
            if gcp_project_configured:
                 app_logger.info(f"GCP Project ID determined via default credentials: {gcp_project_configured}")
        except Exception as e_auth:
            app_logger.warning(f"Could not determine GCP Project ID via default credentials: {e_auth}")

    if not gcp_project_configured:
        app_logger.error("CRITICAL: GCP Project ID could not be determined. Vertex AI functionality will be impaired.")

    if VERTEX_AI_SDK_AVAILABLE and gcp_project_configured and VERTEX_LOCATION:
        try:
            vertex_ai_client_wrapper_instance = VertexAIClientWrapper(project=gcp_project_configured, location=VERTEX_LOCATION)
            vertex_ai_configured_successfully = True
            app_logger.info(f"Vertex AI Client initialized for project '{gcp_project_configured}' in '{VERTEX_LOCATION}'.")

            llm_model_operational = bool(VERTEX_GEMINI_MODEL_NAME)
            if not llm_model_operational:
                app_logger.error("VERTEX_GEMINI_MODEL_NAME not set. LLM functionalities will be disabled.")
            else:
                app_logger.info(f"Using LLM model: {VERTEX_GEMINI_MODEL_NAME}")

        except Exception as e:
            app_logger.error(f"Failed to initialize VertexAIClientWrapper: {e}", exc_info=DEBUG_MODE)
            vertex_ai_configured_successfully = False
            llm_model_operational = False
    else:
        missing_configs = []
        if not VERTEX_AI_SDK_AVAILABLE: missing_configs.append("Vertex AI SDK")
        if not gcp_project_configured: missing_configs.append("GCP Project ID")
        if not VERTEX_LOCATION: missing_configs.append("Vertex Location")
        app_logger.warning(f"Vertex AI features disabled due to missing: {', '.join(missing_configs)}.")
        vertex_ai_configured_successfully = False
        llm_model_operational = False


    if SUPABASE_URL:
        try:
            jwks_uri = SUPABASE_URL.rstrip('/') + "/auth/v1/.well-known/jwks.json"
            _supabase_jwk_client = PyJWKClient(jwks_uri, cache_jwk_set=True, lifespan=3600)
            app_logger.info(f"Supabase JWKS Client initialized for URL: {jwks_uri}")
        except Exception as e:
            app_logger.error(f"Failed to initialize Supabase JWK client (for RS256): {e}", exc_info=DEBUG_MODE)

    if not SUPABASE_JWT_SECRET and not _supabase_jwk_client :
         app_logger.warning("Auth: Neither Supabase JWT Secret (for HS256) nor Supabase URL (for RS256 JWKS) is configured. Authentication may fail.")
    elif SUPABASE_JWT_SECRET:
        app_logger.info("Supabase JWT Secret (for HS256) is configured.")


    if vertex_ai_configured_successfully and VERTEX_AI_SDK_AVAILABLE and RAG_CORPUS_NAME:
        if not RAG_CORPUS_NAME.startswith("projects/"):
            app_logger.critical(f"RAG_CORPUS_NAME ('{RAG_CORPUS_NAME}') must be a full resource name (e.g. projects/...). RAG tool NOT created.")
        else:
            try:
                rag_retrieval = rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[
                            rag.RagResource(rag_corpus=RAG_CORPUS_NAME)
                        ],
                        rag_retrieval_config=rag.RagRetrievalConfig(top_k=RAG_TOP_K)
                    )
                )
                rag_retrieval_tool_global = VertexTool.from_retrieval(retrieval=rag_retrieval)
                app_logger.info(f"RAG Engine tool created for corpus: {RAG_CORPUS_NAME} with top_k={RAG_TOP_K}.")
            except AttributeError as e_attr:
                app_logger.error(f"Failed to create RAG Engine tool due to SDK component issue (possibly placeholder): {e_attr}", exc_info=DEBUG_MODE)
            except Exception as e:
                app_logger.error(f"Failed to create RAG Engine tool for corpus '{RAG_CORPUS_NAME}': {e}", exc_info=DEBUG_MODE)
    elif RAG_CORPUS_NAME and not vertex_ai_configured_successfully:
        app_logger.warning(f"RAG_CORPUS_NAME ('{RAG_CORPUS_NAME}') is set, but Vertex AI is not configured. RAG tool NOT created.")

    yield

    app_logger.info("Application shutdown.")

app = FastAPI(
    title="RAG Chat API with Vertex AI RAG Engine",
    version="2.3.3-prod",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    app_logger.error(f"Request validation error for {request.url.path}: {exc.errors()}", exc_info=DEBUG_MODE)
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors()), "message": "Invalid input format. Please check your request body."}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Utility"], include_in_schema=False)
async def root():
    return {"message": f"{app.title} v{app.version} is running."}

@app.get("/health", tags=["Utility"])
async def health_check():
    auth_service_ok = bool(_supabase_jwk_client or SUPABASE_JWT_SECRET)
    overall_status = "healthy"
    if not (vertex_ai_configured_successfully and llm_model_operational and auth_service_ok):
        overall_status = "degraded"
    if not (vertex_ai_configured_successfully and llm_model_operational):
        overall_status = "unhealthy"

    return {
        "service_status": overall_status,
        "version": app.version,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "details": {
            "gcp_project_id": gcp_project_configured or "N/A",
            "vertex_sdk_initialized": vertex_ai_configured_successfully,
            "llm_model": VERTEX_GEMINI_MODEL_NAME if llm_model_operational else "N/A",
            "llm_operational": llm_model_operational,
            "rag_tool_configured": bool(RAG_CORPUS_NAME),
            "rag_tool_active": bool(rag_retrieval_tool_global),
            "auth_service_configured": auth_service_ok,
            "debug_mode": DEBUG_MODE
        }
    }

@app.get("/api/sessions/", tags=["Session Management"], dependencies=[Depends(rate_limit_dependency)])
async def list_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID (sub claim) not found in token.")

    user_session_dir = CHAT_SESSIONS_DIR / str(user_id)
    if not user_session_dir.exists():
        return {"sessions": [], "pagination": {"total": 0, "offset": offset, "limit": limit, "has_more": False}}

    try:
        session_files = sorted(
            [f for f in user_session_dir.glob("*.jsonl") if f.is_file()],
            key=lambda x: x.stat().st_mtime, reverse=True
        )
    except OSError as e:
        app_logger.error(f"Error listing session files for user {user_id}: {e}", exc_info=DEBUG_MODE)
        raise HTTPException(status_code=500, detail="Error accessing session data.")

    sessions_metadata = []
    for f_path in session_files:
        session_id = f_path.stem
        try:
            created_at_ts = f_path.stat().st_ctime
            title = f"Chat on {time.strftime('%Y-%m-%d', time.gmtime(created_at_ts))}"

            sessions_metadata.append({
                "id": session_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(created_at_ts)),
                "title": title
            })
        except Exception as e_stat:
            app_logger.warning(f"Could not stat/process session file {f_path}: {e_stat}")
            continue

    paginated_sessions = sessions_metadata[offset : offset + limit]
    total_sessions = len(sessions_metadata)
    return {
        "sessions": paginated_sessions,
        "pagination": {
            "total": total_sessions,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total_sessions
        }
    }

@app.get("/api/sessions/{session_id}", tags=["Session Management"], dependencies=[Depends(rate_limit_dependency)])
async def get_session_messages(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in token.")

    safe_session_filename = f"{pathlib.Path(session_id).name}.jsonl"
    session_file = CHAT_SESSIONS_DIR / str(user_id) / safe_session_filename

    if not session_file.exists() or not session_file.is_file():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    try:
        messages: List[Dict[str,str]] = []
        with session_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    msg_data = json.loads(line)
                    if "role" in msg_data and "text" in msg_data and \
                       isinstance(msg_data["role"], str) and isinstance(msg_data["text"], str):
                        messages.append(ChatMessageInput(role=msg_data["role"], text=msg_data["text"]).model_dump())
                    else:
                        app_logger.warning(f"Skipping malformed message (missing fields or wrong types) in session {session_id} for user {user_id}, line {line_num}.")
                except json.JSONDecodeError:
                    app_logger.warning(f"Skipping malformed JSON line in session {session_id} for user {user_id}, line {line_num}.")
                except Exception as e_parse: # Catches Pydantic validation errors too
                    app_logger.warning(f"Skipping invalid message data in session {session_id} for user {user_id}, line {line_num}: {e_parse}")
        return messages
    except IOError as e_io:
        app_logger.error(f"Failed to read session file {session_file}: {e_io}", exc_info=DEBUG_MODE)
        raise HTTPException(status_code=500, detail="Error reading session file.")
    except Exception as e_gen:
        app_logger.error(f"Unexpected error loading session {session_id}: {e_gen}", exc_info=DEBUG_MODE)
        raise HTTPException(status_code=500, detail="Internal server error retrieving session messages.")

def _persist_chat_messages(user_id: str, session_id: str, question: str, model_answer: str, interaction_log_entry: Dict[str, Any]) -> None:
    safe_session_filename = f"{pathlib.Path(session_id).name}.jsonl"
    user_session_dir = CHAT_SESSIONS_DIR / str(user_id)

    try:
        user_session_dir.mkdir(parents=True, exist_ok=True)
        session_file = user_session_dir / safe_session_filename

        text_to_log_for_model = model_answer
        if not model_answer.strip() and interaction_log_entry.get("error_detail"):
            text_to_log_for_model = str(interaction_log_entry.get("error_detail", "Error processing request"))
        elif not model_answer.strip() and interaction_log_entry.get("llm_prompt_feedback_block_reason", "NONE") != "NONE":
             text_to_log_for_model = f"Request blocked: {interaction_log_entry.get('llm_prompt_feedback_block_reason')}"
        elif not model_answer.strip() and interaction_log_entry.get("llm_finish_reason") not in ["STOP", "NONE", "UNKNOWN"]:
             text_to_log_for_model = f"Response issue: {interaction_log_entry.get('llm_finish_reason')}"


        with session_file.open("a", encoding="utf-8") as sf:
            sf.write(json.dumps({"role": "user", "text": question}) + "\n")
            sf.write(json.dumps({"role": "model", "text": text_to_log_for_model.strip()}) + "\n")
    except IOError as e_io:
        app_logger.error(f"Error persisting chat history to session dir {user_session_dir}/{safe_session_filename}: {e_io}", exc_info=DEBUG_MODE)
    except Exception as e_gen:
        app_logger.error(f"Unexpected error persisting chat history for session {session_id}: {e_gen}", exc_info=DEBUG_MODE)

async def _stream_llm_response_generator(
    llm_api_args: Dict[str, Any],
    session_id: str,
    user_id: str,
    question: str,
    client_ip: str
) -> AsyncGenerator[str, None]:
    start_time_stream = time.monotonic()
    full_response_text = ""
    stream_successfully_concluded = False
    response_emitted_error_type_message = False

    log_entry_stream: Dict[str, Any] = {
        "session_id": session_id, "user_id": user_id,
        "query_length": len(question),
        "query_preview": question[:50] + "..." if len(question) > 50 else question,
        "rag_active": "tools" in llm_api_args and llm_api_args["tools"] is not None,
        "llm_model": llm_api_args.get("model_name"), "stream_request": True,
        "client_ip": client_ip, "error_detail": None,
        "llm_prompt_feedback_block_reason": "NONE", "llm_finish_reason": "UNKNOWN",
    }

    response_queue: asyncio.Queue[Tuple[Optional[_SDKGenerationResponseStreamChunk], Optional[BaseException]]] = asyncio.Queue()
    current_loop = asyncio.get_event_loop()

    sdk_thread_task = None

    try:
        processing_message = {"type": "processing", "session_id": session_id, "message": "Processing your request..."}
        yield json.dumps(processing_message) + "\n"

        def sdk_iterate_in_thread():
            try:
                if not vertex_ai_client_wrapper_instance:
                     raise RuntimeError("Vertex AI client not initialized for streaming thread.")

                sdk_stream_iterator = vertex_ai_client_wrapper_instance.generate_content(**llm_api_args)
                for chunk_sdk in sdk_stream_iterator:
                    current_loop.call_soon_threadsafe(response_queue.put_nowait, (chunk_sdk, None))
            except Exception as e_sdk_thread:
                current_loop.call_soon_threadsafe(response_queue.put_nowait, (None, e_sdk_thread))
            finally:
                current_loop.call_soon_threadsafe(response_queue.put_nowait, (None, StopAsyncIteration()))

        sdk_thread_task = asyncio.to_thread(sdk_iterate_in_thread)

        stream_had_candidates_with_text = False
        final_sdk_response_obj: Optional[_SDKGenerationResponseStreamChunk] = None

        while True:
            try:
                chunk, error = await asyncio.wait_for(response_queue.get(), timeout=STREAMING_RESPONSE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                error_msg = f"Streaming response queue timed out after {STREAMING_RESPONSE_TIMEOUT_SECONDS}s."
                log_entry_stream["error_detail"] = error_msg
                if not response_emitted_error_type_message:
                    yield json.dumps({"type": "error", "session_id": session_id, "message": error_msg}) + "\n"
                    response_emitted_error_type_message = True
                return

            if isinstance(error, StopAsyncIteration):
                stream_successfully_concluded = True
                break
            if error:
                raise error

            if chunk and VERTEX_AI_SDK_AVAILABLE:
                final_sdk_response_obj = chunk
                if chunk.candidates:
                    stream_had_candidates_with_text = True
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            full_response_text += part.text
                            yield json.dumps({"type": "chunk", "session_id": session_id, "data": part.text}) + "\n"
            elif chunk is None and error is None:
                app_logger.warning(f"[SID:{session_id}] Received (None, None) from queue, implies an issue.")


        if stream_successfully_concluded:
            if final_sdk_response_obj and VERTEX_AI_SDK_AVAILABLE:
                _prompt_feedback = getattr(final_sdk_response_obj, 'prompt_feedback', None)
                _candidates_list = getattr(final_sdk_response_obj, 'candidates', [])

                log_entry_stream["llm_prompt_feedback_block_reason"] = str(getattr(_prompt_feedback, 'block_reason', "NONE")).upper()
                if _prompt_feedback and _prompt_feedback.block_reason != 0:
                    error_msg = getattr(_prompt_feedback, 'block_reason_message', "Request blocked by content policies.")
                    log_entry_stream["error_detail"] = log_entry_stream.get("error_detail") or error_msg
                    if not stream_had_candidates_with_text and not response_emitted_error_type_message:
                        yield json.dumps({"type": "error", "session_id": session_id, "message": error_msg}) + "\n"
                        response_emitted_error_type_message = True
                    full_response_text = full_response_text or error_msg
                    stream_successfully_concluded = False

                elif not stream_had_candidates_with_text and not full_response_text.strip():
                    is_content_empty_in_final_candidate = True
                    if _candidates_list and _candidates_list[0] and hasattr(_candidates_list[0],'content') and hasattr(_candidates_list[0].content,'parts'):
                         if any(hasattr(p,'text') and p.text for p in _candidates_list[0].content.parts):
                             is_content_empty_in_final_candidate = False

                    if is_content_empty_in_final_candidate:
                        error_msg = "AI model returned no response candidates or content."
                        log_entry_stream["error_detail"] = log_entry_stream.get("error_detail") or error_msg
                        if not response_emitted_error_type_message:
                            yield json.dumps({"type": "error", "session_id": session_id, "message": error_msg}) + "\n"
                            response_emitted_error_type_message = True
                        full_response_text = full_response_text or error_msg
                        stream_successfully_concluded = False

                elif _candidates_list and _candidates_list[0]:
                    candidate = _candidates_list[0]
                    finish_reason_val = getattr(candidate, 'finish_reason', None)
                    finish_reason_name = str(getattr(finish_reason_val, 'name', "UNKNOWN")).upper()
                    log_entry_stream["llm_finish_reason"] = finish_reason_name

                    if finish_reason_name == "SAFETY":
                        safety_msg = "AI response content blocked due to safety policies."
                        log_entry_stream["error_detail"] = log_entry_stream.get("error_detail") or safety_msg
                        if not stream_had_candidates_with_text and not response_emitted_error_type_message:
                             yield json.dumps({"type": "error", "session_id": session_id, "message": safety_msg}) + "\n"
                             response_emitted_error_type_message = True
                        full_response_text = full_response_text or safety_msg
                        stream_successfully_concluded = False
                    elif finish_reason_name == "RECITATION":
                        recitation_msg = "AI response content blocked due to recitation policy."
                        log_entry_stream["error_detail"] = log_entry_stream.get("error_detail") or recitation_msg
                        if not stream_had_candidates_with_text and not response_emitted_error_type_message:
                            yield json.dumps({"type": "error", "session_id": session_id, "message": recitation_msg}) + "\n"
                            response_emitted_error_type_message = True
                        full_response_text = full_response_text or recitation_msg
                        stream_successfully_concluded = False
                    elif finish_reason_name == "MAX_TOKENS" and full_response_text.strip():
                        yield json.dumps({"type": "info", "session_id": session_id, "message": "Response may be truncated (max tokens reached)." }) + "\n"
                    elif finish_reason_name == "STOP" and not full_response_text.strip() and stream_had_candidates_with_text:
                        _empty_stop_msg = "No specific answer found based on the available context."
                        full_response_text = _empty_stop_msg
                        if not (log_entry_stream["error_detail"] or response_emitted_error_type_message):
                            yield json.dumps({"type": "chunk", "session_id": session_id, "data": _empty_stop_msg}) + "\n"

                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        gm = candidate.grounding_metadata
                        grounding_data_to_send: Dict[str,Any] = {}
                        if hasattr(gm, 'retrieval_queries') and gm.retrieval_queries:
                            grounding_data_to_send["retrieval_queries"] = [q for q in gm.retrieval_queries]
                        if grounding_data_to_send:
                            yield json.dumps({"type": "grounding_info", "session_id": session_id, "data": grounding_data_to_send}) + "\n"
                        log_entry_stream["rag_retrieval_queries_count"] = len(grounding_data_to_send.get("retrieval_queries", []))

                _usage_metadata = getattr(final_sdk_response_obj, 'usage_metadata', None)
                if _usage_metadata:
                    log_entry_stream["llm_token_counts"] = {
                        "prompt_token_count": getattr(_usage_metadata,'prompt_token_count',0),
                        "candidates_token_count": getattr(_usage_metadata,'candidates_token_count',0),
                        "total_token_count": getattr(_usage_metadata,'total_token_count',0),
                    }
            elif not final_sdk_response_obj and not log_entry_stream.get("error_detail"):
                error_msg = "AI model provided no response or stream."
                log_entry_stream["error_detail"] = error_msg
                if not response_emitted_error_type_message:
                    yield json.dumps({"type": "error", "session_id": session_id, "message": error_msg}) + "\n"
                    response_emitted_error_type_message = True
                stream_successfully_concluded = False

            if stream_successfully_concluded:
                yield json.dumps({"type": "stream_end", "session_id": session_id, "message": "Response stream complete."}) + "\n"

    except google_api_core_exceptions.GoogleAPIError as e_google:
        error_message = getattr(e_google, 'message', str(e_google))
        log_entry_stream["error_detail"] = f"GoogleAPIError during stream: {type(e_google).__name__} - {error_message}"
        app_logger.error(f"[SID:{session_id}] Google API Error during stream: {log_entry_stream['error_detail']}", exc_info=DEBUG_MODE)
        if not response_emitted_error_type_message:
            yield json.dumps({"type": "error", "session_id": session_id, "message": f"AI service error: {error_message}"}) + "\n"
            response_emitted_error_type_message = True
    except Exception as e_unexp:
        unexp_error_msg = f"Unexpected error during stream: {type(e_unexp).__name__} - {str(e_unexp)}"
        log_entry_stream["error_detail"] = unexp_error_msg
        app_logger.error(f"[SID:{session_id}] Streaming error: {unexp_error_msg}", exc_info=DEBUG_MODE)
        if not response_emitted_error_type_message:
            yield json.dumps({"type": "error", "session_id": session_id, "message": "An internal server error occurred during streaming."}) + "\n"
            response_emitted_error_type_message = True
    finally:
        if sdk_thread_task:
            try:
                await asyncio.wait_for(sdk_thread_task, timeout=SDK_THREAD_FINISH_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                app_logger.error(f"[SID:{session_id}] Timeout ({SDK_THREAD_FINISH_TIMEOUT_SECONDS}s) waiting for SDK thread to finish cleanup.")
                if not log_entry_stream.get("error_detail"):
                    log_entry_stream["error_detail"] = "SDK thread timed out on completion during cleanup."
            except Exception as e_thread_final_cleanup:
                app_logger.error(f"[SID:{session_id}] Error from SDK thread on final await during cleanup: {e_thread_final_cleanup}", exc_info=DEBUG_MODE)
                if not log_entry_stream.get("error_detail"):
                     log_entry_stream["error_detail"] = f"SDK thread error during cleanup: {str(e_thread_final_cleanup)}"

        log_entry_stream["processing_time_ms"] = round((time.monotonic() - start_time_stream) * 1000, 2)
        full_response_text_trimmed = full_response_text.strip()
        log_entry_stream["llm_response_length"] = len(full_response_text_trimmed)
        log_entry_stream["llm_response_preview"] = (full_response_text_trimmed[:70] + "...") if len(full_response_text_trimmed) > 70 else full_response_text_trimmed

        if not full_response_text_trimmed and log_entry_stream.get("error_detail"):
             log_entry_stream["llm_response_preview"] = "Error"
        elif not full_response_text_trimmed:
             log_entry_stream["llm_response_preview"] = "Empty"

        if log_entry_stream.get("error_detail"):
            interaction_logger.error(json.dumps(log_entry_stream, default=str))
        else:
            interaction_logger.info(json.dumps(log_entry_stream, default=str))

        if question:
            _persist_chat_messages(user_id, session_id, question, full_response_text_trimmed, log_entry_stream)


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

    system_instruction_llm = LLM_SYSTEM_INSTRUCTION or None

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
            media_type="application/x-ndjson"
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
    }
    llm_response_text = "Error: Could not generate an AI response due to an internal issue."

    try:
        if not vertex_ai_client_wrapper_instance:
            app_logger.critical("Vertex AI client not initialized for non-streaming chat.")
            raise RuntimeError("Vertex AI client not initialized.")

        llm_api_args_non_stream = llm_api_args.copy()
        llm_api_args_non_stream["stream"] = False

        sdk_response_future = asyncio.to_thread(
            vertex_ai_client_wrapper_instance.generate_content, **llm_api_args_non_stream
        )
        sdk_response: GenerationResponse = await asyncio.wait_for(sdk_response_future, timeout=GENERATIVE_MODEL_API_TIMEOUT_SECONDS)

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
        non_stream_log_entry["error_detail"] = f"Timeout after {GENERATIVE_MODEL_API_TIMEOUT_SECONDS}s."
        app_logger.warning(f"[SID:{session_id}] LLM request timeout: {non_stream_log_entry['error_detail']}")
        raise HTTPException(status_code=504, detail="AI service request timed out. Please try again.")
    except google_api_core_exceptions.GoogleAPIError as e_google:
        error_message = getattr(e_google, 'message', str(e_google))
        non_stream_log_entry["error_detail"] = f"GoogleAPIError: {type(e_google).__name__} - {error_message}"
        status_code = getattr(e_google, 'code', 502)

        detail_msg = "An error occurred with the AI service."
        if isinstance(e_google, google_api_core_exceptions.InvalidArgument):
            detail_msg = f"AI service request failed due to an invalid argument: {error_message}"
            status_code = 400
        elif isinstance(e_google, google_api_core_exceptions.PermissionDenied):
            detail_msg = "AI service request failed due to a permission issue. Check service account or API key permissions."
            status_code = 500
        elif isinstance(e_google, google_api_core_exceptions.ResourceExhausted):
            detail_msg = f"AI service quota exceeded: {error_message}. Please try again later."
            status_code = 429
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

        if not llm_response_text_trimmed and non_stream_log_entry.get("error_detail"):
             non_stream_log_entry["llm_response_preview"] = "Error"
        elif not llm_response_text_trimmed:
             non_stream_log_entry["llm_response_preview"] = "Empty"

        if non_stream_log_entry.get("error_detail"):
            interaction_logger.error(json.dumps(non_stream_log_entry, default=str))
        else:
            interaction_logger.info(json.dumps(non_stream_log_entry, default=str))

    if request_data.question:
        _persist_chat_messages(user_id, session_id, request_data.question, llm_response_text_trimmed, non_stream_log_entry)

    response_payload = ChatResponse(answer=llm_response_text, session_id=session_id)
    if DEBUG_MODE:
        debug_info_to_send = {k:v for k,v in non_stream_log_entry.items() if k not in ['user_id']}
        response_payload.debug_info = debug_info_to_send

    return response_payload


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload_dev = os.getenv("API_RELOAD_DEV", "False").lower() == "true"
    log_level_uvicorn = "debug" if (DEBUG_MODE or reload_dev) else "info"

    app_module_name = pathlib.Path(__file__).stem

    uvicorn.run(
        f"{app_module_name}:app",
        host=host,
        port=port,
        reload=reload_dev,
        log_level=log_level_uvicorn,
    )









