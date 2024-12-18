import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import FileResponse, RedirectResponse

from deadlock_analytics_api.routers import internal, v1, v2

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logging.getLogger("urllib3").setLevel(logging.WARNING)

if "SENTRY_DSN" in os.environ:
    import sentry_sdk

    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        traces_sample_rate=0.2,
        _experiments={
            "continuous_profiling_auto_start": True,
        },
    )

app = FastAPI(
    title="Analytics - Deadlock API",
    description="""
Part of the [https://deadlock-api.com](https://deadlock-api.com) project.

API for Deadlock analytics, including match, player, hero and item statistics.

_deadlock-api.com is not endorsed by Valve and does not reflect the views or opinions of Valve or anyone officially involved in producing or managing Valve properties. Valve and all associated properties are trademarks or registered trademarks of Valve Corporation_
""",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

instrumentator = Instrumentator(should_group_status_codes=False).instrument(app)


@app.on_event("startup")
async def _startup():
    instrumentator.expose(app, include_in_schema=False)


app.include_router(v2.router)
app.include_router(v1.router)
app.include_router(v1.no_tagged_router)
app.include_router(internal.router)
app.include_router(internal.no_key_router)


@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse("/docs")


@app.get("/health", include_in_schema=False)
def get_health():
    return {"status": "ok"}


@app.head("/health", include_in_schema=False)
def head_health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    return FileResponse("favicon.ico")


if __name__ == "__main__":
    import granian
    from granian.constants import Interfaces
    from granian.log import LogLevels

    granian.Granian(
        "deadlock_analytics_api.main:app",
        address="0.0.0.0",
        port=8080,
        interface=Interfaces.ASGI,
        log_level=LogLevels.debug,
    ).serve()
