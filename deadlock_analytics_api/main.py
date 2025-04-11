import logging.config
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import FileResponse, RedirectResponse

from deadlock_analytics_api.logging_middleware import RouterLoggingMiddleware
from deadlock_analytics_api.routers import v1, v2

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))
logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(process)s %(levelname)s %(name)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "level": os.environ.get("LOG_LEVEL", "DEBUG"),
                "class": "logging.StreamHandler",
                "stream": sys.stderr,
            }
        },
        "root": {"level": "DEBUG", "handlers": ["console"], "propagate": True},
    }
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("clickhouse_driver").setLevel(logging.WARNING)

LOGGER = logging.getLogger(__name__)

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
app.add_middleware(RouterLoggingMiddleware, logger=LOGGER)

instrumentator = Instrumentator(should_group_status_codes=False).instrument(app)


@app.on_event("startup")
async def _startup():
    instrumentator.expose(app, include_in_schema=False)


app.include_router(v2.router)
app.include_router(v1.router)
app.include_router(v1.no_tagged_router)


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
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
