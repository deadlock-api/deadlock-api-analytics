import asyncio
import os
import time
from uuid import UUID

from cachetools.func import ttl_cache
from deadlock_analytics_api.globs import postgres_conn, redis_conn
from deadlock_analytics_api.rate_limiter.models import (
    InvalidAPIKey,
    RateLimit,
    RateLimitStatus,
)
from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

MAX_TTL_SECONDS = 60 * 60  # 1 hour
ENFORCE_RATE_LIMITS: bool = bool(os.environ.get("ENFORCE_RATE_LIMITS", False))

WHITELISTED_ROUTES = [
    "/",
    "/docs",
    "/openapi.json",
    "/health",
    "/robots.txt",
    "/metrics",
]
RATE_LIMITS = {
    "ip": [
        RateLimit(limit=20, period=10),
        RateLimit(limit=1, period=60, path="/v1/matches"),
        RateLimit(limit=10, period=60 * 60, path="/v1/matches"),
        RateLimit(limit=20, period=60, path="/v1/matches/search"),
        RateLimit(limit=300, period=60 * 60, path="/v1/matches/search"),
    ],
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path in WHITELISTED_ROUTES:
            return await call_next(request)

        limit_results = await RateLimitMiddleware.get_limit_results(request)
        try:
            for status in limit_results:
                print(
                    f"Checking {status.key}: {status.count}/{status.limit} in {status.period}s"
                )
                status.raise_for_limit()
        except HTTPException as e:
            ip = request.headers.get("CF-Connecting-IP", request.client.host)
            print(f"Rate limit exceeded {e.headers} by IP {ip}")
            if ENFORCE_RATE_LIMITS:
                return JSONResponse(
                    content=e.detail, status_code=e.status_code, headers=e.headers
                )
        except ValueError as e:
            print(f"ValueError: {e}")

        response = await call_next(request)
        status = sorted(limit_results, key=lambda x: x.remaining)[0]
        response.headers.update(status.headers)
        return response

    @staticmethod
    async def get_limit_results(request) -> list[RateLimitStatus]:
        api_key: str = request.headers.get(
            "X-API-Key", request.query_params.get("api_key")
        )
        if api_key is not None:
            try:
                api_key = api_key.lstrip("HEXE-")
                api_key: UUID = UUID(api_key)
                return (
                    [
                        await RateLimitMiddleware.limit_by_key(
                            f"{api_key}:{limit.path or 'default'}", limit
                        )
                        for limit in RateLimitMiddleware.get_limits_by_api_key(api_key)
                    ]
                    + [  # path param matching isn't implemented yet
                        i
                        for i in (
                            [
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:timestamps",
                                    RateLimit(limit=10, period=60),
                                ),
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:timestamps",
                                    RateLimit(limit=100, period=60 * 60),
                                ),
                            ]
                            if request.url.path.endswith("timestamps")
                            else []
                        )
                    ]
                    + [  # path param matching isn't implemented yet
                        i
                        for i in (
                            [
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:/matches/by-account-id",
                                    RateLimit(limit=10, period=60),
                                ),
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:/matches/by-account-id",
                                    RateLimit(limit=100, period=60 * 60),
                                ),
                            ]
                            if request.url.path.startswith("/matches/by-account-id")
                            else []
                        )
                    ]
                    + [  # path param matching isn't implemented yet
                        i
                        for i in (
                            [
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:/v1/matches",
                                    RateLimit(limit=1, period=60),
                                ),
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:/v1/matches",
                                    RateLimit(limit=10, period=60 * 60),
                                ),
                            ]
                            if request.url.path == "/v1/matches"
                            else []
                        )
                    ]
                    + [
                        i
                        for i in (
                            [
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:metadata",
                                    RateLimit(limit=10, period=60),
                                ),
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:metadata",
                                    RateLimit(limit=100, period=60 * 60),
                                ),
                            ]
                            if request.url.path.endswith("metadata")
                            else []
                        )
                    ]
                    + [  # path param matching isn't implemented yet
                        i
                        for i in (
                            [
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:short",
                                    RateLimit(limit=100, period=60),
                                ),
                                await RateLimitMiddleware.limit_by_key(
                                    f"{api_key}:short",
                                    RateLimit(limit=1000, period=60 * 60),
                                ),
                            ]
                            if request.url.path.endswith("short")
                            else []
                        )
                    ]
                )
            except InvalidAPIKey:
                print(f"Invalid API key: {api_key}, falling back to IP rate limits")
            except ValueError as e:
                print(e)
                print(f"Invalid API key: {api_key}, falling back to IP rate limits")
        ip = request.headers.get("CF-Connecting-IP", request.client.host)
        return (
            [
                await RateLimitMiddleware.limit_by_key(
                    f"{ip}:{limit.path or 'default'}", limit
                )
                for limit in RATE_LIMITS.get("ip", {})
                if limit.path is None or request.url.path == limit.path
            ]
            + [  # path param matching isn't implemented yet
                i
                for i in (
                    [
                        await RateLimitMiddleware.limit_by_key(
                            f"{ip}:timestamps", RateLimit(limit=10, period=60)
                        ),
                        await RateLimitMiddleware.limit_by_key(
                            f"{ip}:timestamps",
                            RateLimit(limit=100, period=60 * 60),
                        ),
                    ]
                    if "by-account-id" in request.url.path
                    else []
                )
            ]
            + [  # path param matching isn't implemented yet
                i
                for i in (
                    [
                        await RateLimitMiddleware.limit_by_key(
                            f"{ip}:/matches/by-account-id",
                            RateLimit(limit=10, period=60),
                        ),
                        await RateLimitMiddleware.limit_by_key(
                            f"{ip}:/matches/by-account-id",
                            RateLimit(limit=100, period=60 * 60),
                        ),
                    ]
                    if request.url.path.startswith("/matches/by-account-id")
                    else []
                )
            ]
            + [  # path param matching isn't implemented yet
                i
                for i in (
                    [
                        await RateLimitMiddleware.limit_by_key(
                            f"{ip}:short",
                            RateLimit(limit=100, period=60),
                        ),
                        await RateLimitMiddleware.limit_by_key(
                            f"{ip}:short",
                            RateLimit(limit=1000, period=60 * 60),
                        ),
                    ]
                    if request.url.path.endswith("short")
                    else []
                )
            ]
        )

    @staticmethod
    @ttl_cache(ttl=60)
    def get_limits_by_api_key(key: UUID) -> list[RateLimit]:
        limits = []
        with postgres_conn().cursor() as cursor:
            cursor.execute(
                "SELECT rate_global_limit, rate_global_period FROM api_keys WHERE key = %s",
                (str(key),),
            )
            res = cursor.fetchone()
            if res is None:
                print(res)
                raise InvalidAPIKey
            limits.append(RateLimit(limit=res[0], period=res[1].seconds))
            cursor.execute(
                "SELECT rate_limit, rate_period, path FROM api_key_limits WHERE key = %s",
                (str(key),),
            )
            for res in cursor.fetchall():
                limits.append(
                    RateLimit(limit=res[0], period=res[1].seconds, path=res[2])
                )
        return limits

    @staticmethod
    async def limit_by_key(key: str, rate_limit: RateLimit) -> RateLimitStatus:
        current_time = float(time.time())
        pipe = redis_conn().pipeline()
        pipe.zremrangebyscore(key, 0, current_time - MAX_TTL_SECONDS)
        pipe.zadd(key, {str(current_time): current_time})
        pipe.zrange(key, current_time - rate_limit.period, current_time, byscore=True)
        pipe.expire(key, MAX_TTL_SECONDS)
        result = pipe.execute()
        times = list(map(float, result[2]))
        filtered_times = sorted(
            [t for t in times if t > current_time - rate_limit.period]
        )
        assert len(times) == len(filtered_times)
        current_count = len(filtered_times)
        # if current_count > rate_limit.limit:
        #     REDIS.zrem(key, current_time)
        return RateLimitStatus(
            key=key,
            count=current_count,
            limit=rate_limit.limit,
            period=rate_limit.period,
            oldest_request_time=filtered_times[0] if filtered_times else 0,
        )


async def test_rate_limiter():
    while True:
        status = await RateLimitMiddleware.limit_by_key(
            "test", RateLimit(limit=20, period=10)
        )
        assert status.is_limited is False
        print(
            f"count: {status.count}, "
            f"limit: {status.limit}, "
            f"period: {status.period}, "
            f"remaining: {status.remaining}, "
            f"next_request: {status.next_request_in}"
        )
        await asyncio.sleep(status.next_request_in)


if __name__ == "__main__":
    asyncio.run(test_rate_limiter())