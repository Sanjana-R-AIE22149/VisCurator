"""
VisCurator — Auth
=================
Minimal JWT auth: one configurable user, 24-hour tokens, OAuth2 bearer dependency.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────

_SECRET = os.getenv("JWT_SECRET", "viscurator-dev-secret-change-in-prod")
_ALGORITHM = "HS256"
_EXPIRE_HOURS = 24

_USERNAME = os.getenv("VISCURATOR_USER", "admin")
_PASSWORD = os.getenv("VISCURATOR_PASS", "viscurator")

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Single-user store: hash the password at import time so it's never stored plain.
_USERS: dict[str, dict[str, Any]] = {
    _USERNAME: {
        "username": _USERNAME,
        "hashed_password": _pwd.hash(_PASSWORD),
        "name": _USERNAME.capitalize(),
        "role": "admin",
    }
}

# ── Schemas ───────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    name: str
    role: str

# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=_EXPIRE_HOURS)
    return jwt.encode({"sub": username, "exp": expire}, _SECRET, algorithm=_ALGORITHM)


def _verify_user(username: str, password: str) -> dict[str, Any] | None:
    user = _USERS.get(username)
    if user and _pwd.verify(password, user["hashed_password"]):
        return user
    return None

# ── Dependency ────────────────────────────────────────────────────────────────

_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(token: str = Depends(_oauth2)) -> dict[str, Any]:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
        username: str = payload.get("sub", "")
    except JWTError:
        raise exc
    user = _USERS.get(username)
    if not user:
        raise exc
    return user

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    user = _verify_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(
        access_token=_create_token(user["username"]),
        username=user["username"],
        name=user["name"],
        role=user["role"],
    )
