#!/usr/bin/env python3
"""
Database Token Authentication and Authorization utilities for Revival Medical System
Uses tokens stored in the database instead of JWT
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from dal.database import DatabaseManager
from dal.models.users import Users
from dal.models.role import Role

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

class UserContext(BaseModel):
    """User context for authorization"""
    user_id: int
    role_id: int
    role_name: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    token: str
    can_access_all_patients: bool = False

def get_user_by_token(token: str, user_id: Optional[int] = None) -> Optional[Users]:
    """
    Get user from database by token.

    When multiple accounts share the same device (and therefore the same push
    notification token), passing user_id via the X-User-ID header pins the
    query to the exact account the caller logged in as.

    Without user_id  -> filter by token only  (old behaviour, may match wrong user)
    With user_id     -> filter by token AND id (always matches the right user)
    """
    try:
        # Strip any leading/trailing whitespace from the token
        # The frontend may send tokens with accidental trailing spaces
        token = token.strip()

        with DatabaseManager() as db_manager:
            if not db_manager.db:
                # DB connection failed — this is a server-side problem, not a bad token.
                # Raise 503 so the frontend knows to retry, not re-login.
                logger.error("Database connection failed during token lookup")
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable. Please try again."
                )

            query = db_manager.db.query(Users).filter(
                Users.token == token,
                Users.status.in_([1, 4])   # Active users only
            )

            # If X-User-ID header was sent, add it to the WHERE clause.
            # This disambiguates multiple accounts sharing the same push token
            # (e.g. 3 test accounts on the same device).
            if user_id is not None:
                query = query.filter(Users.id == user_id)

            user = query.first()
            return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user by token: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please try again."
        )

def get_role_name(role_id: int) -> str:
    """Get role name from role ID"""
    role_mapping = {
        Role.PATIENT: "Patient",
        Role.DOCTOR: "Doctor",
        Role.HEALTH_COACH: "Health Coach",
        Role.ADMIN: "Admin",
        Role.DIAGNOSTIC: "Diagnostic",
        Role.VIDEO_UPLOADER: "Video Uploader",
        Role.TRAINER: "Trainer",
        Role.TRACKER: "Tracker",
        Role.CRM_ADMIN: "CRM Admin",
        Role.CRM_EXECUTIVE: "CRM Executive",
        Role.VENDOR: "Vendor",
        Role.ORDER_MANAGER: "Order Manager",
        Role.VIDEO_ADMIN: "Video Admin",
        Role.READ_ONLY: "Read Only"
    }
    return role_mapping.get(role_id, "Unknown")

def determine_access_level(role_id: int) -> bool:
    """Determine if role can access all patients"""
    # Roles that can access all patient data
    privileged_roles = [
        Role.DOCTOR,
        Role.HEALTH_COACH,
        Role.ADMIN,
        Role.DIAGNOSTIC,
        Role.CRM_ADMIN,
        Role.CRM_EXECUTIVE
    ]
    
    return role_id in privileged_roles

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-ID")
) -> UserContext:
    """
    Get current authenticated user with role-based access control.

    Reads Authorization: Bearer <token> as before.
    Also reads optional X-User-ID header to disambiguate accounts that share
    the same push notification token (multiple accounts on one device).

    Without X-User-ID  -> matches by token only  (may pick wrong user if shared)
    With    X-User-ID  -> matches by token AND id (always picks the right user)
    """
    try:
        # Extract token from Authorization header and strip whitespace
        token = credentials.credentials.strip()
        
        if not token:
            raise HTTPException(status_code=401, detail="Token is required")

        # Parse X-User-ID header if present
        user_id: Optional[int] = None
        if x_user_id:
            try:
                user_id = int(x_user_id.strip())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="X-User-ID header must be a valid integer user ID."
                )

        # Get user from database by token (+ optional user_id)
        # get_user_by_token raises 503 on DB failure, returns None only for truly bad tokens
        user = get_user_by_token(token, user_id)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token or user not found")

        # Determine access level
        can_access_all = determine_access_level(user.role_id)
        role_name = get_role_name(user.role_id)
        full_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
        logger.debug(f"Authenticated user_id={user.id} role={role_name}")

        return UserContext(
            user_id=user.id,
            role_id=user.role_id,
            role_name=role_name,
            email=user.email,
            full_name=full_name,
            token=token,
            can_access_all_patients=can_access_all
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user: {e}")
        raise HTTPException(status_code=500, detail="Authentication error")

def require_patient_access(requested_patient_id: Optional[int], current_user: UserContext) -> bool:
    """Check if current user can access requested patient data"""
    try:
        # If user can access all patients, allow access
        if current_user.can_access_all_patients:
            return True
        
        # If user is a patient, they can only access their own data
        if current_user.role_id == Role.PATIENT:
            if requested_patient_id is None:
                # If no specific patient requested, default to current user
                return True
            elif requested_patient_id == current_user.user_id:
                # Patient accessing their own data
                return True
            else:
                # Patient trying to access another patient's data
                raise HTTPException(
                    status_code=403, 
                    detail="Patients can only access their own medical data"
                )
        
        # Default deny for any other scenario
        raise HTTPException(status_code=403, detail="Access denied")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in require_patient_access: {e}")
        raise HTTPException(status_code=500, detail="Authorization error")

def get_authorized_patient_id(requested_patient_id: Optional[int], current_user: UserContext) -> int:
    """Get the authorized patient ID based on user role and request"""
    try:
        # If user can access all patients and a specific patient is requested
        if current_user.can_access_all_patients and requested_patient_id:
            return requested_patient_id
        
        # If user can access all patients but no specific patient requested, return None to indicate "all patients"
        elif current_user.can_access_all_patients and not requested_patient_id:
            return None
        
        # If user is a patient, always return their own ID
        elif current_user.role_id == Role.PATIENT:
            return current_user.user_id
        
        # Default fallback
        else:
            raise HTTPException(status_code=403, detail="Cannot determine authorized patient access")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_authorized_patient_id: {e}")
        raise HTTPException(status_code=500, detail="Authorization error")

# Role-based decorators
def require_roles(*allowed_roles):
    """Decorator to require specific roles"""
    def decorator(func):
        async def wrapper(*args, current_user: UserContext = Depends(get_current_user), **kwargs):
            if current_user.role_id not in allowed_roles:
                role_names = [str(role) for role in allowed_roles]
                raise HTTPException(
                    status_code=403, 
                    detail=f"Access denied. Required roles: {', '.join(role_names)}"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

def require_admin(current_user: UserContext = Depends(get_current_user)) -> UserContext:
    """Require admin role"""
    if current_user.role_id != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

def require_medical_staff(current_user: UserContext = Depends(get_current_user)) -> UserContext:
    """Require medical staff roles (Doctor, Health Coach, Diagnostic)"""
    medical_roles = [Role.DOCTOR, Role.HEALTH_COACH, Role.DIAGNOSTIC]
    if current_user.role_id not in medical_roles:
        raise HTTPException(status_code=403, detail="Medical staff access required")
    return current_user