#!/usr/bin/env python3
"""
MCP Security Utilities
Provides authentication, rate limiting, and security features
"""

import time
import hashlib
import hmac
import jwt
import redis
from functools import wraps
from flask import request, jsonify, g
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

# Security configuration
RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 100))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 3600))    # 1 hour window
JWT_SECRET = os.getenv('JWT_SECRET')
if not JWT_SECRET:
    print("⚠️ WARNING: JWT_SECRET environment variable not set! Set it in your .env file.")
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', 24))

class SecurityManager:
    """Manages security features for MCP server"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.blocked_ips = set()
        self.suspicious_patterns = [
            'script', 'javascript', 'eval', 'exec', 'import',
            'subprocess', 'os.system', '__import__', 'open(',
            'file(', 'input(', 'raw_input('
        ]
    
    def generate_api_key(self, user_id, permissions=None):
        """Generate secure API key for user"""
        if permissions is None:
            permissions = ['read', 'write']
        
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'issued_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)).isoformat()
        }
        
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return token
    
    def validate_api_key(self, token):
        """Validate API key and return user info"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # Check expiration
            expires_at = datetime.fromisoformat(payload['expires_at'])
            if datetime.utcnow() > expires_at:
                return None, "Token expired"
            
            return payload, None
        except jwt.ExpiredSignatureError:
            return None, "Token expired"
        except jwt.InvalidTokenError as e:
            return None, f"Invalid token: {str(e)}"
    
    def check_rate_limit(self, identifier, limit=None, window=None):
        """Check if request is within rate limits"""
        if not self.redis_client:
            return True  # Allow if Redis not available
        
        limit = limit or RATE_LIMIT_REQUESTS
        window = window or RATE_LIMIT_WINDOW
        
        try:
            key = f"rate_limit:{identifier}"
            current_time = int(time.time())
            window_start = current_time - window
            
            # Remove old entries
            self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            current_requests = self.redis_client.zcard(key)
            
            if current_requests >= limit:
                return False
            
            # Add current request
            self.redis_client.zadd(key, {str(current_time): current_time})
            self.redis_client.expire(key, window)
            
            return True
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error
    
    def validate_input_security(self, data):
        """Check input for security threats"""
        if isinstance(data, dict):
            data_str = str(data)
        elif isinstance(data, str):
            data_str = data
        else:
            data_str = str(data)
        
        data_lower = data_str.lower()
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern in data_lower:
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return False, f"Suspicious content detected: {pattern}"
        
        # Check for SQL injection patterns
        sql_patterns = ['union select', 'drop table', 'delete from', 'insert into', '--', ';--']
        for pattern in sql_patterns:
            if pattern in data_lower:
                logger.warning(f"SQL injection pattern detected: {pattern}")
                return False, f"SQL injection attempt detected"
        
        # Check for XSS patterns
        xss_patterns = ['<script', 'javascript:', 'onload=', 'onerror=', 'onclick=']
        for pattern in xss_patterns:
            if pattern in data_lower:
                logger.warning(f"XSS pattern detected: {pattern}")
                return False, f"XSS attempt detected"
        
        return True, None
    
    def get_client_identifier(self, request):
        """Get unique identifier for client (IP + User-Agent)"""
        ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        user_agent = request.headers.get('User-Agent', '')
        return hashlib.sha256(f"{ip}:{user_agent}".encode()).hexdigest()[:16]
    
    def is_ip_blocked(self, ip):
        """Check if IP is blocked"""
        return ip in self.blocked_ips
    
    def block_ip(self, ip, duration=3600):
        """Block IP address"""
        self.blocked_ips.add(ip)
        if self.redis_client:
            try:
                self.redis_client.setex(f"blocked_ip:{ip}", duration, "1")
            except Exception as e:
                logger.error(f"Error blocking IP in Redis: {e}")
        logger.warning(f"IP blocked: {ip}")
    
    def log_security_event(self, event_type, details, severity="medium"):
        """Log security events"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'details': details,
            'severity': severity,
            'ip': request.remote_addr if request else 'unknown'
        }
        
        logger.warning(f"Security Event [{severity.upper()}]: {event_type} - {details}")
        
        # Store in Redis for monitoring
        if self.redis_client:
            try:
                key = f"security_events:{datetime.utcnow().strftime('%Y-%m-%d')}"
                self.redis_client.lpush(key, str(event))
                self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
            except Exception as e:
                logger.error(f"Error storing security event: {e}")

def create_security_decorators(security_manager):
    """Create security decorators with security manager instance"""
    
    def require_auth(f):
        """Decorator to require authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check for API key in header or query param
            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            
            if not api_key:
                security_manager.log_security_event("missing_api_key", "No API key provided")
                return jsonify({'error': 'API key required'}), 401
            
            # Validate API key
            user_info, error = security_manager.validate_api_key(api_key)
            if error:
                security_manager.log_security_event("invalid_api_key", f"Invalid API key: {error}")
                return jsonify({'error': f'Invalid API key: {error}'}), 401
            
            # Store user info in request context
            g.user_info = user_info
            return f(*args, **kwargs)
        
        return decorated_function
    
    def rate_limit(limit=None, window=None):
        """Decorator to apply rate limiting"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                client_id = security_manager.get_client_identifier(request)
                
                if not security_manager.check_rate_limit(client_id, limit, window):
                    security_manager.log_security_event(
                        "rate_limit_exceeded", 
                        f"Client {client_id} exceeded rate limit",
                        "high"
                    )
                    return jsonify({'error': 'Rate limit exceeded'}), 429
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def validate_input(f):
        """Decorator to validate input for security threats"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check JSON data if present
            if request.is_json:
                data = request.get_json()
                is_safe, error = security_manager.validate_input_security(data)
                if not is_safe:
                    security_manager.log_security_event(
                        "malicious_input", 
                        f"Malicious input detected: {error}",
                        "high"
                    )
                    return jsonify({'error': 'Invalid input detected'}), 400
            
            # Check query parameters
            for key, value in request.args.items():
                is_safe, error = security_manager.validate_input_security(value)
                if not is_safe:
                    security_manager.log_security_event(
                        "malicious_input", 
                        f"Malicious query param: {key}={value}",
                        "high"
                    )
                    return jsonify({'error': 'Invalid input detected'}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    
    def block_suspicious_ips(f):
        """Decorator to block suspicious IP addresses"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            
            if security_manager.is_ip_blocked(client_ip):
                security_manager.log_security_event(
                    "blocked_ip_access", 
                    f"Blocked IP attempted access: {client_ip}",
                    "high"
                )
                return jsonify({'error': 'Access denied'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    
    return require_auth, rate_limit, validate_input, block_suspicious_ips

def generate_secure_token(length=32):
    """Generate cryptographically secure random token"""
    import secrets
    return secrets.token_urlsafe(length)

def hash_password(password, salt=None):
    """Hash password with salt"""
    if salt is None:
        salt = os.urandom(32)
    
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt + pwdhash

def verify_password(stored_password, provided_password):
    """Verify password against stored hash"""
    salt = stored_password[:32]
    stored_hash = stored_password[32:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    return pwdhash == stored_hash

def create_csrf_token():
    """Create CSRF token"""
    return generate_secure_token(16)

def verify_csrf_token(token, stored_token):
    """Verify CSRF token"""
    return hmac.compare_digest(token, stored_token)
