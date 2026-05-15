#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server
Manages conversation context across multiple AI providers
"""

from flask import Flask, request, jsonify, render_template
import redis
import json
import time
import hashlib
import uuid
from datetime import datetime, timedelta
import logging
from functools import wraps
import os
from cryptography.fernet import Fernet
import jwt

# Initialize Flask app for MCP server
mcp_app = Flask(__name__, template_folder='templates')
mcp_app.config['SECRET_KEY'] = os.getenv('MCP_SECRET_KEY', 'mcp-secret-key-change-in-production')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

# MCP Configuration
MCP_API_KEY = os.getenv('MCP_API_KEY', 'mcp-api-key-12345')
CONTEXT_TTL = int(os.getenv('CONTEXT_TTL', 3600))  # 1 hour default
MAX_CONTEXT_SIZE = int(os.getenv('MAX_CONTEXT_SIZE', 10000))  # 10KB default

# Encryption key for sensitive data
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
cipher_suite = Fernet(ENCRYPTION_KEY)

# Initialize Redis connection
try:
    from setup_redis import get_redis_client
    redis_client = get_redis_client()
    logger.info("✅ Redis connection established")
except Exception as e:
    logger.error(f"❌ Redis connection failed: {e}")
    redis_client = None

# Metrics storage
metrics = {
    'requests_total': 0,
    'context_updates': 0,
    'context_fetches': 0,
    'errors_total': 0,
    'active_sessions': 0
}

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not api_key or api_key != MCP_API_KEY:
            metrics['errors_total'] += 1
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def validate_input(data, required_fields):
    """Validate input data"""
    if not isinstance(data, dict):
        return False, "Invalid JSON data"
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Missing required field: {field}"
    
    return True, None

def encrypt_sensitive_data(data):
    """Encrypt sensitive context data"""
    try:
        json_data = json.dumps(data)
        encrypted_data = cipher_suite.encrypt(json_data.encode())
        return encrypted_data.decode()
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        return None

def decrypt_sensitive_data(encrypted_data):
    """Decrypt sensitive context data"""
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data.encode())
        return json.loads(decrypted_data.decode())
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        return None

def generate_context_key(user_id, session_id):
    """Generate Redis key for context storage"""
    return f"mcp:context:{user_id}:{session_id}"

def generate_session_id():
    """Generate unique session ID"""
    return str(uuid.uuid4())

@mcp_app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    metrics['requests_total'] += 1
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'redis_connected': redis_client is not None,
        'metrics': metrics
    }
    
    if redis_client:
        try:
            redis_client.ping()
            health_status['redis_status'] = 'connected'
        except:
            health_status['redis_status'] = 'disconnected'
            health_status['status'] = 'degraded'
    else:
        health_status['redis_status'] = 'not_configured'
        health_status['status'] = 'degraded'
    
    return jsonify(health_status)

@mcp_app.route('/context/update', methods=['POST'])
@require_api_key
def update_context():
    """Update conversation context"""
    metrics['requests_total'] += 1
    metrics['context_updates'] += 1
    
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['user_id', 'session_id', 'context']
        is_valid, error_msg = validate_input(data, required_fields)
        if not is_valid:
            metrics['errors_total'] += 1
            return jsonify({'error': error_msg}), 400
        
        user_id = data['user_id']
        session_id = data['session_id']
        context = data['context']
        
        # Validate context size
        context_size = len(json.dumps(context))
        if context_size > MAX_CONTEXT_SIZE:
            metrics['errors_total'] += 1
            return jsonify({'error': f'Context too large: {context_size} bytes (max: {MAX_CONTEXT_SIZE})'}), 400
        
        if not redis_client:
            metrics['errors_total'] += 1
            return jsonify({'error': 'Redis not available'}), 503
        
        # Prepare context data
        context_data = {
            'context': context,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'session_id': session_id,
            'size_bytes': context_size
        }
        
        # Encrypt sensitive data if needed
        if data.get('encrypt', False):
            encrypted_context = encrypt_sensitive_data(context)
            if encrypted_context:
                context_data['context'] = encrypted_context
                context_data['encrypted'] = True
        
        # Store in Redis
        context_key = generate_context_key(user_id, session_id)
        redis_client.setex(
            context_key,
            CONTEXT_TTL,
            json.dumps(context_data)
        )
        
        logger.info(f"Context updated for user {user_id}, session {session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Context updated successfully',
            'context_key': context_key,
            'ttl': CONTEXT_TTL,
            'size_bytes': context_size
        })
        
    except Exception as e:
        metrics['errors_total'] += 1
        logger.error(f"Error updating context: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@mcp_app.route('/context/fetch', methods=['GET'])
@require_api_key
def fetch_context():
    """Fetch conversation context"""
    metrics['requests_total'] += 1
    metrics['context_fetches'] += 1
    
    try:
        user_id = request.args.get('user_id')
        session_id = request.args.get('session_id')
        
        if not user_id or not session_id:
            metrics['errors_total'] += 1
            return jsonify({'error': 'Missing user_id or session_id'}), 400
        
        if not redis_client:
            metrics['errors_total'] += 1
            return jsonify({'error': 'Redis not available'}), 503
        
        # Fetch from Redis
        context_key = generate_context_key(user_id, session_id)
        context_data = redis_client.get(context_key)
        
        if not context_data:
            return jsonify({
                'success': True,
                'context': None,
                'message': 'No context found'
            })
        
        # Parse context data
        parsed_data = json.loads(context_data)
        
        # Decrypt if needed
        if parsed_data.get('encrypted', False):
            decrypted_context = decrypt_sensitive_data(parsed_data['context'])
            if decrypted_context:
                parsed_data['context'] = decrypted_context
            else:
                metrics['errors_total'] += 1
                return jsonify({'error': 'Failed to decrypt context'}), 500
        
        logger.info(f"Context fetched for user {user_id}, session {session_id}")
        
        return jsonify({
            'success': True,
            'context': parsed_data['context'],
            'timestamp': parsed_data['timestamp'],
            'size_bytes': parsed_data.get('size_bytes', 0),
            'ttl_remaining': redis_client.ttl(context_key)
        })
        
    except Exception as e:
        metrics['errors_total'] += 1
        logger.error(f"Error fetching context: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@mcp_app.route('/context/clear', methods=['DELETE'])
@require_api_key
def clear_context():
    """Clear conversation context"""
    metrics['requests_total'] += 1
    
    try:
        user_id = request.args.get('user_id')
        session_id = request.args.get('session_id')
        
        if not user_id or not session_id:
            metrics['errors_total'] += 1
            return jsonify({'error': 'Missing user_id or session_id'}), 400
        
        if not redis_client:
            metrics['errors_total'] += 1
            return jsonify({'error': 'Redis not available'}), 503
        
        # Delete from Redis
        context_key = generate_context_key(user_id, session_id)
        deleted = redis_client.delete(context_key)
        
        logger.info(f"Context cleared for user {user_id}, session {session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Context cleared successfully',
            'deleted': bool(deleted)
        })
        
    except Exception as e:
        metrics['errors_total'] += 1
        logger.error(f"Error clearing context: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@mcp_app.route('/sessions/new', methods=['POST'])
@require_api_key
def create_session():
    """Create new session"""
    metrics['requests_total'] += 1
    
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id')
        
        if not user_id:
            metrics['errors_total'] += 1
            return jsonify({'error': 'Missing user_id'}), 400
        
        session_id = generate_session_id()
        metrics['active_sessions'] += 1
        
        logger.info(f"New session created: {session_id} for user {user_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        metrics['errors_total'] += 1
        logger.error(f"Error creating session: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Add monitoring routes
from mcp_monitoring import MCPMonitor, create_monitoring_routes, create_request_middleware

# Initialize monitoring
monitor = MCPMonitor(redis_client)

# Add monitoring routes
create_monitoring_routes(mcp_app, monitor, redis_client)

# Add request middleware
before_request, after_request = create_request_middleware(monitor)
mcp_app.before_request(before_request)
mcp_app.after_request(after_request)

if __name__ == '__main__':
    print("🚀 Starting MCP Server...")
    print(f"📊 Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"🔑 API Key required: {MCP_API_KEY[:8]}...")
    print(f"⏰ Context TTL: {CONTEXT_TTL} seconds")
    print(f"📈 Monitoring: http://127.0.0.1:5001/admin/dashboard")
    print(f"📊 Metrics: http://127.0.0.1:5001/metrics")
    mcp_app.run(host='127.0.0.1', port=5001, debug=True)
