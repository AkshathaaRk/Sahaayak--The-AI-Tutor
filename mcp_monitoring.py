#!/usr/bin/env python3
"""
MCP Monitoring Utilities
Provides metrics collection, monitoring, and dashboard functionality
"""

from flask import Flask, jsonify, render_template, request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import json
import redis
from datetime import datetime, timedelta
import logging
import psutil
import os

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('mcp_requests_total', 'Total MCP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('mcp_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_SESSIONS = Gauge('mcp_active_sessions', 'Number of active sessions')
CONTEXT_SIZE = Histogram('mcp_context_size_bytes', 'Context size in bytes')
REDIS_OPERATIONS = Counter('mcp_redis_operations_total', 'Redis operations', ['operation', 'status'])
ERROR_COUNT = Counter('mcp_errors_total', 'Total errors', ['error_type'])

class MCPMonitor:
    """Monitoring and metrics collection for MCP server"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.start_time = time.time()
        self.metrics = {
            'requests_total': 0,
            'context_updates': 0,
            'context_fetches': 0,
            'context_clears': 0,
            'errors_total': 0,
            'active_sessions': 0,
            'redis_operations': 0,
            'avg_response_time': 0,
            'uptime_seconds': 0
        }
        
    def record_request(self, method, endpoint, status_code, duration):
        """Record request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
        self.metrics['requests_total'] += 1
        
        # Update average response time
        current_avg = self.metrics['avg_response_time']
        total_requests = self.metrics['requests_total']
        self.metrics['avg_response_time'] = ((current_avg * (total_requests - 1)) + duration) / total_requests
    
    def record_context_operation(self, operation, size_bytes=None):
        """Record context operation metrics"""
        if operation == 'update':
            self.metrics['context_updates'] += 1
            if size_bytes:
                CONTEXT_SIZE.observe(size_bytes)
        elif operation == 'fetch':
            self.metrics['context_fetches'] += 1
        elif operation == 'clear':
            self.metrics['context_clears'] += 1
    
    def record_redis_operation(self, operation, success=True):
        """Record Redis operation metrics"""
        status = 'success' if success else 'error'
        REDIS_OPERATIONS.labels(operation=operation, status=status).inc()
        self.metrics['redis_operations'] += 1
    
    def record_error(self, error_type):
        """Record error metrics"""
        ERROR_COUNT.labels(error_type=error_type).inc()
        self.metrics['errors_total'] += 1
    
    def update_active_sessions(self, count):
        """Update active sessions count"""
        ACTIVE_SESSIONS.set(count)
        self.metrics['active_sessions'] = count
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used // (1024 * 1024),
                'memory_total_mb': memory.total // (1024 * 1024),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used // (1024 * 1024 * 1024),
                'disk_total_gb': disk.total // (1024 * 1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def get_redis_metrics(self):
        """Get Redis performance metrics"""
        if not self.redis_client:
            return {'status': 'not_connected'}
        
        try:
            info = self.redis_client.info()
            return {
                'status': 'connected',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_mb': info.get('used_memory', 0) // (1024 * 1024),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'uptime_seconds': info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis metrics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_context_statistics(self):
        """Get context storage statistics"""
        if not self.redis_client:
            return {}
        
        try:
            # Get all context keys
            context_keys = self.redis_client.keys('mcp:context:*')
            
            total_contexts = len(context_keys)
            total_size = 0
            users = set()
            sessions = set()
            
            for key in context_keys[:100]:  # Limit to avoid performance issues
                try:
                    data = self.redis_client.get(key)
                    if data:
                        context_data = json.loads(data)
                        total_size += context_data.get('size_bytes', 0)
                        users.add(context_data.get('user_id'))
                        sessions.add(context_data.get('session_id'))
                except:
                    continue
            
            return {
                'total_contexts': total_contexts,
                'total_size_mb': total_size // (1024 * 1024),
                'unique_users': len(users),
                'unique_sessions': len(sessions),
                'avg_context_size_kb': (total_size // len(context_keys)) // 1024 if context_keys else 0
            }
        except Exception as e:
            logger.error(f"Error getting context statistics: {e}")
            return {}
    
    def get_security_events(self, hours=24):
        """Get recent security events"""
        if not self.redis_client:
            return []
        
        try:
            events = []
            for i in range(hours):
                date = (datetime.utcnow() - timedelta(hours=i)).strftime('%Y-%m-%d')
                key = f"security_events:{date}"
                day_events = self.redis_client.lrange(key, 0, -1)
                events.extend([eval(event) for event in day_events])
            
            # Sort by timestamp
            events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return events[:50]  # Return last 50 events
        except Exception as e:
            logger.error(f"Error getting security events: {e}")
            return []
    
    def get_comprehensive_status(self):
        """Get comprehensive system status"""
        self.metrics['uptime_seconds'] = int(time.time() - self.start_time)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': self.metrics['uptime_seconds'],
            'metrics': self.metrics,
            'system': self.get_system_metrics(),
            'redis': self.get_redis_metrics(),
            'context_stats': self.get_context_statistics(),
            'security_events_count': len(self.get_security_events(1))  # Last hour
        }

def create_monitoring_routes(app, monitor, redis_client):
    """Create monitoring routes for Flask app"""
    
    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint"""
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    
    @app.route('/admin/dashboard')
    def admin_dashboard():
        """Admin dashboard"""
        try:
            status = monitor.get_comprehensive_status()
            security_events = monitor.get_security_events(24)
            
            return render_template('mcp_dashboard.html', 
                                 status=status, 
                                 security_events=security_events)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return jsonify({'error': 'Dashboard unavailable'}), 500
    
    @app.route('/admin/status')
    def admin_status():
        """JSON status endpoint"""
        return jsonify(monitor.get_comprehensive_status())
    
    @app.route('/admin/security-events')
    def security_events():
        """Security events endpoint"""
        hours = request.args.get('hours', 24, type=int)
        events = monitor.get_security_events(hours)
        return jsonify({'events': events, 'count': len(events)})
    
    @app.route('/admin/clear-cache')
    def clear_cache():
        """Clear Redis cache (admin only)"""
        if not redis_client:
            return jsonify({'error': 'Redis not available'}), 503
        
        try:
            # Clear only MCP-related keys
            keys = redis_client.keys('mcp:*')
            if keys:
                redis_client.delete(*keys)
            
            return jsonify({
                'success': True,
                'message': f'Cleared {len(keys)} cache entries'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def create_request_middleware(monitor):
    """Create middleware to track requests"""
    
    def before_request():
        """Before request handler"""
        request.start_time = time.time()
    
    def after_request(response):
        """After request handler"""
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            monitor.record_request(
                method=request.method,
                endpoint=request.endpoint or 'unknown',
                status_code=response.status_code,
                duration=duration
            )
        return response
    
    return before_request, after_request

class HealthChecker:
    """Health checking utilities"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.checks = {}
    
    def check_redis(self):
        """Check Redis connectivity"""
        try:
            if not self.redis_client:
                return False, "Redis client not configured"
            
            self.redis_client.ping()
            return True, "Redis is healthy"
        except Exception as e:
            return False, f"Redis error: {str(e)}"
    
    def check_disk_space(self, threshold=90):
        """Check disk space"""
        try:
            disk = psutil.disk_usage('/')
            if disk.percent > threshold:
                return False, f"Disk usage high: {disk.percent}%"
            return True, f"Disk usage normal: {disk.percent}%"
        except Exception as e:
            return False, f"Disk check error: {str(e)}"
    
    def check_memory(self, threshold=90):
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > threshold:
                return False, f"Memory usage high: {memory.percent}%"
            return True, f"Memory usage normal: {memory.percent}%"
        except Exception as e:
            return False, f"Memory check error: {str(e)}"
    
    def run_all_checks(self):
        """Run all health checks"""
        checks = {
            'redis': self.check_redis(),
            'disk': self.check_disk_space(),
            'memory': self.check_memory()
        }
        
        overall_healthy = all(check[0] for check in checks.values())
        
        return {
            'healthy': overall_healthy,
            'checks': {name: {'healthy': check[0], 'message': check[1]} 
                      for name, check in checks.items()},
            'timestamp': datetime.utcnow().isoformat()
        }
