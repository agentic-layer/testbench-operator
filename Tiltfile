# -*- mode: Python -*-

# Increase Kubernetes upsert timeout for CRD installations
update_settings(max_parallel_updates=10)

# Load .env file for environment variables
load('ext://dotenv', 'dotenv')
dotenv()

v1alpha1.extension_repo(name='agentic-layer', url='https://github.com/agentic-layer/tilt-extensions', ref='v0.3.1')

v1alpha1.extension(name='cert-manager', repo_name='agentic-layer', repo_path='cert-manager')
load('ext://cert-manager', 'cert_manager_install')
cert_manager_install()

v1alpha1.extension(name='agent-runtime', repo_name='agentic-layer', repo_path='agent-runtime')
load('ext://agent-runtime', 'agent_runtime_install')
agent_runtime_install(version='0.9.0')

v1alpha1.extension(name='ai-gateway-litellm', repo_name='agentic-layer', repo_path='ai-gateway-litellm')
load('ext://ai-gateway-litellm', 'ai_gateway_litellm_install')
ai_gateway_litellm_install(version='0.2.0')

# Apply Kubernetes manifests
k8s_yaml(kustomize('deploy/local'))

k8s_resource('ai-gateway-litellm', port_forwards=['11001:4000'])
k8s_resource('weather-agent', port_forwards='11010:8000', labels=['agents'], resource_deps=['agent-runtime'])
k8s_resource('lgtm', port_forwards=['11000:3000', '9090:9090', '4318:4318'])
