#!/bin/bash
set -e

echo "═══════════════════════════════════════════════"
echo "  DocIntel — Document Intelligence Platform"
echo "═══════════════════════════════════════════════"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found."
    exit 1
fi

echo "✓ Docker found"

# Start services
echo ""
echo "▸ Starting all services..."
docker compose up -d --build

echo ""
echo "▸ Waiting for services to be healthy..."
sleep 10

# Wait for Elasticsearch
echo "  Waiting for Elasticsearch..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        echo "  ✓ Elasticsearch ready"
        break
    fi
    sleep 2
done

# Wait for API
echo "  Waiting for API..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "  ✓ API ready"
        break
    fi
    sleep 2
done

# Pull Ollama model
echo ""
echo "▸ Pulling AI model (tinyllama)..."
docker compose exec ollama ollama pull tinyllama
echo "  ✓ Model ready"

# Create S3 bucket
echo ""
echo "▸ Setting up S3 bucket..."
docker compose exec localstack awslocal s3 mb s3://documents 2>/dev/null || true
echo "  ✓ S3 bucket ready"

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo ""
echo "  Frontend:     http://localhost:3000"
echo "  API:          http://localhost:8000/docs"
echo "  Temporal UI:  http://localhost:8080"
echo "  Elasticsearch: http://localhost:9200"
echo "  ChromaDB:     http://localhost:8100"
echo "  Ollama:       http://localhost:11434"
echo "═══════════════════════════════════════════════"
