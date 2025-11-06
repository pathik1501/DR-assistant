#!/bin/bash
# Simple deployment script for DR Assistant

set -e

echo "🚀 DR Assistant Deployment Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "❌ IMPORTANT: Edit .env file and add your OPENAI_API_KEY"
    echo "   Then run this script again"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=sk-proj-" .env; then
    echo "❌ OPENAI_API_KEY not set in .env file"
    echo "   Please edit .env and add your OpenAI API key"
    exit 1
fi

echo "✅ Environment file found"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    echo "   Install Docker from: https://www.docker.com/get-started"
    exit 1
fi

echo "✅ Docker is installed"
echo ""

# Build Docker image
echo "📦 Building Docker image..."
docker build -t dr-assistant:latest .

echo ""
echo "✅ Docker image built successfully"
echo ""

# Start services
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Deployment Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Access your application:"
echo "   Frontend: http://localhost:8501"
echo "   API:      http://localhost:8080"
echo "   API Docs: http://localhost:8080/docs"
echo ""
echo "📋 Useful commands:"
echo "   View logs:    docker-compose logs -f"
echo "   Stop:         docker-compose down"
echo "   Restart:      docker-compose restart"
echo ""

