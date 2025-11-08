# Start the Complete DR Assistant Frontend
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DR Assistant - Complete Frontend" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting Streamlit app with full features..." -ForegroundColor Yellow
cd "C:\Users\pathi\Documents\DR assistant"
streamlit run frontend/app_complete.py

Write-Host ""
Write-Host "[OK] Streamlit app starting. Open your browser to the URL displayed above." -ForegroundColor Green
Write-Host ""
Write-Host "Features enabled:" -ForegroundColor Cyan
Write-Host "  - Image upload" -ForegroundColor White
Write-Host "  - Classification (Grade 0-4)" -ForegroundColor White
Write-Host "  - Grad-CAM heatmap visualization" -ForegroundColor White
Write-Host "  - RAG model clinical recommendations" -ForegroundColor White
Write-Host ""
Write-Host "Make sure the API server is running on localhost:8080" -ForegroundColor Yellow
Write-Host ""



