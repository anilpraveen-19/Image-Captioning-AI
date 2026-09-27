# 🖼️ AI Image Captioning

An AI-powered image captioning web application that automatically generates a natural-language description for an uploaded image.

The project uses a **CNN + LSTM architecture**, where a pretrained ResNet-50 extracts visual features from an image and an LSTM decoder generates the corresponding caption.

## 🚀 Live Demo
https://image-captioning-ai-delta.vercel.app/


> Note: The frontend is deployed on Vercel. The backend inference service is currently running through a temporary development tunnel, so the backend must be active for caption generation to work.

---

## ✨ Features

- 📷 Upload images from your device
- 🤖 AI-generated image captions
- 🧠 CNN + LSTM based image captioning model
- ⚡ FastAPI backend
- ⚛️ React frontend
- 📋 Copy generated captions
- 🌐 Responsive web interface
- 🔌 REST API for caption generation

---

## 🏗️ Architecture

```text
User
 │
 ▼
React Frontend
 │
 │ Image Upload
 ▼
FastAPI Backend
 │
 ▼
ResNet-50 Encoder
 │
 │ Image Features
 ▼
LSTM Decoder
 │
 ▼
Generated Caption
