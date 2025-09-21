# 🚀 Deployment Guide

## Quick Start: Streamlit Cloud (Recommended)

### Prerequisites
- GitHub account
- Your repository pushed to GitHub (can be public for free tier)
- OpenAI API key

### Steps
1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select your repository: `your-username/stats-compass`
   - Main file path: `stats_compass/app.py`
   - Click "Deploy!"

3. **Set Environment Variables**:
   - In Streamlit Cloud dashboard, go to "Manage app"
   - Click "Secrets" in the sidebar
   - Add your secrets in TOML format:
     ```toml
     OPENAI_API_KEY = "sk-your-key-here"
     ```

4. **Your app will be live at**: `https://your-app-name.streamlit.app`

## Alternative: Render Deployment

### Steps
1. **Create account** at [render.com](https://render.com)
2. **Connect GitHub** repository
3. **Create Web Service**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run stats_compass/app.py --server.port=$PORT --server.address=0.0.0.0`
4. **Set Environment Variables**:
   - Add `OPENAI_API_KEY` in Render dashboard

## Docker Deployment (Local/Server)

### Local Testing
```bash
# Build the image
docker build -t stats-compass .

# Run with environment file
docker run -p 8501:8501 --env-file .env stats-compass
```

### Using Docker Compose
```bash
# Make sure you have .env file with OPENAI_API_KEY
docker-compose up -d
```

## Environment Variables Required

For any deployment, you need:
- `OPENAI_API_KEY`: Your OpenAI API key

## Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` created
- [ ] `.streamlit/config.toml` configured
- [ ] Environment variables configured
- [ ] Test the app locally first
- [ ] Deploy to chosen platform
- [ ] Test deployed app with real data
- [ ] Share with testers!

## Troubleshooting

### Common Issues
1. **Import errors**: Check `requirements.txt` has all dependencies
2. **API key not found**: Verify environment variables are set correctly
3. **App won't start**: Check logs in deployment platform dashboard
4. **Slow performance**: Consider upgrading to paid tier for more resources

### Getting Help
- Streamlit Cloud: [docs.streamlit.io](https://docs.streamlit.io/streamlit-community-cloud)
- Render: [render.com/docs](https://render.com/docs)
- Docker: Check logs with `docker logs container-name`
