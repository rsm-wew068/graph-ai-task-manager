# Deployment Guide for Hugging Face Spaces

## Prerequisites
1. OpenAI API key
2. Hugging Face account
3. This repository code

## Step-by-Step Deployment

### 1. Create a New Hugging Face Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Streamlit" as the SDK
4. Name your space (e.g., "automated-task-manager")
5. Set visibility as desired (Public/Private)

### 2. Upload Code Files
Upload these essential files to your Space:
- `app.py` (main Streamlit application)
- `requirements.txt` (Python dependencies)
- `Dockerfile` (container configuration)
- `utils/` directory (all utility modules)
- `pages/` directory (Streamlit pages)
- `.streamlit/config.toml` (Streamlit configuration)

### 3. Configure Environment Variables
**CRITICAL**: Set up your OpenAI API key in the Space settings:

1. Go to your Space settings
2. Click on "Variables and secrets"
3. Add a new secret:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key (starts with `sk-`)
4. Save the configuration

### 4. Restart and Test
1. Restart your Space after adding the API key
2. Monitor the logs for any errors
3. Test the environment status in the app's debug section

## Common Deployment Issues

### ✅ FIXED: Context Length Errors (Updated July 2025)
**Symptoms**: 
- "This model's maximum context length is 8192 tokens. However, your messages resulted in XXXXX tokens"
- Extraction fails on longer emails

**Solution Applied**:
- **Model Updated**: Changed from `gpt-4` (8k tokens) to `gpt-4o` (128k tokens)
- **File**: `utils/langgraph_nodes.py` line 39
- **Result**: Can now handle emails up to 128,000 tokens (16x improvement)

### ✅ FIXED: File Permission Errors (Updated July 2025)
**Symptoms**: 
- `[Errno 13] Permission denied: 'topic_graph.gpickle'`
- Graph writing fails in deployment

**Solution Applied**:
- **File Path Updated**: All graph files now write to `/tmp/topic_graph.gpickle`
- **Files Modified**: `app.py`, `utils/langgraph_nodes.py`, `utils/graphrag.py`, `pages/AI_Chatbot.py`
- **Result**: Uses Linux `/tmp` directory which is writable in Hugging Face Spaces

### API Key Not Found
**Symptoms**: 
- "Missing OPENAI_API_KEY Environment Variable" error
- Task extraction fails silently

**Solutions**:
1. Verify the API key is set in Space secrets (not variables)
2. Ensure the key name is exactly `OPENAI_API_KEY`
3. Restart the Space after adding the key
4. Check the app's debug section for environment status

### Import Errors
**Symptoms**:
- "Import error" messages on startup
- Missing utils modules

**Solutions**:
1. Ensure all files in `utils/` directory are uploaded
2. Check that `__init__.py` exists in the utils directory
3. Verify file permissions and structure

### Port Configuration Issues
**Symptoms**:
- Space shows as "Building" but never starts
- Connection timeouts

**Solutions**:
1. Ensure Dockerfile and .streamlit/config.toml use port 8501
2. Verify the CMD in Dockerfile uses correct port syntax

### Memory/Resource Limits
**Symptoms**:
- Out of memory errors
- Slow performance
- Timeouts during processing

**Solutions**:
1. Consider upgrading to a paid Hugging Face plan for more resources
2. Reduce the number of emails processed at once
3. Use smaller models if available

## Testing Your Deployment

### 1. Environment Check
- Open your deployed app
- Expand the "🔧 Deployment Diagnostics" section in the sidebar
- Verify OPENAI_API_KEY is found and shows first 10 characters
- Check that Model shows "gpt-4o" (not gpt-4)

### 2. Basic Functionality  
- Upload a small .mbox file (< 10MB for initial testing)
- Parse a few emails including longer ones (>8k tokens)
- Try task extraction on 1-2 emails
- Check for any error messages in extraction results

### 3. Full Workflow
- Upload larger email archive
- Process multiple emails (should handle long emails without context errors)
- Validate extracted tasks (no file permission errors)
- Test GraphRAG search functionality

### 4. Verification of Fixes
- **Context Length**: Try extracting from very long emails (should work with gpt-4o)
- **File Permissions**: Graph files should save to `/tmp/` without errors
- **Error Diagnostics**: Use debug mode to monitor extraction pipeline

## Monitoring and Troubleshooting

### Check Logs
1. Go to your Space page
2. Click on "Logs" tab
3. Look for error messages or warnings
4. Common issues show up in red text

### API Usage Monitoring
- Monitor your OpenAI API usage in the OpenAI dashboard
- Set up usage alerts if needed
- Consider rate limiting for high-volume deployments

### Performance Optimization
- Start with small batches of emails
- Monitor processing time and adjust limits
- Use the built-in progress indicators

## Security Notes

### Environment Variables
- NEVER commit API keys to your repository
- Use Hugging Face Space secrets for sensitive data
- Regularly rotate API keys

### File Uploads
- The app is configured to handle large files securely
- Uploaded files are processed in memory when possible
- No permanent storage of user data

## Support
If you encounter issues not covered here:
1. Check the app's debug information
2. Review Hugging Face Spaces documentation
3. Verify OpenAI API key permissions and usage limits
4. Test locally to isolate deployment-specific issues

## Recent Updates (July 2025) 🚀

### Major Reliability Improvements
The system has been updated to fix the two main deployment issues that were causing failures:

1. **Enhanced Model Performance** 
   - Switched from GPT-4 (8k tokens) to GPT-4o (128k tokens)
   - Can now handle much longer emails without context errors
   - Better performance and lower cost per token

2. **Fixed File System Issues**
   - Graph files now write to `/tmp/` directory instead of app directory
   - Resolves permission denied errors in Hugging Face Spaces
   - Maintains all functionality while being deployment-friendly

### Expected Results After Update
- ✅ No more "maximum context length exceeded" errors
- ✅ No more "[Errno 13] Permission denied" errors  
- ✅ Reliable task extraction for emails of all sizes
- ✅ Improved extraction quality with gpt-4o
- ✅ Faster processing and lower API costs
