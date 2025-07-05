# 🚨 Upload Error Troubleshooting Guide

## 403 Forbidden Error Solutions

If you're getting a "403 Forbidden" or "Request failed with status code 403" error:

### Immediate Solutions
1. **Try a smaller file**: Export a shorter date range from Gmail (e.g., last 3-6 months)
2. **Switch browsers**: Chrome and Firefox handle large uploads better than Safari/Edge
3. **Check your connection**: Use a stable, wired internet connection if possible
4. **Disable antivirus**: Temporarily disable real-time file scanning during upload
5. **Clear browser cache**: Clear cache and cookies, then try again

### Advanced Solutions
1. **Split your mbox file**: 
   - Open Terminal/Command Prompt
   - Use `split -b 500m your_inbox.mbox inbox_part_` to create 500MB chunks
   - Upload the first chunk: `inbox_part_aa`

2. **Local installation**:
   ```bash
   git clone [this-repo]
   cd automated-task-manager
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. **File validation**:
   - Open your .mbox file in a text editor
   - Verify it starts with "From " (not "From:")
   - Check file size isn't corrupted (>0 bytes)

### Why 403 Errors Happen
- **Server limits**: Hosting platform restrictions (Hugging Face, etc.)
- **Network security**: Corporate firewalls or proxy servers
- **File content**: Some email content triggers security filters
- **Rate limiting**: Too many upload attempts in short time

### Alternative Approaches
1. **Use Gmail API**: For programmatic access to recent emails
2. **Export smaller ranges**: Use Gmail's export with date filters
3. **Local processing**: Download and run the app on your computer
4. **Contact support**: If none of these work, file an issue on GitHub

---

## Other Common Upload Issues

### Timeout Errors
- **Cause**: Slow internet or very large files
- **Solution**: Use smaller file chunks or upgrade internet connection

### File Format Errors
- **Cause**: Uploading ZIP instead of .mbox
- **Solution**: Extract the ZIP first, then upload `Inbox.mbox`

### Memory Errors
- **Cause**: Insufficient server memory
- **Solution**: Our app automatically limits to 200MB processing

### CORS Errors
- **Cause**: Browser security restrictions
- **Solution**: Try different browser or disable browser security (temporarily)

---

Need more help? Open an issue at: [GitHub Issues](https://github.com/your-repo/issues)
