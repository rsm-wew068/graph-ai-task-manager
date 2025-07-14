# Google Calendar Integration Setup Guide

This guide will help you set up Google Calendar integration for your AI Task Manager.

## 🚀 Quick Start

1. **Enable Google Calendar API**
2. **Create OAuth 2.0 Credentials**
3. **Download and Configure Credentials**
4. **Test the Integration**

## 📋 Step-by-Step Instructions

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter a project name (e.g., "AI Task Manager Calendar")
4. Click "Create"

### Step 2: Enable Google Calendar API

1. In your new project, go to "APIs & Services" → "Library"
2. Search for "Google Calendar API"
3. Click on "Google Calendar API"
4. Click "Enable"

### Step 3: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - User Type: External
   - App name: "AI Task Manager"
   - User support email: Your email
   - Developer contact information: Your email
   - Save and continue through the steps

4. Create OAuth 2.0 Client ID:
   - Application type: Desktop application
   - Name: "AI Task Manager Desktop"
   - Click "Create"

### Step 4: Download Credentials

1. After creating the OAuth client ID, click "Download JSON"
2. Rename the downloaded file to `credentials.json`
3. Place `credentials.json` in your project root directory (same level as `app.py`)

### Step 5: Install Dependencies

The required dependencies are already in `requirements.txt`:

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### Step 6: Test the Integration

1. Run your Streamlit app: `streamlit run app.py`
2. Go to the Calendar page
3. Click "Connect to Google Calendar"
4. Follow the OAuth flow in your browser
5. Grant permissions to access your Google Calendar

## 🔧 Configuration Options

### Calendar Permissions

The app requests the following permissions:
- `https://www.googleapis.com/auth/calendar` - Full access to calendars
- `https://www.googleapis.com/auth/calendar.events` - Full access to events

### Supported Operations

- ✅ View all calendars
- ✅ Import events from Google Calendar
- ✅ Create events in Google Calendar
- ✅ Update existing events
- ✅ Delete events
- ✅ Sync tasks to Google Calendar

## 🎯 Features

### Unified Calendar View
- View both your extracted tasks and Google Calendar events in one place
- Different colors and icons to distinguish between sources
- Click on any event to see detailed information

### Task Synchronization
- Sync individual tasks to Google Calendar
- Automatic date/time handling
- Include task details in event descriptions

### Google Calendar Integration
- Direct access to your Google Calendar
- Import events for unified view
- Embedded Google Calendar view

## 🔒 Security Notes

- `credentials.json` contains your OAuth client ID and secret
- `token.json` contains your access tokens (auto-generated)
- Never commit these files to version control
- Add them to `.gitignore`:

```
credentials.json
token.json
```

## 🐛 Troubleshooting

### "Credentials not found" Error
- Ensure `credentials.json` is in the project root
- Check that the file name is exactly `credentials.json`
- Verify the file contains valid JSON

### "Authentication failed" Error
- Delete `token.json` and try again
- Check your internet connection
- Ensure Google Calendar API is enabled in your project

### "No calendars found" Error
- Verify you have calendars in your Google account
- Check that the OAuth consent screen is configured
- Ensure you granted calendar permissions

### Permission Denied Errors
- Check that your Google account has access to the calendars
- Verify the OAuth scopes are correctly configured
- Try re-authenticating by deleting `token.json`

## 📱 Mobile Access

The Google Calendar integration works on mobile devices through the Streamlit app. You can:
- View your unified calendar
- Sync tasks to Google Calendar
- Access all calendar features

## 🔄 Sync Behavior

### Task to Calendar Sync
- Creates new events in your selected Google Calendar
- Uses task due dates or start dates
- Includes task details in event description
- Preserves task metadata

### Calendar to App Sync
- Imports events from Google Calendar
- Displays them alongside your tasks
- Maintains original event data
- Updates in real-time

## 📊 Usage Tips

1. **Start with Unified View**: See both tasks and calendar events together
2. **Use Different Views**: Switch between unified, tasks-only, and Google-only views
3. **Sync Important Tasks**: Use the sync button to add tasks to your Google Calendar
4. **Import Calendar Events**: Bring your existing calendar events into the app
5. **Embed Calendar**: Use the embed URL feature for direct Google Calendar access

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your Google Cloud Console setup
3. Ensure all dependencies are installed
4. Check the Streamlit logs for error messages
5. Try re-authenticating by deleting `token.json`

## 🔗 Useful Links

- [Google Cloud Console](https://console.cloud.google.com/)
- [Google Calendar API Documentation](https://developers.google.com/calendar/api)
- [OAuth 2.0 Setup Guide](https://developers.google.com/identity/protocols/oauth2)
- [Streamlit Documentation](https://docs.streamlit.io/) 